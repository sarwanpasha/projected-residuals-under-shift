"""Supervised target references, tuned TENT and SHOT, and label budgets under natural
label drift on DomainNet.

Usage:  python experiments/domainnet/reference.py {clip|dino} <K list> [tag]

Uses the probes and the 4000-point target subsamples of tta.py and obstruction.py.
Up to 20000 target points outside the subsample form a labeled pool that is never
scored.
For every ordered pair the script reports

  * the target ridge probe (penalty by cross-validation), the probe refitted toward
    the source probe, and a 20-nearest-neighbour classifier, each fitted on the
    pool and scored on the subsample;
  * TENT and SHOT at every configuration of prlb.core.TTA_GRID, so that a
    configuration can be chosen on the pairs with a different target domain;
  * five uses of a budget of B labels drawn uniformly from the pool, at the natural
    label drift of the pair: counting the marginal, the refit from scratch, the refit
    anchored to the source probe, weight-space interpolation and the Pro2-style
    probe, all scored on the subsample.

Output: results/domainnet/dnet_reference_<enc>_<tag>.pkl
"""
import pickle
import sys

import numpy as np

from prlb import core as R
from prlb import extra as R2
from prlb.paths import DNET_EMB, DNET_OUT, DOMAINS, ensure

ENC = sys.argv[1]
KS = [int(x) for x in sys.argv[2].split(',')]
TAG = sys.argv[3] if len(sys.argv) > 3 else 'all'
NSUB = 4000
NPOOL = 20000
SEED = 0
BUDG = [200, 1000, 5000]
LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
EPS = 1e-3


def knn_predict(Xa, ya, Xb, K, k=20, chunk=1000):
    k = min(k, len(Xa) - 1)
    A = (Xa / np.linalg.norm(Xa, axis=1, keepdims=True)).astype(np.float32)
    out = []
    for i in range(0, len(Xb), chunk):
        Bq = Xb[i:i + chunk]; Bq = (Bq / np.linalg.norm(Bq, axis=1, keepdims=True)).astype(np.float32)
        S = Bq @ A.T
        nn = np.argpartition(-S, k, axis=1)[:, :k]
        V = np.zeros((len(Bq), K))
        np.add.at(V, (np.repeat(np.arange(len(Bq)), k), ya[nn].ravel()), 1.0)
        out.append(V.argmax(1))
    return np.concatenate(out)


def load():
    D = {}
    for d in DOMAINS:
        X = np.load(DNET_EMB / ('%s_%s.npy' % (d, ENC)))
        y = np.load(DNET_EMB / ('%s_y.npy' % d)).astype(np.int64)
        ok = np.load(DNET_EMB / ('%s_ok.npy' % d)).astype(bool)
        D[d] = (X[ok], y[ok])
    return D


def main():
    ensure(DNET_OUT)
    D = load()
    KG = len((DNET_EMB / 'classes.txt').read_text().split())
    cnt = np.zeros(KG, np.int64)
    for d in DOMAINS:
        cnt += np.bincount(D[d][1], minlength=KG)
    RANK = np.argsort(-cnt)

    def prep(dom, keep, cmap):
        X, y = D[dom]
        m = np.isin(y, keep)
        return X[m].astype(np.float32), cmap[y[m]]

    rows = []
    for K in KS:
        keep = RANK[:K]
        cmap = -np.ones(KG, np.int64); cmap[keep] = np.arange(K)
        ranks = sorted(set(r for r in (16, 32, 64, 128, 256) if r <= 2 * K) | {K - 1})
        PR = {}
        for S in DOMAINS:
            Xs, ys = prep(S, keep, cmap)
            idx = np.random.default_rng(SEED).permutation(len(ys))
            ntr = int(0.6 * len(ys))
            Xtr = Xs[idx[:ntr]].astype(np.float64); ytr = ys[idx[:ntr]]
            W, b = R.ridge_probe(Xtr, ytr, K)
            Q = R2.pro2_basis(Xtr, ytr, K, max(ranks))
            PR[S] = (Xtr, ytr, W, b, Q)
        for si, S in enumerate(DOMAINS):
            Xtr, ytr, W, b, Q = PR[S]
            MS = R.class_means(Xtr, ytr, K)
            pi_s = np.bincount(ytr, minlength=K) / len(ytr)
            for ti, Tg in enumerate(DOMAINS):
                if Tg == S:
                    continue
                Xf, yf = prep(Tg, keep, cmap)
                g = np.random.default_rng(1000 * K + 10 * si + ti)
                perm = g.permutation(len(yf))
                sel = perm[:min(NSUB, len(yf))]; pool = perm[min(NSUB, len(yf)):min(NSUB, len(yf)) + NPOOL]
                Xt = Xf[sel].astype(np.float64); yt = yf[sel]
                Xp = Xf[pool].astype(np.float64); yp = yf[pool]
                Dl = R.class_means(Xt, yt, K) - MS
                dbar = Xt.mean(0) - Xtr.mean(0)
                mu_t = Xt.mean(0)
                pit = np.bincount(yt, minlength=K) / len(yt)
                r = dict(K=K, enc=ENC, src=S, tgt=Tg, n=int(len(yt)), npool=int(len(yp)),
                         drift=float(np.abs(pit - pi_s).sum()))
                r['naive'] = R.acc(Xt, yt, W, b)
                r['mean'] = R.acc(Xt - dbar, yt, W, b)
                r['oracle'] = R.acc(Xt - Dl[yt], yt, W, b)
                Wt, bt, lt = R.ridge_cv(Xp, yp, K)
                r['tprobe'] = R.acc(Xt, yt, Wt, bt); r['tprobe_lam'] = lt
                Wa, ba, la = R2.ridge_to_source_cv(Xp, yp, K, W, b, LAMS, np.random.default_rng(ti))
                r['tanch'] = R.acc(Xt, yt, Wa, ba); r['tanch_lam'] = la
                r['tknn'] = float(np.mean(knn_predict(Xp, yp, Xt, K) == yt))
                r['tbest'] = max(r['tprobe'], r['tanch'], r['tknn'])
                r['tent_grid'] = R.tta_grid(Xt, yt, W, b, 'tent')
                r['shot_grid'] = R.tta_grid(Xt, yt, W, b, 'shot')
                bg = np.random.default_rng(7 * K + 100 * si + ti)
                for B in BUDG:
                    lb = bg.choice(len(yp), size=min(B, len(yp)), replace=False)
                    Xl, yl = Xp[lb], yp[lb]
                    cn = np.bincount(yl, minlength=K).astype(float)
                    pv = (cn + EPS) / (cn + EPS).sum()
                    r['empty%d' % B] = float(np.mean(cn == 0))
                    r['cnt%d' % B] = R.acc(Xt - (mu_t - pv @ MS), yt, W, b)
                    Wr, br, _ = R.ridge_cv(Xl, yl, K)
                    r['scratch%d' % B] = R.acc(Xt, yt, Wr, br)
                    Wn, bn, ln = R2.ridge_to_source_cv(Xl, yl, K, W, b, LAMS, np.random.default_rng(B))
                    r['anch%d' % B] = R.acc(Xt, yt, Wn, bn); r['anch_lam%d' % B] = ln
                    Ww, bw, th = R2.wise_cv(Xl, yl, K, W, b, np.random.default_rng(B))
                    r['wise%d' % B] = R.acc(Xt, yt, Ww, bw); r['theta%d' % B] = th
                    Wp, bp, rk = R2.pro2_cv(Xl, yl, K, Q, np.random.default_rng(B), ranks)
                    r['pro2%d' % B] = R.acc(Xt, yt, Wp, bp); r['rank%d' % B] = rk
                rows.append(r)
                print('%d %s %s->%s mean %.4f orc %.4f tprobe %.4f tanch %.4f tknn %.4f | %s'
                      % (K, ENC, S, Tg, r['mean'], r['oracle'], r['tprobe'], r['tanch'], r['tknn'],
                         ' '.join('B%d cnt %.4f anch %.4f' % (B, r['cnt%d' % B], r['anch%d' % B])
                                  for B in BUDG)), flush=True)
    with open(DNET_OUT / ('dnet_reference_%s_%s.pkl' % (ENC, TAG)), 'wb') as f:
        pickle.dump(rows, f)
    for K in KS:
        rr = [x for x in rows if x['K'] == K]
        print('\n=== %s K=%d, %d pairs ===' % (ENC, K, len(rr)))
        keys = ['naive', 'mean', 'oracle', 'tprobe', 'tanch', 'tknn', 'tbest']
        for B in BUDG:
            keys += ['empty%d' % B, 'cnt%d' % B, 'scratch%d' % B, 'anch%d' % B, 'wise%d' % B, 'pro2%d' % B]
        for k in keys:
            m, h = R.ci95([x[k] for x in rr])
            print('  %-12s %.5f +- %.5f' % (k, m, h))
        for B in BUDG:
            print('  anch-cnt B=%d %+.5f +- %.5f' % ((B,) + tuple(R.ci95([x['anch%d' % B] - x['cnt%d' % B] for x in rr]))))
        for nm in ('tent', 'shot'):
            v = R.select_held_out(rr, nm + '_grid', 'tgt')
            print('  %s held-out selection %.5f' % (nm, v.mean()))


if __name__ == '__main__':
    main()
