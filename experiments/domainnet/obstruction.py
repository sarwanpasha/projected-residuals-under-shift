"""Label-free per-class rules beyond the one-step corrector, on DomainNet (Table 1).

Usage:  python experiments/domainnet/obstruction.py {clip|dino} <K list> [tag]

Uses the probes and the 4000-point target subsamples of tta.py and evaluates, for
every ordered pair, the per-hypothesis and fixed-point correctors with the exact
class displacements, the label-free EM per-class translation, TENT and SHOT in their
standard episodic configuration, and the consistent-set statistics.

The optional tag (default all) names the output file, so that class counts can run
as separate jobs.

Output: results/domainnet/dnet_obstruction_<enc>_<tag>.pkl
"""
import pickle
import sys

import numpy as np

from prlb import core as R
from prlb.paths import DNET_EMB, DNET_OUT, DOMAINS, ensure

ENC = sys.argv[1]
KS = [int(x) for x in sys.argv[2].split(',')]
TAG = sys.argv[3] if len(sys.argv) > 3 else 'all'
NSUB = 4000
SEED = 0


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
        PR = {}
        for S in DOMAINS:
            Xs, ys = prep(S, keep, cmap)
            idx = np.random.default_rng(SEED).permutation(len(ys))
            ntr = int(0.6 * len(ys))
            Xtr = Xs[idx[:ntr]]; ytr = ys[idx[:ntr]]
            W, b = R.ridge_probe(Xtr, ytr, K)
            PR[S] = (Xtr, ytr, W, b)
        for si, S in enumerate(DOMAINS):
            Xtr, ytr, W, b = PR[S]
            MS = R.class_means(Xtr, ytr, K)
            for ti, Tg in enumerate(DOMAINS):
                if Tg == S:
                    continue
                Xf, yf = prep(Tg, keep, cmap)
                g = np.random.default_rng(1000 * K + 10 * si + ti)
                sel = g.permutation(len(yf))[:min(NSUB, len(yf))]
                Xt = Xf[sel].astype(np.float64); yt = yf[sel]
                Dl = R.class_means(Xt, yt, K) - MS
                dbar = Xt.mean(0) - Xtr.mean(0)
                pit = np.bincount(yt, minlength=K) / len(yt)
                Rr = Dl - pit @ Dl
                r0 = R.predict(Xt - dbar, W, b)
                r = dict(K=K, enc=ENC, src=S, tgt=Tg, n=int(len(yt)))
                r['naive'] = R.acc(Xt, yt, W, b)
                r['mean'] = R.acc(Xt - dbar, yt, W, b)
                r['oracle'] = R.acc(Xt - Dl[yt], yt, W, b)
                r['pc_pred'] = float(np.mean(R.predict(Xt - Dl[r0], W, b) == yt))
                r['ph_exact'] = float(np.mean(R.per_hypothesis_predict(Xt, W, b, Dl) == yt))
                fp, conv, it = R.fixed_point_predict(Xt, W, b, Dl, r0)
                r['fp_exact'] = float(np.mean(fp == yt)); r['fp_conv'] = conv; r['fp_iters'] = it
                r['em'] = float(np.mean(R.em_per_class(Xt, W, b, MS, dbar) == yt))
                F = R.consistent_sets(Xt, W, b, Dl)
                inF = F[np.arange(len(yt)), yt]
                r['amb'] = float(np.mean(F.sum(1) >= 2))
                r['amb_correct'] = float(np.mean(inF & (F.sum(1) >= 2)))
                r['band'] = R.conflict_band_mass(Xt, yt, W, b, Dl, Rr)
                for nm in ('tent', 'shot'):
                    try:
                        r[nm] = R.acc(R.tta_standard(Xt, W, b, mode=nm), yt, W, b)
                    except ImportError:
                        r[nm] = float('nan')
                rows.append(r)
                print('%d %s %s->%s mean %.4f orc %.4f ph %.4f fp %.4f em %.4f tent %.4f shot %.4f amb %.4f band %.4f'
                      % (K, ENC, S, Tg, r['mean'], r['oracle'], r['ph_exact'], r['fp_exact'],
                         r['em'], r['tent'], r['shot'], r['amb'], r['band']), flush=True)
    with open(DNET_OUT / ('dnet_obstruction_%s_%s.pkl' % (ENC, TAG)), 'wb') as f:
        pickle.dump(rows, f)
    keys = ['naive', 'mean', 'pc_pred', 'ph_exact', 'fp_exact', 'em', 'tent', 'shot',
            'oracle', 'fp_conv', 'amb', 'amb_correct', 'band']
    for K in KS:
        rr = [x for x in rows if x['K'] == K]
        print('\n=== %s K=%d, %d pairs ===' % (ENC, K, len(rr)))
        for k in keys:
            v = np.array([x[k] for x in rr], float); v = v[np.isfinite(v)]
            if v.size > 1:
                print('  %-12s %.5f +- %.5f' % (k, v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(v.size)))


if __name__ == '__main__':
    main()
