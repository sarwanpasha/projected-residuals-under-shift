"""Transductive prototype shifting against feature translation, on the zero-shot CLIP
head and on the source probe.

Usage:  python experiments/zeroshot/tps.py cifar
        python experiments/zeroshot/tps.py domainnet <K list> [tag]

Two heads are evaluated on the same CLIP ViT-B/32 features.  The zero-shot head scores
a point by its cosine similarity to the text prototypes of data_prep/text_embed.py,
with the CLIP logit scale of 100.  The probe head is the source ridge probe.  On each
head the script measures

  * the feature-translation rules of the paper: no correction, the common-mean
    correction, the fixed-point rule with the exact class displacements, and the
    per-class oracle;
  * transductive prototype shifting: one shift vector per class added to the class
    prototypes, or one shift shared by all classes, learned over the target set by
    one pass of Adam on batches of 64 with the information-maximization objective
    (mean prediction entropy minus the entropy of the mean prediction), which does
    not collapse to a single class as entropy alone does.  The episodic protocol of
    TPS with augmented views needs the images and is run by tps_views.py.

Every learning rate of LRS is stored so that a rate can be selected on held-out
shifts.

Output: results/zeroshot/tps_cifar.pkl or results/zeroshot/tps_domainnet_<tag>.pkl
"""
import pickle
import sys

import numpy as np
import torch

from prlb import core as R
from prlb.paths import CIFAR_EMB, CORRUPTIONS, DNET_EMB, DOMAINS, RESULTS, ensure

OUT = RESULTS / 'zeroshot'
LRS = (1e-5, 1e-4, 1e-3)
SCALE = 100.0
NSUB = 4000
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def nrm(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


class Head:
    """Scores of a cosine head (prototypes P, normalized features) or of a linear head
    (weights P, bias b)."""

    def __init__(self, P, b=None, cosine=True):
        self.P = P; self.b = b; self.cos = cosine

    def scores(self, X):
        if self.cos:
            return SCALE * nrm(X) @ nrm(self.P).T
        return X @ self.P.T + self.b

    def predict(self, X):
        return self.scores(X).argmax(1)


def fixed_point(h, X, D, r0, iters=50):
    r = r0.copy()
    for _ in range(iters):
        rn = np.empty_like(r)
        for k in np.unique(r):
            m = r == k
            rn[m] = h.predict(X[m] - D[k])
        if np.array_equal(rn, r):
            break
        r = rn
    pred = np.empty_like(r)
    for k in np.unique(r):
        m = r == k
        pred[m] = h.predict(X[m] - D[k])
    return pred


def torch_scores(h, Xt, P):
    if h.cos:
        Pn = P / P.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return SCALE * Xt @ Pn.T if Pn.dim() == 2 else SCALE * torch.einsum('nd,nkd->nk', Xt, Pn)
    b = torch.tensor(h.b, dtype=Xt.dtype, device=dev)
    return (Xt @ P.T if P.dim() == 2 else torch.einsum('nd,nkd->nk', Xt, P)) + b


def entropy(S):
    lp = torch.log_softmax(S, 1)
    return -(lp.exp() * lp).sum(1)


def tps_batch(h, X, lr, shared, seed=0):
    """Transductive prototype shift: one pass of Adam over shuffled batches of 64 on
    the information-maximization objective."""
    torch.manual_seed(seed)
    Xt = torch.tensor(nrm(X) if h.cos else X, dtype=torch.float32, device=dev)
    P0 = torch.tensor(h.P, dtype=torch.float32, device=dev)
    K, d = P0.shape
    s = torch.zeros((1 if shared else K, d), dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([s], lr=lr)
    perm = np.random.default_rng(seed).permutation(len(X))
    for i in range(0, len(X), 64):
        idx = torch.tensor(perm[i:i + 64], device=dev)
        S = torch_scores(h, Xt[idx], P0 + s)
        p = torch.softmax(S, 1).mean(0)
        L = entropy(S).mean() + (p * torch.log(p + 1e-12)).sum()
        opt.zero_grad(); L.backward(); opt.step()
    with torch.no_grad():
        return torch_scores(h, Xt, P0 + s).argmax(1).cpu().numpy()


def evaluate(h, Xt, yt, D, dbar):
    r = {}
    r['naive'] = float(np.mean(h.predict(Xt) == yt))
    r['mean'] = float(np.mean(h.predict(Xt - dbar) == yt))
    r['oracle'] = float(np.mean(h.predict(Xt - D[yt]) == yt))
    r['fp_exact'] = float(np.mean(fixed_point(h, Xt, D, h.predict(Xt - dbar)) == yt))
    for shared in (False, True):
        key = 'tpsb' + ('_shared' if shared else '_class')
        r[key] = {lr: float(np.mean(tps_batch(h, Xt, lr, shared) == yt)) for lr in LRS}
    return r


def run_cifar():
    Xtr = np.load(CIFAR_EMB / 'clip_train.npy').astype(np.float64)
    ytr = np.load(CIFAR_EMB / 'labels_train.npy')
    yte = np.load(CIFAR_EMB / 'labels_test.npy')
    T = np.load(CIFAR_EMB / 'clip_text.npy').astype(np.float64)
    K = 10
    W, b = R.ridge_probe(Xtr, ytr, K, lam=1e-2)
    MS = R.class_means(Xtr, ytr, K)
    heads = {'zs': Head(T), 'probe': Head(W, b, cosine=False)}
    Xcl = np.load(CIFAR_EMB / 'clip_clean.npy').astype(np.float64)
    print('clean zero-shot %.4f probe %.4f' % (np.mean(heads['zs'].predict(Xcl) == yte),
                                               np.mean(heads['probe'].predict(Xcl) == yte)), flush=True)
    rows = []
    for cn in CORRUPTIONS:
        Z = np.load(CIFAR_EMB / ('clip_%s.npz' % cn))
        for sv in range(1, 6):
            Xt = Z['s%d' % sv].astype(np.float64)
            D = R.class_means(Xt, yte, K) - MS
            dbar = Xt.mean(0) - Xtr.mean(0)
            for hn, h in heads.items():
                r = dict(corr=cn, sev=sv, head=hn, group=cn)
                r.update(evaluate(h, Xt, yte, D, dbar))
                rows.append(r)
                print('%s s%d %s naive %.4f mean %.4f orc %.4f fp %.4f tpsb_class %s tpsb_shared %s'
                      % (cn, sv, hn, r['naive'], r['mean'], r['oracle'], r['fp_exact'],
                         r['tpsb_class'], r['tpsb_shared']), flush=True)
    return rows


def run_domainnet(KS):
    D_ = {}
    for d in DOMAINS:
        X = np.load(DNET_EMB / ('%s_clip.npy' % d))
        y = np.load(DNET_EMB / ('%s_y.npy' % d)).astype(np.int64)
        ok = np.load(DNET_EMB / ('%s_ok.npy' % d)).astype(bool)
        D_[d] = (X[ok], y[ok])
    TG = np.load(DNET_EMB / 'clip_text.npy').astype(np.float64)
    KG = TG.shape[0]
    cnt = np.zeros(KG, np.int64)
    for d in DOMAINS:
        cnt += np.bincount(D_[d][1], minlength=KG)
    RANK = np.argsort(-cnt)
    rows = []
    for K in KS:
        keep = RANK[:K]
        cmap = -np.ones(KG, np.int64); cmap[keep] = np.arange(K)

        def prep(dom):
            X, y = D_[dom]
            m = np.isin(y, keep)
            return X[m].astype(np.float32), cmap[y[m]]

        PR = {}
        for S in DOMAINS:
            Xs, ys = prep(S)
            idx = np.random.default_rng(0).permutation(len(ys))
            ntr = int(0.6 * len(ys))
            Xtr = Xs[idx[:ntr]].astype(np.float64); ytr = ys[idx[:ntr]]
            W, b = R.ridge_probe(Xtr, ytr, K)
            PR[S] = (Xtr, ytr, W, b)
        zs = Head(TG[keep])
        for si, S in enumerate(DOMAINS):
            Xtr, ytr, W, b = PR[S]
            MS = R.class_means(Xtr, ytr, K)
            heads = {'zs': zs, 'probe': Head(W, b, cosine=False)}
            for ti, Tg in enumerate(DOMAINS):
                if Tg == S:
                    continue
                Xf, yf = prep(Tg)
                g = np.random.default_rng(1000 * K + 10 * si + ti)
                sel = g.permutation(len(yf))[:min(NSUB, len(yf))]
                Xt = Xf[sel].astype(np.float64); yt = yf[sel]
                Dl = R.class_means(Xt, yt, K) - MS
                dbar = Xt.mean(0) - Xtr.mean(0)
                for hn, h in heads.items():
                    r = dict(K=K, src=S, tgt=Tg, head=hn, group=Tg)
                    r.update(evaluate(h, Xt, yt, Dl, dbar))
                    rows.append(r)
                    print('%d %s->%s %s naive %.4f mean %.4f orc %.4f fp %.4f tpsb_class %s tpsb_shared %s'
                          % (K, S, Tg, hn, r['naive'], r['mean'], r['oracle'], r['fp_exact'],
                             r['tpsb_class'], r['tpsb_shared']), flush=True)
    return rows


def report(rows, groups):
    keys = ['tpsb_class', 'tpsb_shared']
    for gk in groups:
        for hn in ('zs', 'probe'):
            s = [r for r in rows if r['head'] == hn and all(r.get(k) == v for k, v in gk.items())]
            if not s:
                continue
            line = '%s %s n=%d' % (gk, hn, len(s))
            for k in ('naive', 'mean', 'fp_exact', 'oracle'):
                line += '  %s %.4f' % (k, np.mean([r[k] for r in s]))
            for k in keys:
                sel = []
                for r in s:
                    oth = [q for q in s if q['group'] != r['group']]
                    lr = max(LRS, key=lambda l: np.mean([q[k][l] for q in oth]))
                    sel.append(r[k][lr])
                line += '  %s %.4f' % (k, np.mean(sel))
            print(line, flush=True)


def main():
    ensure(OUT)
    if sys.argv[1] == 'cifar':
        rows = run_cifar()
        name = 'tps_cifar.pkl'
        groups = [{}]
    else:
        KS = [int(x) for x in sys.argv[2].split(',')]
        tag = sys.argv[3] if len(sys.argv) > 3 else 'all'
        rows = run_domainnet(KS)
        name = 'tps_domainnet_%s.pkl' % tag
        groups = [{'K': K} for K in KS]
    with open(OUT / name, 'wb') as f:
        pickle.dump(rows, f)
    print('\n=== SUMMARY, learning rate selected on the other shifts ===')
    report(rows, groups)


if __name__ == '__main__':
    main()
