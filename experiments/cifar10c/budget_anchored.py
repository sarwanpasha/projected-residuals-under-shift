"""Source-anchored Route A on the label-budget cells of budget_allocation.py.

Usage:  python experiments/cifar10c/budget_anchored.py {clip|dinov2} [budgets tag]

Reproduces the target samples and the labeled draws of budget_allocation.py exactly
and adds the refit that is shrunk toward the source probe,
argmin_T ||Z T - Y||^2 / B + lam ||T - T_S||^2, with lam chosen by five-fold
cross-validation on the budget alone.  Counting the marginal is recomputed on the same
draws so that the two arms are paired.  An optional comma-separated budget list and
tag run a different budget grid; the labeled draws then follow that grid.

Output: results/cifar10c/budget_anchored_<enc>.pkl, or budget_anchored_<enc>_<tag>.pkl
"""
import pickle
import sys

import numpy as np

from prlb import extra as R2
from prlb.core import ridge_probe, acc, class_means, ci95
from prlb.paths import CIFAR_EMB, CIFAR_OUT, CORRUPTIONS, ensure

K = 10
SEVS = [1, 3, 5]
ALPHAS = [0.0, 0.4, 0.8, 1.0]
SEEDS = [0, 1, 2]
BUDG = [50, 100, 200, 400, 800]
LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
EPS = 1e-3
ENC = sys.argv[1]
TAG = ''
if len(sys.argv) > 3:
    BUDG = [int(x) for x in sys.argv[2].split(',')]
    TAG = '_' + sys.argv[3]

ensure(CIFAR_OUT)
Xtr = np.load(CIFAR_EMB / ('%s_train.npy' % ENC)).astype(np.float64)
ytr = np.load(CIFAR_EMB / 'labels_train.npy')
Xs = np.load(CIFAR_EMB / ('%s_clean.npy' % ENC)).astype(np.float64)
ys = np.load(CIFAR_EMB / 'labels_test.npy')
sc = np.linalg.norm(Xtr, axis=1).mean()
Xtr = Xtr / sc; Xs = Xs / sc
W, b = ridge_probe(Xtr, ytr, K, lam=1e-2)
MS = class_means(Xs, ys, K)
pi_s = np.bincount(ys, minlength=K) / len(ys)
idx_by_c = [np.where(ys == c)[0] for c in range(K)]

rows = []
for cn in CORRUPTIONS:
    Z = np.load(CIFAR_EMB / ('%s_%s.npz' % (ENC, cn)))
    for sv in SEVS:
        Xt = Z['s%d' % sv].astype(np.float64) / sc
        for a in ALPHAS:
            for sd in SEEDS:
                rng = np.random.default_rng(10000 * sd + 17 * sv + sum(ord(ch) for ch in cn) % 997)
                pk = rng.dirichlet(np.full(K, 0.5))
                pk = np.maximum(pk, 0.01); pk = pk / pk.sum()
                pit = (1.0 - a) * pi_s + a * pk; pit = pit / pit.sum()
                M = int(min(8000, np.floor((np.array([len(i) for i in idx_by_c]) / pit).min())))
                nc = np.maximum((pit * M).astype(int), 5)
                sel = np.concatenate([rng.choice(idx_by_c[c], size=nc[c], replace=False)
                                      for c in range(K)])
                Xts = Xt[sel]; yts = ys[sel]; n = len(sel)
                mu_t = Xts.mean(0)
                out = dict(corr=cn, sev=sv, alpha=a, seed=sd, n=n,
                           drift=float(np.abs(nc / nc.sum() - pi_s).sum()),
                           a_none=acc(Xts, yts, W, b),
                           a_naive=acc(Xts - (mu_t - pi_s @ MS), yts, W, b))
                for B in BUDG:
                    lb = rng.choice(n, size=min(B, n), replace=False)
                    yl = yts[lb]
                    cnts = np.bincount(yl, minlength=K).astype(float)
                    pv = (cnts + EPS) / (cnts + EPS).sum()
                    out['a_cnt%d' % B] = acc(Xts - (mu_t - pv @ MS), yts, W, b)
                    Wa, ba, lm = R2.ridge_to_source_cv(Xts[lb], yl, K, W, b, LAMS,
                                                       np.random.default_rng(B + sd))
                    out['a_anch%d' % B] = acc(Xts, yts, Wa, ba)
                    out['lam%d' % B] = lm
                rows.append(out)
    print('done %s' % cn, flush=True)

with open(CIFAR_OUT / ('budget_anchored_%s%s.pkl' % (ENC, TAG)), 'wb') as f:
    pickle.dump(rows, f)

print('\n=== SOURCE-ANCHORED ROUTE A: %s, %d cells ===' % (ENC, len(rows)))
for a in ALPHAS:
    s = [r for r in rows if r['alpha'] == a]
    print('\n--- drift %.4f  none %.5f  naive %.5f' % (ci95([r['drift'] for r in s])[0],
          ci95([r['a_none'] for r in s])[0], ci95([r['a_naive'] for r in s])[0]))
    for B in BUDG:
        d = ci95([r['a_cnt%d' % B] - r['a_anch%d' % B] for r in s])
        dn = ci95([r['a_cnt%d' % B] - r['a_none'] for r in s])
        print('  B=%-4d count %.5f  anchored %.5f  count-anchored %+.5f+-%.5f  count-none %+.5f+-%.5f  medlam %g'
              % (B, ci95([r['a_cnt%d' % B] for r in s])[0], ci95([r['a_anch%d' % B] for r in s])[0],
                 d[0], d[1], dn[0], dn[1], float(np.median([r['lam%d' % B] for r in s]))))
