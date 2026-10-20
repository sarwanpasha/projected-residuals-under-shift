"""Few-shot baselines on the label-budget cells of budget_allocation.py.

Usage:  python experiments/cifar10c/fewshot.py {clip|dinov2}

Reproduces the target samples and the labeled draws of budget_allocation.py and
budget_anchored.py exactly and evaluates two further uses of the budget: weight-space
interpolation between the source probe and the refit from scratch (WiSE-FT style),
and a probe on the leading source-predictive directions (Pro2 style).  Both select
their single hyperparameter by cross-validation on the budget alone.  Counting the
marginal is recomputed on the same draws as a pairing check.

Output: results/cifar10c/fewshot_<enc>.pkl
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
EPS = 1e-3
ENC = sys.argv[1]
RANKS = (10, 20, 40, 80, 160)

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
Q = R2.pro2_basis(Xtr, ytr, K, max(RANKS))

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
                    Ww, bw, th = R2.wise_cv(Xts[lb], yl, K, W, b, np.random.default_rng(B + sd))
                    out['a_wise%d' % B] = acc(Xts, yts, Ww, bw)
                    out['theta%d' % B] = th
                    Wp, bp, rk = R2.pro2_cv(Xts[lb], yl, K, Q, np.random.default_rng(B + sd), RANKS)
                    out['a_pro2%d' % B] = acc(Xts, yts, Wp, bp)
                    out['rank%d' % B] = rk
                rows.append(out)
    print('done %s' % cn, flush=True)

with open(CIFAR_OUT / ('fewshot_%s.pkl' % ENC), 'wb') as f:
    pickle.dump(rows, f)

print('\n=== FEW-SHOT BASELINES: %s, %d cells ===' % (ENC, len(rows)))
for a in ALPHAS:
    s = [r for r in rows if r['alpha'] == a]
    print('\n--- alpha %.1f' % a)
    for B in BUDG:
        print('  B=%-4d count %.5f  wise %.5f  pro2 %.5f  medtheta %.2f  medrank %g'
              % (B, ci95([r['a_cnt%d' % B] for r in s])[0], ci95([r['a_wise%d' % B] for r in s])[0],
                 ci95([r['a_pro2%d' % B] for r in s])[0],
                 float(np.median([r['theta%d' % B] for r in s])),
                 float(np.median([r['rank%d' % B] for r in s]))))
