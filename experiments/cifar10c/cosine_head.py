"""The gap bound for a probe on L2-normalized features.

Usage:  python experiments/cifar10c/cosine_head.py {clip|dinov2}

Every embedding is projected to the unit sphere before the probe is fitted and before
any correction, so the linear head acts on normalized features as a cosine-style head
does, and all displacements are measured in that space.  The script reports, for the
95 shift pairs, the accuracies of no correction, the common-mean, the per-class
predicted and the oracle correctors, and the Theorem 1 bound under the three margin
laws.

Output: results/cifar10c/cosine_head_<enc>.pkl
"""
import pickle
import sys

import numpy as np

from prlb.core import ridge_probe, acc, predict, class_means, theory_bound, ci95
from prlb.paths import CIFAR_EMB, CIFAR_OUT, CORRUPTIONS, ensure

K = 10
ENC = sys.argv[1]
ensure(CIFAR_OUT)


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


Xtr = unit(np.load(CIFAR_EMB / ('%s_train.npy' % ENC)).astype(np.float64))
ytr = np.load(CIFAR_EMB / 'labels_train.npy')
Xs = unit(np.load(CIFAR_EMB / ('%s_clean.npy' % ENC)).astype(np.float64))
ys = np.load(CIFAR_EMB / 'labels_test.npy')
W, b = ridge_probe(Xtr, ytr, K, lam=1e-2)
MS = class_means(Xs, ys, K)
pis = np.bincount(ys, minlength=K) / len(ys)
rows = []
for cn in CORRUPTIONS:
    Z = np.load(CIFAR_EMB / ('%s_%s.npz' % (ENC, cn)))
    for sv in range(1, 6):
        Xt = unit(Z['s%d' % sv].astype(np.float64))
        D = class_means(Xt, ys, K) - MS
        dbar = Xt.mean(0) - Xs.mean(0)
        rho = D - dbar[None, :]
        Xg = Xt - dbar; pl = predict(Xg, W, b)
        r = dict(corr=cn, sev=sv, a_none=acc(Xt, ys, W, b), a_mean=acc(Xg, ys, W, b),
                 a_orc=acc(Xt - D[ys], ys, W, b), a_pc=acc(Xt - D[pl], ys, W, b))
        r['gap'] = r['a_orc'] - r['a_mean']
        r['bound'] = theory_bound(Xs, ys, W, b, rho, pis)[0]
        r['bound_t'] = theory_bound(Xt - D[ys], ys, W, b, rho, pis)[0]
        r['bound_h'] = theory_bound(Xg, pl, W, b, rho, pis)[0]
        rows.append(r)
with open(CIFAR_OUT / ('cosine_head_%s.pkl' % ENC), 'wb') as f:
    pickle.dump(rows, f)
print('=== L2-NORMALIZED HEAD: %s, %d pairs ===' % (ENC, len(rows)))
for k in ('a_none', 'a_mean', 'a_pc', 'a_orc', 'gap', 'bound_t', 'bound_h', 'bound'):
    m, h = ci95([r[k] for r in rows]); print('  %-8s %.5f +- %.5f' % (k, m, h))
for k in ('bound_t', 'bound_h', 'bound'):
    v = sum(1 for r in rows if r[k] < r['gap'] - 1e-12)
    rat = [r[k] / r['gap'] for r in rows if r['gap'] > 1e-4]
    print('  %-8s violations %d/%d  median ratio %.3f' % (k, v, len(rows), np.median(rat)))
d = np.array([r['a_pc'] - r['a_mean'] for r in rows])
print('  per-class predicted minus common-mean: mean %+.5f  max |.| %.5f' % (d.mean(), np.abs(d).max()))
