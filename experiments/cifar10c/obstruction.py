"""Label-free per-class rules beyond the one-step corrector, on CIFAR-10-C (Table 1).

Usage:  python experiments/cifar10c/obstruction.py {clip|dinov2}

For every one of the 95 shift pairs, with the conventions of leaderboard.py, the
script evaluates:
  * the per-hypothesis corrector argmax_j <w_j, x - delta_j> + b_j and the fixed-point
    corrector r <- h(x - delta_r), both with the exact class displacements, which is
    the most favorable case for any label-free per-class rule;
  * the label-free EM per-class translation, which re-estimates the displacements from
    its own pseudo-classes until the assignment is stable;
  * TENT and SHOT in their standard episodic configuration;
  * the consistent-set statistics: the fraction of points whose consistent set
    {k : h(x - delta_k) = k} holds more than one label, and the mass of the
    residual-conflict band 0 < gamma_k(z) <= kappa_{ck}.

Output: results/cifar10c/obstruction_<enc>.pkl
"""
import pickle
import sys
import time

import numpy as np

from prlb.core import (ridge_probe, acc, predict, class_means, per_hypothesis_predict,
                       fixed_point_predict, em_per_class, consistent_sets,
                       conflict_band_mass, tta_standard)
from prlb.paths import CIFAR_EMB, CIFAR_OUT, CORRUPTIONS, ensure

K = 10
SEVS = ['s1', 's2', 's3', 's4', 's5']


def main():
    enc = sys.argv[1]
    ensure(CIFAR_OUT)
    t0 = time.time()
    Xtr = np.load(CIFAR_EMB / ('%s_train.npy' % enc)).astype(np.float64)
    ytr = np.load(CIFAR_EMB / 'labels_train.npy')
    yte = np.load(CIFAR_EMB / 'labels_test.npy')
    W, b = ridge_probe(Xtr, ytr, K, lam=1e-2)
    MS = class_means(Xtr, ytr, K)
    rows = []
    for cn in CORRUPTIONS:
        Z = np.load(CIFAR_EMB / ('%s_%s.npz' % (enc, cn)))
        for sv in SEVS:
            Xt = Z[sv].astype(np.float64)
            D = class_means(Xt, yte, K) - MS
            dbar = Xt.mean(0) - Xtr.mean(0)
            R = D - (np.bincount(yte, minlength=K) / len(yte)) @ D
            r0 = predict(Xt - dbar, W, b)
            r = dict(corr=cn, sev=sv, enc=enc)
            r['naive'] = acc(Xt, yte, W, b)
            r['mean'] = acc(Xt - dbar, yte, W, b)
            r['oracle'] = acc(Xt - D[yte], yte, W, b)
            r['pc_pred'] = float(np.mean(predict(Xt - D[r0], W, b) == yte))
            r['ph_exact'] = float(np.mean(per_hypothesis_predict(Xt, W, b, D) == yte))
            fp, conv, it = fixed_point_predict(Xt, W, b, D, r0)
            r['fp_exact'] = float(np.mean(fp == yte)); r['fp_conv'] = conv; r['fp_iters'] = it
            r['em'] = float(np.mean(em_per_class(Xt, W, b, MS, dbar) == yte))
            F = consistent_sets(Xt, W, b, D)
            inF = F[np.arange(len(yte)), yte]
            r['amb'] = float(np.mean(F.sum(1) >= 2))
            r['amb_correct'] = float(np.mean(inF & (F.sum(1) >= 2)))
            r['band'] = conflict_band_mass(Xt, yte, W, b, D, R)
            for nm in ('tent', 'shot'):
                try:
                    r[nm] = acc(tta_standard(Xt, W, b, mode=nm), yte, W, b)
                except ImportError:
                    r[nm] = float('nan')
            rows.append(r)
        print('  %-20s %6.1f s' % (cn, time.time() - t0), flush=True)
    with open(CIFAR_OUT / ('obstruction_%s.pkl' % enc), 'wb') as f:
        pickle.dump(rows, f)
    keys = ['naive', 'mean', 'pc_pred', 'ph_exact', 'fp_exact', 'em', 'tent', 'shot',
            'oracle', 'fp_conv', 'amb', 'amb_correct', 'band']
    print('\n=== %s, %d pairs: mean and 95 pct half-width ===' % (enc, len(rows)))
    for k in keys:
        v = np.array([x[k] for x in rows], float); v = v[np.isfinite(v)]
        print('  %-12s %.5f +- %.5f' % (k, v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(v.size)))
    for k in ('pc_pred', 'ph_exact', 'fp_exact', 'em'):
        d = np.array([x[k] - x['mean'] for x in rows])
        print('  %-12s minus mean: %+.5f  min %+.5f  max %+.5f' % (k, d.mean(), d.min(), d.max()))
    g = np.array([x['oracle'] - x['fp_exact'] for x in rows]); bm = np.array([x['band'] for x in rows])
    print('  oracle minus fixed point %.5f, conflict band %.5f, band >= shortfall on %d/%d pairs'
          % (g.mean(), bm.mean(), int(np.sum(bm >= g - 1e-12)), len(rows)))


if __name__ == '__main__':
    main()
