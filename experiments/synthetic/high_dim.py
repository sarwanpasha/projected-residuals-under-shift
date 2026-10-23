"""The gap bound in high dimension, with anisotropic spectra and heavy tails.

Dimensions d in {50, 256, 512, 1024}, an isotropic or power-law (alpha = 1) spectrum,
and Gaussian or multivariate-t tails (5 or 3 degrees of freedom), at residual angles of
0, 45 and 90 degrees with 15 seeds each: 720 cells in total.  The script reports the
measured gap, the bound and the number of violations per configuration.

Output: results/synthetic/high_dim.pkl
"""
import numpy as np, pickle
from prlb.core import *
from prlb.paths import SYN_OUT, ensure

ensure(SYN_OUT)

K = 8; NPC = 2500; NTR = 1250; SEP = 4.0; RNORM = 2.0; DBAR_NORM = 5.0
CONFIGS = []
for d in [50, 256, 512, 1024]:
    for alpha, df in [(0.0, None), (1.0, None), (0.0, 5), (1.0, 3)]:
        CONFIGS.append((d, alpha, df))
SEEDS = list(range(15))
rows = []
for (d, alpha, df) in CONFIGS:
    for sd in SEEDS:
        rng = np.random.default_rng(1000 + sd)
        X, y, _, _ = make_source(rng, d=d, K=K, n=NPC, sep=SEP, alpha=alpha, df=df)
        Xtr, ytr, Xev, yev = split_by_class(X, y, K, NTR, rng)
        W, b = ridge_probe(Xtr, ytr, K)
        pis = np.full(K, 1.0 / K)
        g = np.random.default_rng(3000 + sd).standard_normal(d)
        dbar = g / np.linalg.norm(g) * DBAR_NORM
        for th in [0.0, 45.0, 90.0]:
            rho = make_residuals(np.random.default_rng(4000 + sd), W, pis, RNORM, np.deg2rad(th))
            deltas = dbar[None, :] + rho
            Xt = Xev + deltas[yev]
            a_or = acc(Xt - deltas[yev], yev, W, b)
            a_gl = acc(Xt - dbar, yev, W, b)
            bnd, G, _ = theory_bound(Xev, yev, W, b, rho, pis)
            rows.append(dict(d=d, alpha=alpha, df=(0 if df is None else df), th=th, seed=sd,
                             gap=a_or - a_gl, bound=bnd, a_or=a_or, a_gl=a_gl))
with open(SYN_OUT / 'high_dim.pkl', 'wb') as f:
    pickle.dump(rows, f)
print('HIGH-DIM / ANISOTROPIC / HEAVY-TAILED   K=%d seeds=%d' % (K, len(SEEDS)))
print('%5s %6s %4s %6s %10s %18s %18s %8s %8s' % ('d', 'alpha', 'df', 'theta', 'accOracle', 'measured gap', 'Thm1 bound', 'tight', 'viol'))
for (d, alpha, df) in CONFIGS:
    for th in [0.0, 45.0, 90.0]:
        R = [r for r in rows if r['d'] == d and r['alpha'] == alpha and r['df'] == (0 if df is None else df) and r['th'] == th]
        gm, gc = ci95([r['gap'] for r in R]); bm, bc = ci95([r['bound'] for r in R])
        am, _ = ci95([r['a_or'] for r in R])
        v = sum(1 for r in R if r['bound'] < r['gap'] - 1e-12)
        tt = bm / gm if gm > 1e-9 else float('nan')
        print('%5d %6.1f %4d %6.0f %10.4f %8.5f +- %-7.5f %8.5f +- %-7.5f %8.2f %8d'
              % (d, alpha, (0 if df is None else df), th, am, gm, gc, bm, bc, tt, v))
print('TOTAL_VIOLATIONS %d / %d' % (sum(1 for r in rows if r['bound'] < r['gap'] - 1e-12), len(rows)))
