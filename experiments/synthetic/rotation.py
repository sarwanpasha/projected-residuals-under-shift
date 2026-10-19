"""Rotation sweep of the residual against the discriminative subspace (Figure 1).

Gaussian classes in d = 50 with K = 8, a common displacement of norm 5 and residuals
of RMS norm 2 whose angle theta to D = span{w_j - w_k} is swept from 0 to 90 degrees,
30 seeds per angle.  For every cell the script measures the accuracy gap between the
per-class oracle correction and the common-mean correction, evaluates the indicator
bound of Theorem 1, and evaluates three comparators: the naive projected norm, a
Chebyshev bound and a sub-Gaussian bound on the margin tail.

Output: results/synthetic/rotation.pkl
"""
import numpy as np, pickle
from prlb.core import *
from prlb.paths import SYN_OUT, ensure

ensure(SYN_OUT)

D = 50; K = 8; NPC = 3000; NTR = 1500; SEP = 4.0
RNORM = 2.0; DBAR_NORM = 5.0
THETAS = np.arange(0, 91, 5.0)
SEEDS = list(range(30))

rows = []
for sd in SEEDS:
    rng = np.random.default_rng(1000 + sd)
    X, y, _, _ = make_source(rng, d=D, K=K, n=NPC, sep=SEP)
    Xtr, ytr, Xev, yev = split_by_class(X, y, K, NTR, rng)
    W, b = ridge_probe(Xtr, ytr, K)
    pis = np.full(K, 1.0 / K)
    g = np.random.default_rng(3000 + sd).standard_normal(D)
    dbar = g / np.linalg.norm(g) * DBAR_NORM
    Vd = disc_basis(W); P = Vd.T @ Vd
    mcache = {c: margins(Xev[yev == c], yev[yev == c], W, b) for c in range(K)}
    for th in THETAS:
        rho = make_residuals(np.random.default_rng(4000 + sd), W, pis, RNORM, np.deg2rad(th))
        deltas = dbar[None, :] + rho
        Xt = Xev + deltas[yev]
        a_or = acc(Xt - deltas[yev], yev, W, b)
        a_gl = acc(Xt - dbar, yev, W, b)
        bnd, G, per = theory_bound(Xev, yev, W, b, rho, pis)
        rp = float(np.sqrt(np.sum(pis * np.sum((rho @ P) ** 2, axis=1))))
        cheb = 0.0; subg = 0.0
        for c in range(K):
            m = mcache[c]; mb = float(m.mean()); s2 = float(m.var())
            if G[c] < mb:
                cheb += pis[c] * min(1.0, s2 / max((mb - G[c]) ** 2, 1e-12))
                subg += pis[c] * float(np.exp(-((mb - G[c]) ** 2) / (2 * max(s2, 1e-12))))
            else:
                cheb += pis[c]; subg += pis[c]
        rows.append(dict(seed=sd, theta=float(th), gap=a_or - a_gl, bound=bnd,
                         rproj=rp, naive=float(min(1.0, rp)), cheb=cheb, subg=subg,
                         a_or=a_or, a_gl=a_gl, Gmax=float(G.max())))

with open(SYN_OUT / 'rotation.pkl', 'wb') as f:
    pickle.dump(rows, f)
print('ROTATION SWEEP d=%d K=%d sep=%.1f r=%.1f seeds=%d angles=%d' % (D, K, SEP, RNORM, len(SEEDS), len(THETAS)))
print('%6s %20s %20s %9s %9s %9s %9s' % ('theta', 'measured gap', 'Thm1 bound', 'naive', 'cheb', 'subG', 'tight'))
for th in THETAS:
    R = [r for r in rows if r['theta'] == th]
    gm, gc = ci95([r['gap'] for r in R]); bm, bc = ci95([r['bound'] for r in R])
    nm, _ = ci95([r['naive'] for r in R]); cm, _ = ci95([r['cheb'] for r in R]); sm, _ = ci95([r['subg'] for r in R])
    tt = bm / gm if gm > 1e-9 else float('nan')
    print('%6.1f %10.5f +- %-7.5f %10.5f +- %-7.5f %9.4f %9.4f %9.4f %9.2f' % (th, gm, gc, bm, bc, nm, cm, sm, tt))
viol = sum(1 for r in rows if r['bound'] < r['gap'] - 1e-12)
print('BOUND_VIOLATIONS %d / %d' % (viol, len(rows)))
R90 = [r for r in rows if r['theta'] == 90.0]
print('AT_90DEG mean_gap=%.6f max_abs_gap=%.6f  (Gamma_c<=0 regime)' % (np.mean([r['gap'] for r in R90]), max(abs(r['gap']) for r in R90)))
print('NONNEG_CHECK  frac(a_gl >= a_or) at 90deg = %.3f' % np.mean([r['a_gl'] >= r['a_or'] - 1e-12 for r in R90]))
