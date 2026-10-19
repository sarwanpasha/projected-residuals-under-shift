"""Margin-quantile frontier evaluated on held-out data.

The per-class threshold is the q-quantile of the positive source margins on a
calibration split, and the per-class loss of the common-mean correction is measured on
a disjoint test split.  Among the classes whose projected residual Gamma_c falls below
the threshold, the script reports the mean and maximum loss and the number of classes
whose loss exceeds q plus the DKW band.

Output: results/synthetic/frontier.pkl
"""
import numpy as np, pickle
from prlb.core import *
from prlb.paths import SYN_OUT, ensure

ensure(SYN_OUT)

D = 50; K = 8; NPC = 4000; NTR = 1500; NCAL = 1250; SEP = 4.0
DBAR_NORM = 5.0
QS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
RS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
THS = [0.0, 30.0, 60.0, 75.0, 85.0]
SEEDS = list(range(20))
cells = []
for sd in SEEDS:
    rng = np.random.default_rng(1000 + sd)
    X, y, _, _ = make_source(rng, d=D, K=K, n=NPC, sep=SEP)
    Xtr, ytr, Xrest, yrest = split_by_class(X, y, K, NTR, rng)
    Xca, yca, Xte, yte = split_by_class(Xrest, yrest, K, NCAL, rng)
    W, b = ridge_probe(Xtr, ytr, K)
    pis = np.full(K, 1.0 / K)
    g = np.random.default_rng(3000 + sd).standard_normal(D)
    dbar = g / np.linalg.norm(g) * DBAR_NORM
    mca = {c: margins(Xca[yca == c], yca[yca == c], W, b) for c in range(K)}
    for r in RS:
        for th in THS:
            rho = make_residuals(np.random.default_rng(4000 + sd), W, pis, r, np.deg2rad(th))
            G = all_gamma_caps(rho, W)
            deltas = dbar[None, :] + rho
            Xt = Xte + deltas[yte]
            per_loss = np.zeros(K)
            for c in range(K):
                m = (yte == c)
                ok_or = predict((Xt - deltas[yte])[m], W, b) == c
                ok_gl = predict((Xt - dbar)[m], W, b) == c
                per_loss[c] = float(np.mean(ok_or & ~ok_gl))
            cells.append(dict(seed=sd, r=r, th=th, G=G.copy(), per_loss=per_loss.copy(),
                              qth={q: np.array([pos_quantile(mca[c], q) for c in range(K)]) for q in QS},
                              ncal=len(yca) // K, nte=int(np.sum(yte == 0))))
with open(SYN_OUT / 'frontier.pkl', 'wb') as f:
    pickle.dump(cells, f)
print('MARGIN-QUANTILE FRONTIER  d=%d K=%d seeds=%d cells=%d  ncal/class=%d ntest/class=%d'
      % (D, K, len(SEEDS), len(cells), cells[0]['ncal'], cells[0]['nte']))
print('%6s %10s %14s %14s %14s %10s' % ('q', 'n_qualif', 'mean loss', 'max loss', 'DKW band', 'violations'))
for q in QS:
    vals = []
    for cell in cells:
        thr = cell['qth'][q]
        for c in range(K):
            if cell['G'][c] <= thr[c]:
                vals.append(cell['per_loss'][c])
    if not vals:
        print('%6.3f %10d %14s' % (q, 0, 'none qualify')); continue
    vals = np.array(vals)
    band = dkw_band(cells[0]['ncal'])
    viol = int(np.sum(vals > q + band))
    print('%6.3f %10d %14.6f %14.6f %14.6f %10d' % (q, len(vals), vals.mean(), vals.max(), band, viol))
