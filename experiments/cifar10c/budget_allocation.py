"""Where a small target label budget should be spent under label drift.

Usage:  python experiments/cifar10c/budget_allocation.py {clip|dinov2}

Three uses of B target labels are compared at B in {50, 100, 200, 400, 800}:
  Route A  refit the probe on the B labels, with a fixed ridge penalty and with a
           penalty chosen by cross-validation on the budget alone;
  Route B  count the target label marginal from the B labels (post-stratified
           counting) and use it in the common-displacement estimate;
  Route C  estimate the target confusion conditional from the B labels and solve the
           label-shift equations on the full unlabeled target.
The label marginal drifts with alpha in {0, 0.4, 0.8, 1}, over 19 corruptions,
severities 1, 3, 5 and 3 seeds.

Output: results/cifar10c/budget_<enc>.pkl
"""
import sys, pickle
import numpy as np
from prlb.core import ridge_probe, ridge_cv, acc, predict, class_means, ci95
from prlb.paths import CIFAR_EMB, CIFAR_OUT, CORRUPTIONS, ensure

EMB = str(CIFAR_EMB)
ensure(CIFAR_OUT)
K = 10
CORR = CORRUPTIONS
SEVS = [1, 3, 5]
ALPHAS = [0.0, 0.4, 0.8, 1.0]
SEEDS = [0, 1, 2]
BUDG = [50, 100, 200, 400, 800]
ENC = sys.argv[1]

Xtr = np.load('%s/%s_train.npy' % (EMB, ENC)).astype(np.float64)
ytr = np.load('%s/labels_train.npy' % EMB)
Xs = np.load('%s/%s_clean.npy' % (EMB, ENC)).astype(np.float64)
ys = np.load('%s/labels_test.npy' % EMB)
sc = np.linalg.norm(Xtr, axis=1).mean()
Xtr = Xtr / sc; Xs = Xs / sc
W, b = ridge_probe(Xtr, ytr, K, lam=1e-2)
MS = class_means(Xs, ys, K)
pi_s = np.bincount(ys, minlength=K) / len(ys)
idx_by_c = [np.where(ys == c)[0] for c in range(K)]
ps_src = predict(Xs, W, b)
CS = np.zeros((K, K))
for k in range(K):
    for c in range(K):
        CS[k, c] = np.mean((ps_src == k) & (ys == c))
print('%s clean=%.4f' % (ENC, acc(Xs, ys, W, b)), flush=True)
EPS = 1e-3


def solve_pi(A, q):
    p = np.linalg.lstsq(A, q, rcond=None)[0]
    p = np.maximum(p, 0.0)
    s = p.sum()
    return p / s if s > 1e-12 else pi_s.copy()



rows = []
for cn in CORR:
    Z = np.load('%s/%s_%s.npz' % (EMB, ENC, cn))
    for sv in SEVS:
        Xt = Z['s%d' % sv].astype(np.float64) / sc
        deltas = class_means(Xt, ys, K) - MS
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
                pit_emp = nc / nc.sum()
                Xts = Xt[sel]; yts = ys[sel]; n = len(sel)
                mu_t = Xts.mean(0)
                dbar_true = pit_emp @ deltas
                pt = predict(Xts, W, b)
                q = np.bincount(pt, minlength=K) / n
                out = dict(corr=cn, sev=sv, alpha=a, seed=sd, n=n,
                           drift=float(np.abs(pit_emp - pi_s).sum()))

                def rec(tag, pv, extra=None):
                    d = mu_t - (pv @ MS)
                    out['a_' + tag] = acc(Xts - d, yts, W, b)
                    out['e_' + tag] = float(np.linalg.norm(d - dbar_true))
                    out['p_' + tag] = float(np.abs(pv - pit_emp).sum())

                rec('naive', pi_s)
                rec('deb0', solve_pi(CS, q))
                rec('orcb', pit_emp)
                out['a_none'] = acc(Xts, yts, W, b)
                out['a_oracle'] = acc(Xts - deltas[yts], yts, W, b)
                for B in BUDG:
                    lb = rng.choice(n, size=min(B, n), replace=False)
                    yl = yts[lb]; pl = pt[lb]
                    cnts = np.bincount(yl, minlength=K).astype(float)
                    rec('cnt%d' % B, (cnts + EPS) / (cnts + EPS).sum())
                    CT = np.zeros((K, K))
                    for c in range(K):
                        m = (yl == c)
                        if m.sum() > 0:
                            CT[:, c] = np.bincount(pl[m], minlength=K) / m.sum()
                        else:
                            CT[:, c] = CS[:, c] / max(CS[:, c].sum(), 1e-12)
                    rec('cnd%d' % B, solve_pi(CT, q))
                    Wr, br = ridge_probe(Xts[lb], yl, K, lam=1e-2)
                    out['a_refit%d' % B] = acc(Xts, yts, Wr, br)
                    Wc, bc, lmb = ridge_cv(Xts[lb], yl, K)
                    out['a_refitcv%d' % B] = acc(Xts, yts, Wc, bc)
                    out['lam%d' % B] = lmb
                rows.append(out)
    print('done %s' % cn, flush=True)

with open(CIFAR_OUT / ('budget_%s.pkl' % ENC), 'wb') as f:
    pickle.dump(rows, f)

print('\n=== BUDGETED LABEL SHIFT: %s, %d cells ===' % (ENC, len(rows)))
for a in ALPHAS:
    s = [r for r in rows if r['alpha'] == a]
    print('\n--- drift L1 = %.4f  (none %.4f  naive %.4f  BBSE-src %.4f  oracle-pi %.4f  per-class ORACLE %.4f)'
          % (ci95([r['drift'] for r in s])[0], ci95([r['a_none'] for r in s])[0],
             ci95([r['a_naive'] for r in s])[0], ci95([r['a_deb0'] for r in s])[0],
             ci95([r['a_orcb'] for r in s])[0], ci95([r['a_oracle'] for r in s])[0]))
    print('  B     refit-probe   count-pi      conditional-pi   cnd-naive        piL1 cnt   piL1 cnd')
    for B in BUDG:
        m, h = ci95([r['a_cnd%d' % B] - r['a_naive'] for r in s])
        print('  %-4d  %.4f        %.4f        %.4f           %+.4f+-%.4f  %.4f     %.4f'
              % (B, ci95([r['a_refit%d' % B] for r in s])[0],
                 ci95([r['a_cnt%d' % B] for r in s])[0],
                 ci95([r['a_cnd%d' % B] for r in s])[0], m, h,
                 ci95([r['p_cnt%d' % B] for r in s])[0], ci95([r['p_cnd%d' % B] for r in s])[0]))
print('\nDOMINANCE of conditional-pi over naive, by drift and budget')
for a in ALPHAS:
    s = [r for r in rows if r['alpha'] == a]
    print('drift %.2f  ' % ci95([r['drift'] for r in s])[0] + '  '.join(
        'B=%d %3d/%3d' % (B, sum(1 for r in s if r['a_cnd%d' % B] >= r['a_naive'] - 1e-9), len(s))
        for B in BUDG))
print('\n=== REFIT REGULARIZATION CHECK: %s ===' % ENC)
print('drift B none labelfree countpi refitfix refitcv | cv-fix | cv-none | cv-count | medlam')
for a in ALPHAS:
    s = [r for r in rows if r['alpha'] == a]
    dr = ci95([r['drift'] for r in s])[0]
    for B in BUDG:
        d1 = ci95([r['a_refitcv%d' % B] - r['a_refit%d' % B] for r in s])
        d2 = ci95([r['a_refitcv%d' % B] - r['a_none'] for r in s])
        d3 = ci95([r['a_refitcv%d' % B] - r['a_cnt%d' % B] for r in s])
        print('%.5f %-5d %.5f %.5f %.5f %.5f %.5f | %+.5f+-%.5f | %+.5f+-%.5f | %+.5f+-%.5f | %g'
              % (dr, B, ci95([r['a_none'] for r in s])[0],
                 ci95([r['a_naive'] for r in s])[0],
                 ci95([r['a_cnt%d' % B] for r in s])[0],
                 ci95([r['a_refit%d' % B] for r in s])[0],
                 ci95([r['a_refitcv%d' % B] for r in s])[0],
                 d1[0], d1[1], d2[0], d2[1], d3[0], d3[1],
                 float(np.median([r['lam%d' % B] for r in s]))))
