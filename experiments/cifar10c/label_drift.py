"""Label-marginal drift on real embeddings.

Usage:  python experiments/cifar10c/label_drift.py {clip|dinov2}

Label-free estimation of the common displacement as a difference of means silently
assumes pi^T = pi^S.  Target samples are resampled so that their label marginal moves
from the source marginal (alpha = 0) to a Dirichlet(0.5) draw (alpha = 1), for 19
corruptions, severities 1, 3, 5 and 3 seeds.  The script compares the naive estimate,
the BBSE-debiased estimate and the estimate with the true marginal, both in accuracy
and in the error of the estimated common displacement.

Output: results/cifar10c/label_drift_<enc>.pkl
"""
import sys, pickle
import numpy as np
from prlb.core import ridge_probe, acc, predict, class_means, bbse_pi, ci95
from prlb.paths import CIFAR_EMB, CIFAR_OUT, CORRUPTIONS, ensure

EMB = str(CIFAR_EMB)
ensure(CIFAR_OUT)
K = 10
CORR = CORRUPTIONS
SEVS = [1, 3, 5]
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = [0, 1, 2]
ENC = sys.argv[1]

Xtr = np.load('%s/%s_train.npy' % (EMB, ENC)).astype(np.float64)
ytr = np.load('%s/labels_train.npy' % EMB)
Xs = np.load('%s/%s_clean.npy' % (EMB, ENC)).astype(np.float64)
ys = np.load('%s/labels_test.npy' % EMB)
sc = np.linalg.norm(Xtr, axis=1).mean()
Xtr = Xtr / sc; Xs = Xs / sc
W, b = ridge_probe(Xtr, ytr, K, lam=1e-2)
print('%s d=%d  probe clean-test=%.4f' % (ENC, Xtr.shape[1], acc(Xs, ys, W, b)), flush=True)
MS = class_means(Xs, ys, K)
pi_s = np.bincount(ys, minlength=K) / len(ys)
mu_s = Xs.mean(0)
idx_by_c = [np.where(ys == c)[0] for c in range(K)]

rows = []
for cn in CORR:
    Z = np.load('%s/%s_%s.npz' % (EMB, ENC, cn))
    for sv in SEVS:
        Xt = Z['s%d' % sv].astype(np.float64) / sc
        MT = class_means(Xt, ys, K)
        deltas = MT - MS
        for a in ALPHAS:
            for sd in SEEDS:
                rng = np.random.default_rng(10000 * sd + 17 * sv + sum(ord(ch) for ch in cn) % 997)
                pk = rng.dirichlet(np.full(K, 0.5))
                pk = np.maximum(pk, 0.01); pk = pk / pk.sum()
                pit = (1.0 - a) * pi_s + a * pk
                pit = pit / pit.sum()
                M = int(min(8000, np.floor((np.array([len(i) for i in idx_by_c]) / pit).min())))
                nc = np.maximum((pit * M).astype(int), 5)
                sel = np.concatenate([rng.choice(idx_by_c[c], size=nc[c], replace=False) for c in range(K)])
                pit_emp = nc / nc.sum()
                Xts = Xt[sel]; yts = ys[sel]
                mu_t = Xts.mean(0)
                dbar_true = pit_emp @ deltas
                d_naive = mu_t - mu_s
                pih = bbse_pi(Xs, ys, Xts, W, b, K, pi_s)
                d_deb = mu_t - (pih @ MS)
                d_orc = mu_t - (pit_emp @ MS)
                rows.append(dict(
                    corr=cn, sev=sv, alpha=a, seed=sd, n=int(nc.sum()),
                    drift=float(np.abs(pit_emp - pi_s).sum()),
                    pierr=float(np.abs(pih - pit_emp).sum()),
                    a_none=acc(Xts, yts, W, b),
                    a_naive=acc(Xts - d_naive, yts, W, b),
                    a_deb=acc(Xts - d_deb, yts, W, b),
                    a_orcb=acc(Xts - d_orc, yts, W, b),
                    a_oracle=acc(Xts - deltas[yts], yts, W, b),
                    e_naive=float(np.linalg.norm(d_naive - dbar_true)),
                    e_deb=float(np.linalg.norm(d_deb - dbar_true)),
                    e_orcb=float(np.linalg.norm(d_orc - dbar_true))))
    print('done %s' % cn, flush=True)

with open(CIFAR_OUT / ('label_drift_%s.pkl' % ENC), 'wb') as f:
    pickle.dump(rows, f)

def f(key, sub):
    return ci95([r[key] for r in sub])

print('\n=== REAL LABEL DRIFT: %s, %d cells (%d pairs x %d alphas x %d seeds) ==='
      % (ENC, len(rows), len(CORR) * len(SEVS), len(ALPHAS), len(SEEDS)))
print('alpha  driftL1   none      naive     debiased  oracle-b  ORACLE    deb-naive        pi_err')
for a in ALPHAS:
    s = [r for r in rows if r['alpha'] == a]
    dn = [r['a_deb'] - r['a_naive'] for r in s]
    m, h = ci95(dn)
    print('%.1f   %.4f   %.4f    %.4f    %.4f    %.4f    %.4f    %+.4f+-%.4f  %.4f'
          % (a, f('drift', s)[0], f('a_none', s)[0], f('a_naive', s)[0], f('a_deb', s)[0],
             f('a_orcb', s)[0], f('a_oracle', s)[0], m, h, f('pierr', s)[0]))
print('\nESTIMATION ERROR of common displacement  ||dhat - dbar_true||')
print('alpha   naive     debiased  oracle-b   ratio naive/deb')
for a in ALPHAS:
    s = [r for r in rows if r['alpha'] == a]
    en, ed, eo = f('e_naive', s)[0], f('e_deb', s)[0], f('e_orcb', s)[0]
    print('%.1f    %.4f    %.4f    %.4f     %.2fx' % (a, en, ed, eo, en / max(ed, 1e-12)))
hi = [r for r in rows if r['alpha'] == ALPHAS[-1]]
w = sum(1 for r in hi if r['a_deb'] > r['a_naive'])
print('\nAT MAX DRIFT: debiased beats naive on %d/%d cells' % (w, len(hi)))
