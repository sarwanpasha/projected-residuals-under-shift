"""Crossover budget as a function of the embedding dimension.

Usage:  python experiments/cifar10c/dim_sweep.py {clip|dinov2}

The encoder, the task and K = 10 are held fixed while the dimension d is varied by PCA
truncation of the same embeddings, d in {8, ..., 512}.  For each d the script measures
the budget B*(d) at which refitting the probe overtakes counting the target label
marginal under maximum drift, fits log B* against log d, and reports the zero-drift
cost of counting relative to the common-mean correction.

Output: results/cifar10c/dim_sweep_<enc>.pkl
"""
import sys, pickle
import numpy as np
from prlb.core import ridge_probe, acc, predict, class_means, ci95
from prlb.paths import CIFAR_EMB, CIFAR_OUT, CORRUPTIONS, ensure

EMB = str(CIFAR_EMB)
ensure(CIFAR_OUT)
K = 10
CORR = CORRUPTIONS
SEVS = [3, 5]
ALPHAS = [0.0, 1.0]
SEEDS = [0, 1, 2]
BUDG = [25, 50, 100, 200, 400, 800, 1600]
DIMS = [8, 16, 32, 64, 128, 256, 512]
ENC = sys.argv[1]
EPS = 1e-3

Xtr0 = np.load('%s/%s_train.npy' % (EMB, ENC)).astype(np.float64)
ytr = np.load('%s/labels_train.npy' % EMB)
Xs0 = np.load('%s/%s_clean.npy' % (EMB, ENC)).astype(np.float64)
ys = np.load('%s/labels_test.npy' % EMB)
sc = np.linalg.norm(Xtr0, axis=1).mean()
Xtr0 = Xtr0 / sc; Xs0 = Xs0 / sc
D0 = Xtr0.shape[1]
print('%s d=%d n_train=%d n_test=%d' % (ENC, D0, len(Xtr0), len(Xs0)), flush=True)

cen = Xtr0.mean(0)
U, S, Vt = np.linalg.svd(Xtr0 - cen, full_matrices=False)
print('PCA spectrum top5 %s tail %s' % (np.round(S[:5], 3).tolist(), np.round(S[-3:], 4).tolist()), flush=True)

pi_s = np.bincount(ys, minlength=K) / len(ys)
idx_by_c = [np.where(ys == c)[0] for c in range(K)]

PJ = {}
for dd in DIMS:
    P = Vt[:dd].T
    Xtr = (Xtr0 - cen) @ P
    Xsp = (Xs0 - cen) @ P
    W, b = ridge_probe(Xtr, ytr, K, lam=1e-2)
    MS = class_means(Xsp, ys, K)
    PJ[dd] = dict(P=P, W=W, b=b, MS=MS, Xs=Xsp)
    print('  dd=%3d source-clean acc=%.4f' % (dd, acc(Xsp, ys, W, b)), flush=True)

rows = []
for cn in CORR:
    Z = np.load('%s/%s_%s.npz' % (EMB, ENC, cn))
    for sv in SEVS:
        Xt0 = Z['s%d' % sv].astype(np.float64) / sc
        for dd in DIMS:
            J = PJ[dd]
            Xt = (Xt0 - cen) @ J['P']
            W, b, MS = J['W'], J['b'], J['MS']
            deltas = class_means(Xt, ys, K) - MS
            for a in ALPHAS:
                for sd in SEEDS:
                    rng = np.random.default_rng(10000 * sd + 17 * sv + 3 * dd + sum(ord(ch) for ch in cn) % 997)
                    pk = rng.dirichlet(np.full(K, 0.5)); pk = np.maximum(pk, 0.01); pk = pk / pk.sum()
                    pit = (1.0 - a) * pi_s + a * pk; pit = pit / pit.sum()
                    M = int(min(8000, np.floor((np.array([len(i) for i in idx_by_c]) / pit).min())))
                    nc = np.maximum((pit * M).astype(int), 5)
                    sel = np.concatenate([rng.choice(idx_by_c[c], size=nc[c], replace=False) for c in range(K)])
                    pit_emp = nc / nc.sum()
                    Xts = Xt[sel]; yts = ys[sel]; n = len(sel)
                    mu_t = Xts.mean(0)
                    out = dict(corr=cn, sev=sv, dd=dd, alpha=a, seed=sd,
                               drift=float(np.abs(pit_emp - pi_s).sum()))
                    out['a_none'] = acc(Xts, yts, W, b)
                    out['a_naive'] = acc(Xts - (mu_t - MS.T @ pi_s), yts, W, b)
                    out['a_orcb'] = acc(Xts - (mu_t - MS.T @ pit_emp), yts, W, b)
                    out['a_oracle'] = acc(Xts - deltas[yts], yts, W, b)
                    for B in BUDG:
                        lb = rng.choice(n, size=min(B, n), replace=False)
                        y1 = yts[lb]
                        cnts = np.bincount(y1, minlength=K).astype(float)
                        pcnt = (cnts + EPS) / (cnts + EPS).sum()
                        out['a_cnt%d' % B] = acc(Xts - (mu_t - MS.T @ pcnt), yts, W, b)
                        Wr, br = ridge_probe(Xts[lb], y1, K, lam=1e-2)
                        out['a_refit%d' % B] = acc(Xts, yts, Wr, br)
                    rows.append(out)
    print('done %s' % cn, flush=True)

with open(CIFAR_OUT / ('dim_sweep_%s.pkl' % ENC), 'wb') as f:
    pickle.dump(rows, f)

print('\n=== DIMENSION SWEEP: %s, %d cells ===' % (ENC, len(rows)), flush=True)


def mean_of(s, key):
    return float(np.mean([r[key] for r in s]))


def crossover(s):
    lo = None
    prev = None
    for B in BUDG:
        dcur = mean_of(s, 'a_refit%d' % B) - mean_of(s, 'a_cnt%d' % B)
        if dcur >= 0.0:
            if prev is None:
                return float(BUDG[0])
            Bp, dp = prev
            t = (0.0 - dp) / (dcur - dp)
            return float(np.exp(np.log(Bp) + t * (np.log(B) - np.log(Bp))))
        prev = (B, dcur)
    return float('nan')


bstar = {}
for a in ALPHAS:
    print('\n--- alpha=%.1f' % a, flush=True)
    print('  dd   drift   none    naive   oracle-pi  ORACLE   ' + '  '.join('cnt%-5d refit%-5d' % (B, B) for B in BUDG))
    for dd in DIMS:
        s = [r for r in rows if r['dd'] == dd and r['alpha'] == a]
        line = ' %4d %6.3f  %.4f  %.4f  %.4f     %.4f  ' % (dd, mean_of(s, 'drift'), mean_of(s, 'a_none'),
                                                            mean_of(s, 'a_naive'), mean_of(s, 'a_orcb'), mean_of(s, 'a_oracle'))
        line += '  '.join('%.4f %.4f' % (mean_of(s, 'a_cnt%d' % B), mean_of(s, 'a_refit%d' % B)) for B in BUDG)
        print(line, flush=True)
    if a > 0:
        print('\n  CROSSOVER budget B* where refit overtakes counting', flush=True)
        for dd in DIMS:
            s = [r for r in rows if r['dd'] == dd and r['alpha'] == a]
            bstar[dd] = crossover(s)
            print('   dd=%4d  d/K=%6.1f  Bstar=%8.1f  gapB100=%+.4f  gapB25=%+.4f' %
                  (dd, dd / K, bstar[dd], mean_of(s, 'a_cnt100') - mean_of(s, 'a_refit100'),
                   mean_of(s, 'a_cnt25') - mean_of(s, 'a_refit25')), flush=True)
        ok = [dd for dd in DIMS if np.isfinite(bstar[dd])]
        if len(ok) >= 3:
            x = np.log(np.array(ok, dtype=float)); yv = np.log(np.array([bstar[dd] for dd in ok]))
            A = np.stack([x, np.ones_like(x)], 1)
            sl, ic = np.linalg.lstsq(A, yv, rcond=None)[0]
            pred = A @ np.array([sl, ic])
            r2 = 1.0 - ((yv - pred) ** 2).sum() / max(((yv - yv.mean()) ** 2).sum(), 1e-12)
            print('\n  SCALING LAW  log Bstar = %.4f * log d + %.4f   R2=%.4f'
                  % (sl, ic, r2), flush=True)
            print('  FITTED Bstar at d=512: %.1f' % np.exp(sl * np.log(512.0) + ic), flush=True)

print('\nZERO-DRIFT COST OF COUNTING', flush=True)
worst = 0.0
for dd in DIMS:
    s = [r for r in rows if r['dd'] == dd and r['alpha'] == 0.0]
    v = [mean_of(s, 'a_naive') - mean_of(s, 'a_cnt%d' % B) for B in BUDG]
    worst = max(worst, max(v))
    print('   dd=%4d  ' % dd + '  '.join('%+.4f' % z for z in v), flush=True)
print('   WORST %+.4f' % worst, flush=True)
