"""Label requirement of counting the marginal (Route B) on a dimension-by-class grid.

Usage:  python experiments/cifar10c/route_b_grid.py {clip|dinov2}

For d in {32, 128, 512} (PCA truncation of the same embeddings) and K in {2, ..., 10}
(random class subsets of CIFAR-10-C, three subsets per K below 10) the script measures
B99, the budget at which counting attains 99 percent of the oracle-marginal ceiling at
severity 5 and maximum drift, and fits log B99 = a log d + b log K + c.  It also
reports whether counting at B close to 4K beats refitting at the same budget.

Output: results/cifar10c/route_b_<enc>.pkl
"""
import sys, pickle
import numpy as np
from prlb.core import ridge_probe, acc, class_means
from prlb.paths import CIFAR_EMB, CIFAR_OUT, CORRUPTIONS, ensure

EMB = str(CIFAR_EMB)
ensure(CIFAR_OUT)
K0 = 10
CORR = CORRUPTIONS
SEV = 5
DIMS = [32, 128, 512]
KS = [2, 4, 6, 8, 10]
REPS = [0, 1, 2]
SEEDS = [0, 1]
BUDG = [5, 10, 20, 40, 80, 160, 320, 640]
ENC = sys.argv[1]
EPS = 1e-3

Xtr0 = np.load('%s/%s_train.npy' % (EMB, ENC)).astype(np.float64)
ytr = np.load('%s/labels_train.npy' % EMB)
Xs0 = np.load('%s/%s_clean.npy' % (EMB, ENC)).astype(np.float64)
ys = np.load('%s/labels_test.npy' % EMB)
sc = np.linalg.norm(Xtr0, axis=1).mean()
Xtr0 = Xtr0 / sc; Xs0 = Xs0 / sc
cen = Xtr0.mean(0)
S, Vt = np.linalg.svd(Xtr0 - cen, full_matrices=False)[1:]
print('%s ambient d=%d' % (ENC, Xtr0.shape[1]), flush=True)

SUBS = {}
for K in KS:
    for rp in REPS:
        if K == K0:
            SUBS[(K, rp)] = list(range(K0)) if rp == 0 else None
        else:
            r = np.random.default_rng(991 * K + rp)
            SUBS[(K, rp)] = sorted(r.choice(K0, size=K, replace=False).tolist())
SUBS = {k: v for k, v in SUBS.items() if v is not None}
print('SUBSETS', {k: v for k, v in sorted(SUBS.items())}, flush=True)

BASE = {}
for dd in DIMS:
    P = Vt[:dd].T
    Xtrp = (Xtr0 - cen) @ P
    Xsp = (Xs0 - cen) @ P
    for (K, rp), cls in sorted(SUBS.items()):
        remap = -np.ones(K0, dtype=np.int64)
        for i, c in enumerate(cls):
            remap[c] = i
        mtr = np.isin(ytr, cls); mte = np.isin(ys, cls)
        W, b = ridge_probe(Xtrp[mtr], remap[ytr[mtr]], K, lam=1e-2)
        Xsk = Xsp[mte]; ysk = remap[ys[mte]]
        MS = class_means(Xsk, ysk, K)
        pi_s = np.bincount(ysk, minlength=K) / len(ysk)
        BASE[(dd, K, rp)] = dict(P=P, W=W, b=b, MS=MS, pi_s=pi_s, mte=mte, ysk=ysk,
                                 idx=[np.where(ysk == c)[0] for c in range(K)],
                                 a0=acc(Xsk, ysk, W, b))
    print('  dd=%3d built %d subset probes, clean acc K=10 %.4f'
          % (dd, len(SUBS), BASE[(dd, K0, 0)]['a0']), flush=True)

rows = []
for cn in CORR:
    Z = np.load('%s/%s_%s.npz' % (EMB, ENC, cn))
    Xt0 = Z['s%d' % SEV].astype(np.float64) / sc
    for dd in DIMS:
        for (K, rp), cls in sorted(SUBS.items()):
            J = BASE[(dd, K, rp)]
            Xt = (Xt0[J['mte']] - cen) @ J['P']
            ysk = J['ysk']; MS = J['MS']; W = J['W']; b = J['b']; pi_s = J['pi_s']
            deltas = class_means(Xt, ysk, K) - MS
            for sd in SEEDS:
                rng = np.random.default_rng(7919 * sd + 131 * dd + 17 * K + 5 * rp
                                            + sum(ord(ch) for ch in cn) % 997)
                pk = rng.dirichlet(np.full(K, 0.5)); pk = np.maximum(pk, 0.02); pk = pk / pk.sum()
                M = int(min(6000, np.floor((np.array([len(i) for i in J['idx']]) / pk).min())))
                nc = np.maximum((pk * M).astype(int), 5)
                sel = np.concatenate([rng.choice(J['idx'][c], size=nc[c], replace=False) for c in range(K)])
                pit_emp = nc / nc.sum()
                Xts = Xt[sel]; yts = ysk[sel]; n = len(sel)
                mu_t = Xts.mean(0)
                out = dict(corr=cn, dd=dd, K=K, rep=rp, seed=sd,
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

with open(CIFAR_OUT / ('route_b_%s.pkl' % ENC), 'wb') as f:
    pickle.dump(rows, f)

print('\n=== ROUTE B d-K GRID: %s, %d cells ===' % (ENC, len(rows)), flush=True)


def mn(s, k):
    return float(np.mean([r[k] for r in s]))


def b99(s):
    lo = mn(s, 'a_naive'); hi = mn(s, 'a_orcb')
    if hi - lo < 0.005:
        return float('nan'), lo, hi
    tgt = lo + 0.99 * (hi - lo)
    prev = None
    for B in BUDG:
        v = mn(s, 'a_cnt%d' % B)
        if v >= tgt:
            if prev is None:
                return float(BUDG[0]), lo, hi
            Bp, vp = prev
            t = (tgt - vp) / (v - vp)
            return float(np.exp(np.log(Bp) + t * (np.log(B) - np.log(Bp)))), lo, hi
        prev = (B, v)
    return float('nan'), lo, hi


tab = {}
print('\n  dd    K  rep   drift  naive   ceiling  ORACLE   B99      cnt(4K)  refit(4K)', flush=True)
for dd in DIMS:
    for (K, rp) in sorted(SUBS.keys()):
        s = [r for r in rows if r['dd'] == dd and r['K'] == K and r['rep'] == rp]
        v, lo, hi = b99(s)
        tab[(dd, K, rp)] = v
        Bq = min(BUDG, key=lambda B: abs(B - 4 * K))
        print('  %4d %4d %3d  %6.3f  %.4f  %.4f  %.4f  %8.2f  %.4f  %.4f'
              % (dd, K, rp, mn(s, 'drift'), lo, hi, mn(s, 'a_oracle'), v,
                 mn(s, 'a_cnt%d' % Bq), mn(s, 'a_refit%d' % Bq)), flush=True)

keys = [k for k in tab if np.isfinite(tab[k])]
print('\n  usable cells for the fit: %d of %d' % (len(keys), len(tab)), flush=True)
if len(keys) >= 6:
    X = np.array([[np.log(k[0]), np.log(k[1]), 1.0] for k in keys])
    yv = np.log(np.array([tab[k] for k in keys]))
    co = np.linalg.lstsq(X, yv, rcond=None)[0]
    pr = X @ co
    r2 = 1.0 - ((yv - pr) ** 2).sum() / max(((yv - yv.mean()) ** 2).sum(), 1e-12)
    print('  TWO-EXPONENT FIT  log B99 = %+.4f*log d %+.4f*log K %+.4f   R2=%.4f'
          % (co[0], co[1], co[2], r2), flush=True)
    Xd = np.array([[np.log(k[0]), 1.0] for k in keys])
    cd = np.linalg.lstsq(Xd, yv, rcond=None)[0]
    print('  d-only slope %+.4f' % cd[0], flush=True)
    Xk = np.array([[np.log(k[1]), 1.0] for k in keys])
    ck = np.linalg.lstsq(Xk, yv, rcond=None)[0]
    print('  K-only slope %+.4f' % ck[0], flush=True)

print('\n  counting at B~4K vs refitting at B~4K', flush=True)
bad = 0
for dd in DIMS:
    for (K, rp) in sorted(SUBS.keys()):
        s = [r for r in rows if r['dd'] == dd and r['K'] == K and r['rep'] == rp]
        Bq = min(BUDG, key=lambda B: abs(B - 4 * K))
        if mn(s, 'a_cnt%d' % Bq) <= mn(s, 'a_refit%d' % Bq):
            bad += 1
print('   cells where counting does not lead: %d of %d' % (bad, len(tab)), flush=True)
