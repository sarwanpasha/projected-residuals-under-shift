"""Label requirement of refitting the probe (Route A) on the same grid as Route B.

Usage:  python experiments/cifar10c/route_a_grid.py {clip|dinov2}

Uses the same d by K grid and the same class subsets as route_b_grid.py.  B_eps is the
budget at which the refitted probe comes within eps in {0.05, 0.02} of a probe fitted
on the whole labeled target sample, and the script fits
log B_eps = a log d + b log K + c.

Output: results/cifar10c/route_a_<enc>.pkl
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
BUDG = [25, 50, 100, 200, 400, 800, 1600, 3200]
EPSL = [0.05, 0.02]
ENC = sys.argv[1]

Xtr0 = np.load('%s/%s_train.npy' % (EMB, ENC)).astype(np.float64)
ytr = np.load('%s/labels_train.npy' % EMB)
Xs0 = np.load('%s/%s_clean.npy' % (EMB, ENC)).astype(np.float64)
ys = np.load('%s/labels_test.npy' % EMB)
sc = np.linalg.norm(Xtr0, axis=1).mean()
Xtr0 = Xtr0 / sc; Xs0 = Xs0 / sc
cen = Xtr0.mean(0)
Vt = np.linalg.svd(Xtr0 - cen, full_matrices=False)[2]
print('%s ambient d=%d' % (ENC, Xtr0.shape[1]), flush=True)

SUBS = {}
for K in KS:
    for rp in REPS:
        if K == K0:
            if rp == 0:
                SUBS[(K, rp)] = list(range(K0))
        else:
            r = np.random.default_rng(991 * K + rp)
            SUBS[(K, rp)] = sorted(r.choice(K0, size=K, replace=False).tolist())
print('SUBSETS %d' % len(SUBS), flush=True)

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
        BASE[(dd, K, rp)] = dict(P=P, W=W, b=b, MS=class_means(Xsk, ysk, K),
                                 pi_s=np.bincount(ysk, minlength=K) / len(ysk),
                                 mte=mte, ysk=ysk,
                                 idx=[np.where(ysk == c)[0] for c in range(K)])
    print('  dd=%3d probes built' % dd, flush=True)

rows = []
for cn in CORR:
    Z = np.load('%s/%s_%s.npz' % (EMB, ENC, cn))
    Xt0 = Z['s%d' % SEV].astype(np.float64) / sc
    for dd in DIMS:
        for (K, rp), cls in sorted(SUBS.items()):
            J = BASE[(dd, K, rp)]
            Xt = (Xt0[J['mte']] - cen) @ J['P']
            ysk = J['ysk']; W = J['W']; b = J['b']; MS = J['MS']; pi_s = J['pi_s']
            for sd in SEEDS:
                rng = np.random.default_rng(7919 * sd + 131 * dd + 17 * K + 5 * rp
                                            + sum(ord(ch) for ch in cn) % 997)
                pk = rng.dirichlet(np.full(K, 0.5)); pk = np.maximum(pk, 0.02); pk = pk / pk.sum()
                M = int(min(6000, np.floor((np.array([len(i) for i in J['idx']]) / pk).min())))
                nc = np.maximum((pk * M).astype(int), 5)
                sel = np.concatenate([rng.choice(J['idx'][c], size=nc[c], replace=False) for c in range(K)])
                Xts = Xt[sel]; yts = ysk[sel]; n = len(sel)
                mu_t = Xts.mean(0)
                out = dict(corr=cn, dd=dd, K=K, rep=rp, seed=sd, n=n)
                out['a_naive'] = acc(Xts - (mu_t - MS.T @ pi_s), yts, W, b)
                Wf, bf = ridge_probe(Xts, yts, K, lam=1e-2)
                out['a_full'] = acc(Xts, yts, Wf, bf)
                for B in BUDG:
                    lb = rng.choice(n, size=min(B, n), replace=False)
                    Wr, br = ridge_probe(Xts[lb], yts[lb], K, lam=1e-2)
                    out['a_refit%d' % B] = acc(Xts, yts, Wr, br)
                rows.append(out)
    print('done %s' % cn, flush=True)

with open(CIFAR_OUT / ('route_a_%s.pkl' % ENC), 'wb') as f:
    pickle.dump(rows, f)

print('\n=== ROUTE A d-K GRID: %s, %d cells ===' % (ENC, len(rows)), flush=True)


def mn(s, k):
    return float(np.mean([r[k] for r in s]))


def beps(s, eps):
    full = mn(s, 'a_full')
    prev = None
    for B in BUDG:
        v = mn(s, 'a_refit%d' % B)
        if full - v <= eps:
            if prev is None:
                return float(BUDG[0])
            Bp, vp = prev
            num = (full - eps) - vp
            den = (v - vp)
            t = num / den if abs(den) > 1e-12 else 1.0
            t = min(max(t, 0.0), 1.0)
            return float(np.exp(np.log(Bp) + t * (np.log(B) - np.log(Bp))))
        prev = (B, v)
    return float('nan')


for eps in EPSL:
    tab = {}
    print('\n--- eps=%.2f' % eps, flush=True)
    print('  dd    K  rep   naive   full    ' + '  '.join('%5d' % B for B in BUDG) + '     Beps', flush=True)
    for dd in DIMS:
        for (K, rp) in sorted(SUBS.keys()):
            s = [r for r in rows if r['dd'] == dd and r['K'] == K and r['rep'] == rp]
            v = beps(s, eps)
            tab[(dd, K, rp)] = v
            print('  %4d %4d %3d  %.4f  %.4f  ' % (dd, K, rp, mn(s, 'a_naive'), mn(s, 'a_full'))
                  + '  '.join('%.3f' % mn(s, 'a_refit%d' % B) for B in BUDG)
                  + '  %8.1f' % v, flush=True)
    keys = [k for k in tab if np.isfinite(tab[k])]
    print('  usable %d of %d' % (len(keys), len(tab)), flush=True)
    if len(keys) >= 6:
        X = np.array([[np.log(k[0]), np.log(k[1]), 1.0] for k in keys])
        yv = np.log(np.array([tab[k] for k in keys]))
        co = np.linalg.lstsq(X, yv, rcond=None)[0]
        pr = X @ co
        r2 = 1.0 - ((yv - pr) ** 2).sum() / max(((yv - yv.mean()) ** 2).sum(), 1e-12)
        print('  TWO-EXPONENT FIT  log Beps = %+.4f*log d %+.4f*log K %+.4f   R2=%.4f'
              % (co[0], co[1], co[2], r2), flush=True)
        Xd = np.array([[np.log(k[0]), 1.0] for k in keys])
        print('  d-only slope %+.4f' % np.linalg.lstsq(Xd, yv, rcond=None)[0][0], flush=True)
        Xk = np.array([[np.log(k[1]), 1.0] for k in keys])
        print('  K-only slope %+.4f' % np.linalg.lstsq(Xk, yv, rcond=None)[0][0], flush=True)
