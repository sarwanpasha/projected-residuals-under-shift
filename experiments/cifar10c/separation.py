"""Separation exponent between Route A and Route B.

For every (d, K, subset) cell, the ratio of the Route A requirement B_eps to the
Route B requirement B99 is fitted as log ratio = a log d + b log K + c.  The script
also fits Route A against the parameter count K d and Route B against the estimand
dimension K.

Inputs: results/cifar10c/route_a_<enc>.pkl and route_b_<enc>.pkl for both encoders.
"""
import pickle
import numpy as np

from prlb.paths import CIFAR_OUT

BUD_B = [5, 10, 20, 40, 80, 160, 320, 640]
BUD_A = [25, 50, 100, 200, 400, 800, 1600, 3200]


def mn(s, k):
    return float(np.mean([r[k] for r in s]))


def terp(Bs, vals, tgt):
    prev = None
    for B, v in zip(Bs, vals):
        if v >= tgt:
            if prev is None:
                return float(Bs[0])
            Bp, vp = prev
            den = v - vp
            t = (tgt - vp) / den if abs(den) > 1e-12 else 1.0
            t = min(max(t, 0.0), 1.0)
            return float(np.exp(np.log(Bp) + t * (np.log(B) - np.log(Bp))))
        prev = (B, v)
    return float('nan')


def fit(keys, y, cols):
    X = np.stack([np.array(c, float) for c in cols] + [np.ones(len(y))], 1)
    co = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1.0 - ((y - X @ co) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return co, r2


for enc in ['clip', 'dinov2']:
    r15 = pickle.load(open(CIFAR_OUT / ('route_b_%s.pkl' % enc), 'rb'))
    r16 = pickle.load(open(CIFAR_OUT / ('route_a_%s.pkl' % enc), 'rb'))
    cells = sorted({(r['dd'], r['K'], r['rep']) for r in r15})
    print('\n================ %s, %d cells' % (enc, len(cells)), flush=True)
    for eps in [0.05, 0.02]:
        rows = []
        for (dd, K, rp) in cells:
            s5 = [r for r in r15 if r['dd'] == dd and r['K'] == K and r['rep'] == rp]
            s6 = [r for r in r16 if r['dd'] == dd and r['K'] == K and r['rep'] == rp]
            if not s5 or not s6:
                continue
            lo, hi = mn(s5, 'a_naive'), mn(s5, 'a_orcb')
            if hi - lo < 0.005:
                continue
            b99 = terp(BUD_B, [mn(s5, 'a_cnt%d' % B) for B in BUD_B], lo + 0.99 * (hi - lo))
            full = mn(s6, 'a_full')
            beps = terp(BUD_A, [mn(s6, 'a_refit%d' % B) for B in BUD_A], full - eps)
            if not (np.isfinite(b99) and np.isfinite(beps)):
                continue
            rows.append((dd, K, b99, beps))
        if len(rows) < 8:
            print('  eps=%.2f too few cells (%d)' % (eps, len(rows))); continue
        dd = np.array([r[0] for r in rows], float)
        KK = np.array([r[1] for r in rows], float)
        b99 = np.array([r[2] for r in rows]); beps = np.array([r[3] for r in rows])
        ld, lk = np.log(dd), np.log(KK)
        print('  --- eps=%.2f, %d cells' % (eps, len(rows)), flush=True)
        co, r2 = fit(None, np.log(beps / b99), [ld, lk])
        print('    SEPARATION  log(B_A/B_B) = %+.4f log d %+.4f log K %+.4f   R2=%.4f'
              % (co[0], co[1], co[2], r2), flush=True)
        co1, r21 = fit(None, np.log(beps), [ld + lk])
        print('    Route A vs parameter count: log B_A = %+.4f log(K d) %+.4f   R2=%.4f'
              % (co1[0], co1[1], r21), flush=True)
        co2, r22 = fit(None, np.log(b99), [lk])
        print('    Route B vs estimand dim:    log B_B = %+.4f log K %+.4f       R2=%.4f'
              % (co2[0], co2[1], r22), flush=True)
        co3, r23 = fit(None, np.log(b99), [ld + lk])
        print('    Route B vs parameter count: log B_B = %+.4f log(K d) %+.4f   R2=%.4f'
              % (co3[0], co3[1], r23), flush=True)
        print('    median ratio B_A/B_B by d:', flush=True)
        for u in sorted(set(dd)):
            m = dd == u
            print('      d=%4d  median %6.2f   n=%d  (median B_A %7.1f, B_B %6.1f)'
                  % (u, np.median(beps[m] / b99[m]), int(m.sum()),
                     np.median(beps[m]), np.median(b99[m])), flush=True)
        print('    median ratio B_A/B_B by K:', flush=True)
        for u in sorted(set(KK)):
            m = KK == u
            print('      K=%4d  median %6.2f   n=%d' % (u, np.median(beps[m] / b99[m]), int(m.sum())), flush=True)
