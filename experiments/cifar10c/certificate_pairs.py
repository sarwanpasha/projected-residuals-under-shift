"""Per-pair comparison of the plug-in certificate with the mean-confidence null.

For each of the 95 shift pairs per encoder, the rank correlation of each score with
the true Gamma_c is compared by a paired sign-flip permutation test.

Input: results/cifar10c/shift_pairs_<enc>.pkl for both encoders.
"""
import pickle

import numpy as np

from prlb import extra as R2
from prlb.core import ci95
from prlb.paths import CIFAR_OUT

for enc in ['clip', 'dinov2']:
    rows = pickle.load(open(CIFAR_OUT / ('shift_pairs_%s.pkl' % enc), 'rb'))
    print('=' * 72)
    print('%s  n_pairs=%d' % (enc, len(rows)))
    for tag, ck, nk in [('spearman', 'sp_cert', 'sp_conf'), ('kendall', 'kt_cert', 'kt_conf')]:
        a = np.array([float(r[ck]) for r in rows]); b = np.array([float(r[nk]) for r in rows])
        ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
        ma, ha = ci95(a); mb, hb = ci95(b)
        m, p, lo, hi = R2.paired_perm(a, b, n=20000, rng=np.random.default_rng(0))
        win = 100 * float((a > b).mean())
        print('  %-9s cert %.4f+-%.4f  null %.4f+-%.4f' % (tag, ma, ha, mb, hb))
        print('  %-9s diff %+.5f  95pct [%+.5f,%+.5f]  p=%.5f  cert>null on %.1f pct of pairs'
              % (tag, m, lo, hi, p, win))
