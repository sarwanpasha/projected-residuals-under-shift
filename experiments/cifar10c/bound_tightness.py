"""Tightness of the three evaluations of the Theorem 1 bound on real shift pairs.

For each encoder and each bound (source margins, oracle-corrected target margins,
pseudo-labeled target margins) the script reports the mean, the number of violations,
and the median and 90th percentile of the ratio bound / gap over pairs with a gap above
1e-4, overall and stratified by severity.

Input: results/cifar10c/shift_pairs_<enc>.pkl for both encoders.
"""
import pickle

import numpy as np

from prlb.paths import CIFAR_OUT

EPS = 1e-4


def col(rows, k):
    return np.array([r[k] for r in rows], dtype=float)


for enc in ['clip', 'dinov2']:
    rows = pickle.load(open(CIFAR_OUT / ('shift_pairs_%s.pkl' % enc), 'rb'))
    gap = col(rows, 'gap')
    sev = np.array([int(r['sev']) for r in rows])
    print('')
    print('=' * 78)
    print('%s  n=%d  severities %s' % (enc, len(rows), sorted(set(sev.tolist()))))
    print('mean gap %.5f  sd %.5f  n(gap>1e-4)=%d' % (gap.mean(), gap.std(ddof=1), int((gap > EPS).sum())))
    for k in ('bound', 'bound_t', 'bound_h'):
        b = col(rows, k)
        m = gap > EPS
        r = b[m] / gap[m]
        print('')
        print('  [%s] mean %.5f sd %.5f | violations(b<gap) %d/%d | n_ratio %d'
              % (k, b.mean(), b.std(ddof=1), int((b < gap).sum()), len(b), int(m.sum())))
        print('  [%s] ratio median %.4f  p90 %.4f  mean %.4f' % (k, np.median(r), np.percentile(r, 90), r.mean()))
        line_med, line_gap, line_n = [], [], []
        for s in sorted(set(sev.tolist())):
            ms = m & (sev == s)
            line_med.append('%.4f' % np.median(b[ms] / gap[ms]))
            line_gap.append('%.5f' % gap[sev == s].mean())
            line_n.append(str(int(ms.sum())))
        print('  [%s] by severity  median ratio: %s' % (k, '  '.join(line_med)))
        print('  [%s] by severity  mean gap    : %s' % (k, '  '.join(line_gap)))
        print('  [%s] by severity  n           : %s' % (k, '  '.join(line_n)))
