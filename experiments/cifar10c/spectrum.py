"""Ridge effective dimension of the PCA-truncated embeddings used by the grids.

The refit requirement of Route A scales with the effective dimension of the features
rather than with the ambient dimension.  For d in {8, ..., 512} the script reports the
ridge effective dimension sum_i l_i / (l_i + lam / B) of the normalized source features
at the penalty and budgets used in the experiments, and its log-log slope in d.
"""
import numpy as np

from prlb import extra as R2
from prlb.paths import CIFAR_EMB

DIMS = [8, 16, 32, 64, 128, 256, 512]
for enc in ['clip', 'dinov2']:
    X = np.load(CIFAR_EMB / ('%s_train.npy' % enc)).astype(np.float64)
    X = X / np.linalg.norm(X, axis=1).mean()
    lam = R2.spectrum(X)
    print('=== %s: ambient d=%d, participation ratio %.2f, effective rank %.2f'
          % (enc, X.shape[1], R2.pr_dim(lam), R2.entropy_dim(lam)))
    for B in (100, 400, 1600):
        de = [R2.ridge_dim(lam[:d], 1e-2 / B) for d in DIMS]
        sl = np.polyfit(np.log(DIMS), np.log(de), 1)[0]
        print('  B=%-5d d_eff %s   slope %.3f' % (B, ' '.join('%.1f' % v for v in de), sl))
