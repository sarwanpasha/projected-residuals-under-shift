"""Label-free leaderboard with paired inference (Table 1).

Usage:  python experiments/cifar10c/leaderboard.py {clip|dinov2}

Every corrector is run on all 19 corruptions at all 5 severities, which gives 95
paired shift pairs per encoder.  Correctors: no correction, the common-mean
correction, scale-only alignment, BN alignment, CORAL, LAME with its default graph
strength and with an oracle-tuned strength, T3A, TENT, SHOT, the common-mean
correction followed by LAME, and the per-class oracle.  Each comparison is a paired
sign-flip permutation test with a percentile bootstrap interval on the mean paired
difference.

Output: results/cifar10c/leaderboard_<enc>.pkl
"""
import sys, pickle, time
import numpy as np

from prlb.core import ridge_probe, acc, bn_align, coral_align, t3a, class_means, tta_baseline
from prlb import extra as R2
from prlb.paths import CIFAR_EMB, CIFAR_OUT, CORRUPTIONS, ensure

EMB = str(CIFAR_EMB)
K = 10
CORR = CORRUPTIONS
SEVS = ['s1', 's2', 's3', 's4', 's5']
ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
METHODS = ['naive', 'mean', 'scale', 'bn', 'coral', 'lame_def', 'lame_orc',
           't3a', 'tent', 'shot', 'mean_lame', 'oracle_pc']


def main():
    enc = sys.argv[1]
    t0 = time.time()
    Xtr = np.load('%s/%s_train.npy' % (EMB, enc)).astype(np.float64)
    ytr = np.load('%s/labels_train.npy' % EMB)
    yte = np.load('%s/labels_test.npy' % EMB)
    Xcl = np.load('%s/%s_clean.npy' % (EMB, enc)).astype(np.float64)
    W, b = ridge_probe(Xtr, ytr, K, lam=1e-2)
    MS = class_means(Xtr, ytr, K)
    ensure(CIFAR_OUT)
    print('=== PAIRED LEADERBOARD: %s   d=%d   clean acc %.4f ==='
          % (enc, Xtr.shape[1], acc(Xcl, yte, W, b)), flush=True)

    rows = []
    for cn in CORR:
        try:
            Z = np.load('%s/%s_%s.npz' % (EMB, enc, cn))
        except Exception:
            print('  no file for %s' % cn, flush=True)
            continue
        for sv in SEVS:
            if sv not in Z:
                continue
            Xt = Z[sv].astype(np.float64)
            r = dict(corr=cn, sev=sv, enc=enc)
            r['naive'] = acc(Xt, yte, W, b)
            Xm = R2.mean_align(Xt, Xtr)
            r['mean'] = acc(Xm, yte, W, b)
            r['scale'] = acc(R2.scale_align(Xt, Xtr), yte, W, b)
            r['bn'] = acc(bn_align(Xt, Xtr), yte, W, b)
            r['coral'] = acc(coral_align(Xt, Xtr), yte, W, b)
            deltas = np.array([MS[c] - Xt[yte == c].mean(0) for c in range(K)])
            r['oracle_pc'] = acc(Xt + deltas[yte], yte, W, b)
            r['t3a'] = float((t3a(Xt, W, b, K) == yte).mean())
            for nm, md in (('tent', 'tent'), ('shot', 'shot')):
                r[nm] = np.nan
                try:
                    Xa, Wa, ba = tta_baseline(Xt, W, b, mode=md)
                    r[nm] = acc(Xa, yte, Wa, ba)
                except Exception as e:
                    print('      %s unavailable: %s' % (nm, type(e).__name__), flush=True)
            idx = R2.knn_index(Xt.astype(np.float32), 5)
            r['lame_def'] = float((R2.lame(Xt, W, b, K, alpha=1.0, idx=idx) == yte).mean())
            best = -1.0
            for al in ALPHAS:
                v = float((R2.lame(Xt, W, b, K, alpha=al, idx=idx) == yte).mean())
                if v > best:
                    best, r['lame_alpha'] = v, al
            r['lame_orc'] = best
            idxm = R2.knn_index(Xm.astype(np.float32), 5)
            r['mean_lame'] = float((R2.lame(Xm, W, b, K, alpha=1.0, idx=idxm) == yte).mean())
            rows.append(r)
        print('  %-20s  %6.1f s  pairs=%d' % (cn, time.time() - t0, len(rows)), flush=True)

    with open(CIFAR_OUT / ('leaderboard_%s.pkl' % enc), 'wb') as f:
        pickle.dump(rows, f)

    print('\n  === MEAN ACCURACY OVER %d PAIRED SHIFTS (%s) ==='
          % (len(rows), enc), flush=True)
    for m in METHODS:
        v = np.array([r.get(m, np.nan) for r in rows], dtype=np.float64)
        ok = np.isfinite(v)
        if ok.sum() == 0:
            print('    %-11s unavailable' % m, flush=True)
            continue
        sd = float(v[ok].std(ddof=1))
        print('    %-11s %.4f  sd %.4f  se %.4f  n=%d'
              % (m, v[ok].mean(), sd, sd / np.sqrt(ok.sum()), int(ok.sum())), flush=True)

    rng = np.random.default_rng(5)
    pairs = [('mean', 'naive'), ('bn', 'mean'), ('scale', 'naive'), ('bn', 'scale'),
             ('coral', 'mean'), ('lame_def', 'mean'), ('lame_orc', 'mean'),
             ('t3a', 'mean'), ('tent', 'mean'), ('shot', 'mean'),
             ('mean_lame', 'mean'), ('oracle_pc', 'mean')]
    print('\n  === PAIRED SIGN FLIP PERMUTATION TESTS ===', flush=True)
    print('    comparison            mean diff      95 pct interval        p', flush=True)
    res = {}
    for a, c in pairs:
        va = np.array([r.get(a, np.nan) for r in rows], dtype=np.float64)
        vc = np.array([r.get(c, np.nan) for r in rows], dtype=np.float64)
        ok = np.isfinite(va) & np.isfinite(vc)
        if ok.sum() < 5:
            print('    %-9s vs %-9s  unavailable' % (a, c), flush=True)
            continue
        m, p, lo, hi = R2.paired_perm(va[ok], vc[ok], n=20000, rng=rng)
        res[(a, c)] = (m, p, lo, hi, int(ok.sum()))
        print('    %-9s vs %-9s  %+.5f   [%+.5f, %+.5f]   %.5f  n=%d'
              % (a, c, m, lo, hi, p, int(ok.sum())), flush=True)

    al = [r.get('lame_alpha', np.nan) for r in rows]
    print('\n    oracle graph strength picked, median %.2f' % float(np.nanmedian(al)),
          flush=True)
    print('  total %.1f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
