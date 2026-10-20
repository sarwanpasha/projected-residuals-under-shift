"""Supervised target references and tuned TENT and SHOT on the corruption pairs.

Usage:  python experiments/cifar10c/reference.py {clip|dinov2}

For each of the 95 corruption pairs the target set is split in two class-stratified
halves.  On the evaluation half the script reports no correction, the common-mean
correction and the per-class oracle, together with three references fitted on the
labels of the other half: the target ridge probe with its penalty chosen by
cross-validation, the probe refitted toward the source probe with every label of
that half, and a 20-nearest-neighbour classifier on the target labels.  The best of
the three estimates what a classifier of the frozen features can reach on the target.

TENT and SHOT are run on the full target set at every configuration of
prlb.core.TTA_GRID (Adam or SGD, four learning rates, with and without the SAR
reliable-sample filter).  The configuration reported for a corruption is the one
with the best mean accuracy over the other eighteen corruptions.

Output: results/cifar10c/reference_<enc>.pkl
"""
import pickle
import sys

import numpy as np

from prlb import core as R
from prlb import extra as R2
from prlb.paths import CIFAR_EMB, CIFAR_OUT, CORRUPTIONS, ensure

K = 10
LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
ENC = sys.argv[1]


def knn_predict(Xa, ya, Xb, K, k=20, chunk=2000):
    k = min(k, len(Xa) - 1)
    A = Xa / np.linalg.norm(Xa, axis=1, keepdims=True)
    out = []
    for i in range(0, len(Xb), chunk):
        Bq = Xb[i:i + chunk]; Bq = Bq / np.linalg.norm(Bq, axis=1, keepdims=True)
        S = Bq @ A.T
        nn = np.argpartition(-S, k, axis=1)[:, :k]
        V = np.zeros((len(Bq), K))
        np.add.at(V, (np.repeat(np.arange(len(Bq)), k), ya[nn].ravel()), 1.0)
        out.append(V.argmax(1))
    return np.concatenate(out)


def main():
    ensure(CIFAR_OUT)
    Xtr = np.load(CIFAR_EMB / ('%s_train.npy' % ENC)).astype(np.float64)
    ytr = np.load(CIFAR_EMB / 'labels_train.npy')
    yte = np.load(CIFAR_EMB / 'labels_test.npy')
    W, b = R.ridge_probe(Xtr, ytr, K, lam=1e-2)
    MS = R.class_means(Xtr, ytr, K)
    rows = []
    for cn in CORRUPTIONS:
        Z = np.load(CIFAR_EMB / ('%s_%s.npz' % (ENC, cn)))
        for sv in range(1, 6):
            Xt = Z['s%d' % sv].astype(np.float64)
            rng = np.random.default_rng(97 * sv + sum(ord(ch) for ch in cn))
            fit = np.zeros(len(yte), bool)
            for c in range(K):
                ic = rng.permutation(np.where(yte == c)[0])
                fit[ic[:len(ic) // 2]] = True
            ev = ~fit
            Xa, ya, Xb, yb = Xt[fit], yte[fit], Xt[ev], yte[ev]
            D = R.class_means(Xt, yte, K) - MS
            dbar = Xt.mean(0) - Xtr.mean(0)
            r = dict(corr=cn, sev=sv)
            r['naive'] = R.acc(Xb, yb, W, b)
            r['mean'] = R.acc(Xb - dbar, yb, W, b)
            r['oracle'] = R.acc(Xb - D[yb], yb, W, b)
            Wt, bt, lt = R.ridge_cv(Xa, ya, K)
            r['tprobe'] = R.acc(Xb, yb, Wt, bt); r['tprobe_lam'] = lt
            Wa, ba, la = R2.ridge_to_source_cv(Xa, ya, K, W, b, LAMS, np.random.default_rng(sv))
            r['tanch'] = R.acc(Xb, yb, Wa, ba); r['tanch_lam'] = la
            r['tknn'] = float(np.mean(knn_predict(Xa, ya, Xb, K) == yb))
            r['tbest'] = max(r['tprobe'], r['tanch'], r['tknn'])
            r['tent_grid'] = R.tta_grid(Xt, yte, W, b, 'tent')
            r['shot_grid'] = R.tta_grid(Xt, yte, W, b, 'shot')
            rows.append(r)
            print('%s s%d mean %.4f orc %.4f tprobe %.4f tanch %.4f tknn %.4f tent* %.4f shot* %.4f'
                  % (cn, sv, r['mean'], r['oracle'], r['tprobe'], r['tanch'], r['tknn'],
                     max(r['tent_grid'].values()), max(r['shot_grid'].values())), flush=True)
    with open(CIFAR_OUT / ('reference_%s.pkl' % ENC), 'wb') as f:
        pickle.dump(rows, f)
    print('\n=== REFERENCES: %s, %d pairs ===' % (ENC, len(rows)))
    for k in ('naive', 'mean', 'oracle', 'tprobe', 'tanch', 'tknn', 'tbest'):
        m, h = R.ci95([r[k] for r in rows])
        print('  %-8s %.5f +- %.5f' % (k, m, h))
    d = R.ci95([r['oracle'] - r['tbest'] for r in rows])
    print('  oracle - tbest %+.5f +- %.5f' % d)
    for nm in ('tent', 'shot'):
        v = R.select_held_out(rows, nm + '_grid', 'corr')
        best = max(R.TTA_GRID, key=lambda c: np.mean([r[nm + '_grid'][c] for r in rows]))
        print('  %s held-out selection %.5f   best single config %s %.5f'
              % (nm, v.mean(), best, np.mean([r[nm + '_grid'][best] for r in rows])))


if __name__ == '__main__':
    main()
