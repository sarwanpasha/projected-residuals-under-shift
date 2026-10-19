"""Operating characteristic of the certificate against the confidence null.

Both scores are swept over all thresholds on the same class instances produced by
certificate.py and compared by AUROC, AUPRC and precision at matched recall.  The
positive label is Gamma_c > 0.  The base rate of positives is reported alongside the
confusion counts of the executable rule, since it fixes what precision and NPV mean.

Input: results/synthetic/certificate.pkl
"""
import pickle

import numpy as np

from prlb.paths import SYN_OUT

rows = pickle.load(open(SYN_OUT / 'certificate.pkl', 'rb'))
G = np.concatenate([np.asarray(r['G']) for r in rows]).astype(float)
Gh = np.concatenate([np.asarray(r['Ghat']) for r in rows]).astype(float)
cf = np.concatenate([np.asarray(r['conf']) for r in rows]).astype(float)
ls = np.concatenate([np.asarray(r['loss']) for r in rows]).astype(float)
th = np.concatenate([np.asarray(r['thr']) for r in rows]).astype(float)
N = G.size
print('instances %d' % N)
for name, y in [('G>0', (G > 0)), ('loss>0', (ls > 0))]:
    print('  label %-8s positives %d (%.4f)' % (name, int(y.sum()), y.mean()))
y = (G > 0)


def auroc(s, y):
    o = np.argsort(s); r = np.empty(N, float); r[o] = np.arange(1, N + 1)
    np1 = int(y.sum()); n0 = N - np1
    return float((r[y].sum() - np1 * (np1 + 1) / 2.0) / (np1 * n0))


def pr(s, y):
    o = np.argsort(-s); yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(~yy)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(int(y.sum()), 1)
    ap = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    return prec, rec, ap


def prec_at_rec(s, y, target):
    prec, rec, _ = pr(s, y)
    i = int(np.argmin(np.abs(rec - target)))
    return float(prec[i]), float(rec[i])


print()
for nm, s in [('certificate Ghat', Gh), ('confidence null', cf)]:
    a = auroc(s, y); _, _, ap = pr(s, y)
    p50, r50 = prec_at_rec(s, y, 0.500)
    p69, r69 = prec_at_rec(s, y, 0.690)
    p90, r90 = prec_at_rec(s, y, 0.900)
    print('%-17s AUROC %.4f  AUPRC %.4f' % (nm, a, ap))
    print('%-17s precision at recall %.3f = %.4f | at %.3f = %.4f | at %.3f = %.4f'
          % ('', r50, p50, r69, p69, r90, p90))
print()
print('rule as executed (Ghat > thr):')
pred = Gh > th
tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
print('  TP %d FP %d FN %d TN %d' % (tp, fp, fn, tn))
print('  precision %.4f recall %.4f NPV %.4f accuracy %.4f'
      % (tp / max(tp + fp, 1), tp / max(tp + fn, 1), tn / max(tn + fn, 1), (tp + tn) / N))
print('  base-rate NPV of a random negative prediction = %.4f' % (1.0 - y.mean()))
