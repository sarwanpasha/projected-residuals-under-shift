"""CIFAR-10 to CIFAR-10-C with frozen foundation-model embeddings: 19 corruptions x 5
severities = 95 shift pairs per encoder (Figure 2 and the real-data rows of the text).

Usage:  python experiments/cifar10c/shift_pairs.py {clip|dinov2} [tta]

A ridge probe is trained on the CIFAR-10 training embeddings and the clean test split
plays the role of the source.  For every shift pair the script records:
  * the collinearity of the class displacements with the common displacement, the
    residual-to-common norm ratio and the effective rank of the residuals;
  * label-free accuracies: no correction, BN standardization, CORAL, T3A, the
    common-mean correction, the per-class oracle and the per-class correction with
    predicted assignments (TENT and SHOT upper bounds when "tta" is given);
  * the Theorem 1 bound evaluated with source margins, with oracle-corrected target
    margins and with pseudo-labeled target margins;
  * the plug-in certificate and the mean-confidence null, ranked against Gamma_c;
  * label-budgeted arms at B in {20, 40, 80, 160, 240, 400}: the probe refitted with a
    fixed ridge penalty, the probe refitted with a cross-validated penalty, and the
    certificate-driven selective corrector.

Output: results/cifar10c/shift_pairs_<enc>.pkl
"""
import sys, pickle, zlib
import numpy as np
from scipy import stats
from prlb.core import *
from prlb.paths import CIFAR_EMB, CIFAR_OUT, CORRUPTIONS, ensure

EMB = str(CIFAR_EMB)
ensure(CIFAR_OUT)
K = 10
CORR = CORRUPTIONS
SEVS = [1, 2, 3, 4, 5]
BUDGETS = [20, 40, 80, 160, 240, 400]
ENC = sys.argv[1]
DO_TTA = (len(sys.argv) > 2 and sys.argv[2] == 'tta')

Xtr = np.load('%s/%s_train.npy' % (EMB, ENC)).astype(np.float64)
ytr = np.load('%s/labels_train.npy' % EMB)
Xs = np.load('%s/%s_clean.npy' % (EMB, ENC)).astype(np.float64)
ys = np.load('%s/labels_test.npy' % EMB)
sc = np.linalg.norm(Xtr, axis=1).mean()
Xtr = Xtr / sc; Xs = Xs / sc
W, b = ridge_probe(Xtr, ytr, K, lam=1e-2)
print('%s d=%d  probe acc: train=%.4f  clean-test=%.4f' % (ENC, Xtr.shape[1], acc(Xtr, ytr, W, b), acc(Xs, ys, W, b)), flush=True)
MS = class_means(Xs, ys, K)
pis = np.bincount(ys, minlength=K) / len(ys)
Vd = disc_basis(W); Pd = Vd.T @ Vd

rows = []

for cn in CORR:
    Z = np.load('%s/%s_%s.npz' % (EMB, ENC, cn))
    for sv in SEVS:
        Xt = Z['s%d' % sv].astype(np.float64) / sc
        yt = ys
        MT = class_means(Xt, yt, K)
        deltas = MT - MS
        dbar = Xt.mean(0) - Xs.mean(0)
        rho = deltas - dbar[None, :]
        G = all_gamma_caps(rho, W)
        cosd = np.array([float(deltas[c] @ dbar / (np.linalg.norm(deltas[c]) * np.linalg.norm(dbar) + 1e-12)) for c in range(K)])
        nrho = np.linalg.norm(rho, axis=1); ndb = float(np.linalg.norm(dbar))
        sv_ = np.linalg.svd(rho, compute_uv=False)
        p_ = (sv_ ** 2) / max((sv_ ** 2).sum(), 1e-30)
        erank = float(np.exp(-(p_ * np.log(p_ + 1e-30)).sum()))
        a_none = acc(Xt, yt, W, b)
        a_gl = acc(Xt - dbar, yt, W, b)
        a_or = acc(Xt - deltas[yt], yt, W, b)
        plg = predict(Xt - dbar, W, b)
        a_pc = acc(Xt - deltas[plg], yt, W, b)
        bnd, _, _ = theory_bound(Xs, ys, W, b, rho, pis)
        Xor = Xt - deltas[yt]
        bnd_t, _, _ = theory_bound(Xor, yt, W, b, rho, pis)
        Xg_ = Xt - dbar; pl_ = predict(Xg_, W, b)
        bnd_h, _, _ = theory_bound(Xg_, pl_, W, b, rho, pis)
        r = dict(enc=ENC, corr=cn, sev=sv, a_none=a_none, a_gl=a_gl, a_or=a_or, a_pc=a_pc,
                 gap=a_or - a_gl, bound=bnd, cos_mean=float(cosd.mean()), cos_min=float(cosd.min()),
                 rho_ratio=float(nrho.mean() / (ndb + 1e-12)), erank=erank, Gmax=float(G.max()),
                 a_bn=acc(bn_align(Xt, Xs), yt, W, b), a_coral=acc(coral_align(Xt, Xs), yt, W, b),
                 a_t3a=float(np.mean(t3a(Xt, W, b, K) == yt)), bound_t=bnd_t, bound_h=bnd_h,
                 asg_acc=float(np.mean(plg == yt)),
                 pred_changed=float(np.mean(predict(Xt - deltas[plg], W, b) != predict(Xg_, W, b))))
        Ghat, _ = plugin_certificate(Xt, Xs, ys, W, b, K, dbar)
        r['sp_cert'] = float(stats.spearmanr(Ghat, G).correlation)
        r['kt_cert'] = float(stats.kendalltau(Ghat, G).correlation)
        cf = mean_conf_score(Xt, W, b, K, dbar)
        r['sp_conf'] = float(stats.spearmanr(cf, G).correlation)
        r['kt_conf'] = float(stats.kendalltau(cf, G).correlation)
        r['G'] = G; r['Ghat'] = Ghat; r['conf'] = cf
        qr = np.random.default_rng(zlib.crc32(("%s|%d" % (cn, sv)).encode()) % (2 ** 31))
        for B in BUDGETS:
            q = qr.choice(len(yt), size=B, replace=False)
            Wr, br = ridge_probe(Xt[q], yt[q], K, lam=1e-2)
            r['refit_%d' % B] = acc(Xt, yt, Wr, br)
            Wc, bc, lmb = ridge_cv(Xt[q], yt[q], K)
            r['refitcv_%d' % B] = acc(Xt, yt, Wc, bc)
            r['lam_%d' % B] = lmb
            Xc, S = selective_corrector(Xt, yt, Xs, ys, W, b, K, dbar, B, qr, n_min=8)
            r['ours_%d' % B] = acc(Xc, yt, W, b)
            r['nsel_%d' % B] = len(S)
        if DO_TTA:
            r['a_tent'] = tta_best_oracle(Xt, yt, W, b, 'tent', steps=100, lrs=(1e-4, 1e-3))
            r['a_shot'] = tta_best_oracle(Xt, yt, W, b, 'shot', steps=100, lrs=(1e-4, 1e-3))
        rows.append(r)
    print('done', cn, flush=True)

with open(CIFAR_OUT / ('shift_pairs_%s.pkl' % ENC), 'wb') as f:
    pickle.dump(rows, f)

n = len(rows)
f = lambda k: ci95([r[k] for r in rows if k in r])
print('')
print('=== REAL SHIFT PAIRS: %s, n=%d (19 corruptions x 5 severities) ===' % (ENC, n))
print('COLLINEARITY (multi-class, K=10)  cos(delta_c, dbar) mean=%.4f+-%.4f  min-over-classes mean=%.4f+-%.4f'
      % (f('cos_mean')[0], f('cos_mean')[1], f('cos_min')[0], f('cos_min')[1]))
print('RESIDUAL/GLOBAL norm ratio mean=%.4f+-%.4f   residual effective rank mean=%.3f+-%.3f (max %d)'
      % (f('rho_ratio')[0], f('rho_ratio')[1], f('erank')[0], f('erank')[1], K))
print('')
print('%-28s %18s' % ('method (label-free)', 'accuracy'))
for k, nm in [('a_none', 'no correction'), ('a_bn', 'BN / standardization'), ('a_coral', 'CORAL'),
              ('a_t3a', 'T3A prototypes'), ('a_tent', 'TENT (linear head)'), ('a_shot', 'SHOT (linear head)'),
              ('a_gl', 'common-mean correction')]:
    if any(k in r for r in rows):
        m, c = f(k); print('%-28s %9.4f +- %-7.4f' % (nm, m, c))
print('%-28s %9.4f +- %-7.4f   (uses true label at inference)' % ('per-class ORACLE', f('a_or')[0], f('a_or')[1]))
print('%-28s %9.4f +- %-7.4f   (predicted assignment, deployable)' % ('per-class, pred assign', f('a_pc')[0], f('a_pc')[1]))
print('')
print('%-28s %18s %18s' % ('method (B labels)', 'accuracy', 'gain over global'))
for B in BUDGETS:
    for tag, nm in [('refit', 'refit probe'), ('refitcv', 'refit probe CV'), ('ours', 'selective corrector')]:
        m, c = f('%s_%d' % (tag, B))
        d, dc = ci95([r['%s_%d' % (tag, B)] - r['a_gl'] for r in rows])
        print('%-28s %9.4f +- %-7.4f %9.4f +- %-7.4f' % ('%s B=%d' % (nm, B), m, c, d, dc))
print('')
vio = sum(1 for r in rows if r['bound'] < r['gap'] - 1e-12)
print('THEORY  measured gap=%.5f+-%.5f  Thm1 bound=%.5f+-%.5f  violations=%d/%d'
      % (f('gap')[0], f('gap')[1], f('bound')[0], f('bound')[1], vio, n))
vio_t = sum(1 for r in rows if r['bound_t'] < r['gap'] - 1e-12)
vio_h = sum(1 for r in rows if r['bound_h'] < r['gap'] - 1e-12)
print('MODEL-CHECK  bound from SOURCE margins: violations=%d/%d  (pure-translation model assumed)' % (vio, n))
print('MODEL-CHECK  bound from ORACLE-CORRECTED TARGET margins: value=%.5f+-%.5f violations=%d/%d' % (f('bound_t')[0], f('bound_t')[1], vio_t, n))
print('MODEL-CHECK  bound from PSEUDO-LABELLED target margins (label-free): value=%.5f+-%.5f violations=%d/%d' % (f('bound_h')[0], f('bound_h')[1], vio_h, n))
print('ASSIGNMENT  accuracy of predicted class used by deployable per-class corrector = %.4f+-%.4f' % f('asg_acc'))
print('ASSIGNMENT  fraction of predictions CHANGED by per-class correction with predicted assignment = %.5f+-%.5f' % f('pred_changed'))
tight = [r['bound_t'] / r['gap'] for r in rows if r['gap'] > 1e-4]
print('TIGHTNESS on pairs with gap>1e-4 (n=%d): median=%.2fx  p90=%.2fx' % (len(tight), np.median(tight), np.quantile(tight, 0.9)))
neg = sum(1 for r in rows if r['a_or'] <= r['a_gl'])
print('PAIRS WHERE ORACLE <= GLOBAL (recovery denominator non-positive): %d/%d' % (neg, n))
Ga = np.concatenate([r['G'] for r in rows]); Gh = np.concatenate([r['Ghat'] for r in rows]); Cf = np.concatenate([r['conf'] for r in rows])
sl, ic, rv, pv, se = stats.linregress(Ga, Gh)
print('')
print('CERTIFICATE pooled over %d class-instances' % len(Ga))
print('  plug-in vs true Gamma   Spearman=%.4f  Kendall=%.4f' % (stats.spearmanr(Gh, Ga).correlation, stats.kendalltau(Gh, Ga).correlation))
print('  mean-conf null vs Gamma Spearman=%.4f  Kendall=%.4f' % (stats.spearmanr(Cf, Ga).correlation, stats.kendalltau(Cf, Ga).correlation))
print('  scale calibration Ghat = %.4f*Gamma + %.4f   R2=%.4f' % (sl, ic, rv ** 2))
print('  per-pair Spearman: cert %.4f+-%.4f   conf-null %.4f+-%.4f' % (f('sp_cert')[0], f('sp_cert')[1], f('sp_conf')[0], f('sp_conf')[1]))
print('')
print('MEDIAN CV LAMBDA by budget: '+' '.join('B=%d %g'%(B,float(np.median([r['lam_%d'%B] for r in rows]))) for B in BUDGETS))
