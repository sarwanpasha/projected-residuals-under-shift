"""Label-free plug-in certificate against the mean-confidence null.

The plug-in estimate of Gamma_c uses pseudo-labels from the globally corrected probe
and is compared with the true Gamma_c and with the true per-class loss by Spearman and
Kendall correlation, over 500 cells and 4000 class instances.  The executable decision
rule declares a class safe when the plug-in certificate is below the calibrated
q = 0.02 margin quantile, and its confusion counts are reported.

Output: results/synthetic/certificate.pkl
"""
import numpy as np, pickle
from scipy import stats
from prlb.core import *
from prlb.paths import SYN_OUT, ensure

ensure(SYN_OUT)

D = 50; K = 8; NPC = 4000; NTR = 1500; NCAL = 1250; SEP = 4.0; DBAR_NORM = 5.0
RS = [0.5, 1.0, 1.5, 2.0, 3.0]
THS = [0.0, 20.0, 40.0, 60.0, 80.0]
SEEDS = list(range(20))
Q = 0.02
cells = []
for sd in SEEDS:
    rng = np.random.default_rng(1000 + sd)
    X, y, _, _ = make_source(rng, d=D, K=K, n=NPC, sep=SEP)
    Xtr, ytr, Xrest, yrest = split_by_class(X, y, K, NTR, rng)
    Xca, yca, Xte, yte = split_by_class(Xrest, yrest, K, NCAL, rng)
    W, b = ridge_probe(Xtr, ytr, K)
    pis = np.full(K, 1.0 / K)
    g = np.random.default_rng(3000 + sd).standard_normal(D)
    dbar = g / np.linalg.norm(g) * DBAR_NORM
    thr = np.array([pos_quantile(margins(Xca[yca == c], yca[yca == c], W, b), Q) for c in range(K)])
    for r in RS:
        for th in THS:
            rho = make_residuals(np.random.default_rng(4000 + sd), W, pis, r, np.deg2rad(th))
            G = all_gamma_caps(rho, W)
            deltas = dbar[None, :] + rho
            Xt = Xte + deltas[yte]
            Ghat, _ = plugin_certificate(Xt, Xtr, ytr, W, b, K, dbar)
            conf = mean_conf_score(Xt, W, b, K, dbar)
            loss = np.zeros(K)
            for c in range(K):
                m = (yte == c)
                loss[c] = float(np.mean((predict((Xt - deltas[yte])[m], W, b) == c) & (predict((Xt - dbar)[m], W, b) != c)))
            cells.append(dict(seed=sd, r=r, th=th, G=G, Ghat=Ghat, conf=conf, loss=loss, thr=thr))
with open(SYN_OUT / 'certificate.pkl', 'wb') as f:
    pickle.dump(cells, f)
Ga = np.concatenate([c['G'] for c in cells]); Gh = np.concatenate([c['Ghat'] for c in cells])
Cf = np.concatenate([c['conf'] for c in cells]); Ls = np.concatenate([c['loss'] for c in cells])
n = len(Ga)
print('CERTIFICATE  cells=%d class-instances=%d  (K=%d, seeds=%d, %d r-values, %d angles)'
      % (len(cells), n, K, len(SEEDS), len(RS), len(THS)))
def rep(name, x, ref):
    s = stats.spearmanr(x, ref); k = stats.kendalltau(x, ref)
    z = np.arctanh(np.clip(s.correlation, -0.999999, 0.999999)); se = 1.0 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    print('%-26s Spearman=%.4f [%.4f,%.4f] p=%.2e   Kendall=%.4f p=%.2e' % (name, s.correlation, lo, hi, s.pvalue, k.correlation, k.pvalue))
rep('plug-in cert vs true Gamma', Gh, Ga)
rep('mean-conf null vs true Gam', Cf, Ga)
rep('plug-in cert vs true loss', Gh, Ls)
rep('mean-conf null vs true los', Cf, Ls)
sl, ic, rv, pv, se = stats.linregress(Ga, Gh)
print('SCALE CALIBRATION  Ghat = %.4f * Gamma + %.4f   R2=%.4f' % (sl, ic, rv ** 2))
rel = np.abs(Gh - Ga) / np.maximum(np.abs(Ga), 1e-9)
print('RELATIVE SCALE ERROR  median=%.4f  p90=%.4f' % (np.median(rel), np.quantile(rel, 0.9)))
TH = np.concatenate([c['thr'] for c in cells])
pred_safe = Gh <= TH; true_safe = Ls <= Q
tp = int(np.sum(pred_safe & true_safe)); fp = int(np.sum(pred_safe & ~true_safe))
fn = int(np.sum(~pred_safe & true_safe)); tn = int(np.sum(~pred_safe & ~true_safe))
print('EXECUTABLE DECISION RULE at q=%.2f: TP=%d FP=%d FN=%d TN=%d  acc=%.4f  precision=%.4f  recall=%.4f'
      % (Q, tp, fp, fn, tn, (tp + tn) / n, tp / max(tp + fp, 1), tp / max(tp + fn, 1)))
best = max(((np.mean((Cf <= t) == true_safe), t) for t in np.quantile(Cf, np.linspace(0.01, 0.99, 99))))
print('MEAN-CONF NULL with ORACLE-TUNED threshold: best acc=%.4f' % best[0])
