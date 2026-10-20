"""Core numerical routines.

A linear probe is a pair (W, b) with W of shape (K, d) and b of shape (K,); its scores
are X @ W.T + b.  The module contains the ridge probe, margins and the projected
residual Gamma_c, the indicator bound of Theorem 1, the label-free correctors and
baselines, the label-budgeted selective corrector, and the controlled generator used
by the synthetic experiments.
"""
import numpy as np


def ridge_probe(X, y, K, lam=1e-2):
    """One-versus-rest least-squares probe with an unpenalized bias and ridge lam."""
    n, d = X.shape
    Y = np.zeros((n, K)); Y[np.arange(n), y] = 1.0
    Xa = np.c_[X, np.ones(n)]
    A = Xa.T @ Xa + lam * np.eye(d + 1)
    Wa = np.linalg.solve(A, Xa.T @ Y)
    return Wa[:d].T, Wa[d]


def scores(X, W, b):
    return X @ W.T + b


def predict(X, W, b):
    return scores(X, W, b).argmax(1)


def acc(X, y, W, b):
    return float(np.mean(predict(X, W, b) == y))


def margins(X, y, W, b):
    S = scores(X, W, b)
    idx = np.arange(len(y))
    t = S[idx, y]
    S2 = S.copy(); S2[idx, y] = -np.inf
    return t - S2.max(1)


def gamma_cap(rho_c, W, c):
    """Gamma_c = max_{j != c} <rho_c, w_j - w_c>."""
    v = (W - W[c]) @ rho_c
    v[c] = -np.inf
    return float(v.max())


def all_gamma_caps(rhos, W):
    return np.array([gamma_cap(rhos[c], W, c) for c in range(W.shape[0])])


def theory_bound(Xs, ys, W, b, rhos, pis):
    """Indicator bound: sum_c pi_c P(0 < gamma <= Gamma_c) under P^S_c."""
    G = all_gamma_caps(rhos, W)
    tot = 0.0
    per = np.zeros(len(pis))
    for c in range(W.shape[0]):
        m = margins(Xs[ys == c], ys[ys == c], W, b)
        per[c] = float(np.mean((m > 0) & (m <= G[c]))) if m.size else 0.0
        tot += pis[c] * per[c]
    return tot, G, per


def pos_quantile(m, q):
    """q-quantile of the positive part of a margin sample."""
    mp = m[m > 0]
    if mp.size == 0:
        return 0.0
    return float(np.quantile(mp, q))


def dkw_band(n, eta=0.05):
    return float(np.sqrt(np.log(2.0 / eta) / (2.0 * max(n, 1))))


# ---------------- correctors and baselines ----------------

def psd_sqrt(C, eps=1e-6, inv=False):
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, eps)
    p = -0.5 if inv else 0.5
    return (V * (w ** p)) @ V.T


def global_shift(Xt, dbar):
    return Xt - dbar


def oracle_shift(Xt, yt, deltas):
    return Xt - deltas[yt]


def bn_align(Xt, Xs):
    mt, st = Xt.mean(0), Xt.std(0) + 1e-8
    ms, ss = Xs.mean(0), Xs.std(0) + 1e-8
    return (Xt - mt) / st * ss + ms


def coral_align(Xt, Xs, shrink=1e-3):
    d = Xt.shape[1]
    Ct = np.cov(Xt, rowvar=False) + shrink * np.eye(d)
    Cs = np.cov(Xs, rowvar=False) + shrink * np.eye(d)
    A = psd_sqrt(Ct, inv=True) @ psd_sqrt(Cs)
    return (Xt - Xt.mean(0)) @ A + Xs.mean(0)


def _adapt_head(Xt, W, b, mode='tent', steps=150, lr=1e-3):
    import torch
    Wt = torch.tensor(W, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    X = torch.tensor(Xt, dtype=torch.float64)
    opt = torch.optim.Adam([Wt, bt], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        S = X @ Wt.T + bt
        logp = torch.log_softmax(S, 1)
        p = logp.exp()
        loss = -(p * logp).sum(1).mean()
        if mode == 'shot':
            pb = p.mean(0)
            loss = loss + (pb * torch.log(pb + 1e-12)).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        S = X @ Wt.T + bt
        logp = torch.log_softmax(S, 1)
        p = logp.exp()
        obj = float(-(p * logp).sum(1).mean())
        if mode == 'shot':
            pb = p.mean(0)
            obj = obj + float((pb * torch.log(pb + 1e-12)).sum())
    return Wt.detach().numpy(), bt.detach().numpy(), obj


def adapt_head(Xt, W, b, mode='tent', steps=150, lrs=(1e-4, 1e-3, 1e-2)):
    """Label-free model selection over lr using the method's own objective."""
    best = None
    for lr in lrs:
        Wn, bn, obj = _adapt_head(Xt, W, b, mode, steps, lr)
        if best is None or obj < best[2]:
            best = (Wn, bn, obj)
    return best[0], best[1]


def t3a(Xt, W, b, K, keep=20):
    S = scores(Xt, W, b)
    logp = S - S.max(1, keepdims=True)
    p = np.exp(logp); p = p / p.sum(1, keepdims=True)
    ent = -(p * np.log(p + 1e-12)).sum(1)
    pl = p.argmax(1)
    protos = np.zeros((K, Xt.shape[1]))
    for c in range(K):
        idx = np.where(pl == c)[0]
        if idx.size == 0:
            protos[c] = Xt.mean(0)
            continue
        idx = idx[np.argsort(ent[idx])[:max(1, min(keep, idx.size))]]
        protos[c] = Xt[idx].mean(0)
    Z = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-12)
    P = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
    return (Z @ P.T).argmax(1)


def bbse_pi(Xs_ho, ys_ho, Xt, W, b, K, pi_s):
    ps = predict(Xs_ho, W, b)
    C = np.zeros((K, K))
    for k in range(K):
        for c in range(K):
            C[k, c] = np.mean((ps == k) & (ys_ho == c))
    q = np.bincount(predict(Xt, W, b), minlength=K) / float(len(Xt))
    w = np.linalg.lstsq(C, q, rcond=None)[0]
    pit = np.clip(w * pi_s, 1e-8, None)
    return pit / pit.sum()


def class_means(X, y, K):
    M = np.zeros((K, X.shape[1]))
    for c in range(K):
        m = (y == c)
        M[c] = X[m].mean(0) if m.any() else X.mean(0)
    return M


def plugin_certificate(Xt, Xs, ys, W, b, K, dbar):
    """Label-free plug-in estimate of rho_c and Gamma_c, on Gamma_c's own scale.
    Pseudo-labels come from the globally corrected probe."""
    Xg = Xt - dbar
    pl = predict(Xg, W, b)
    MS = class_means(Xs, ys, K)
    MTg = class_means(Xg, pl, K)
    rho_hat = MTg - MS
    G_hat = np.array([gamma_cap(rho_hat[c], W, c) for c in range(K)])
    return G_hat, rho_hat


def mean_conf_score(Xt, W, b, K, dbar):
    """Null-model class ranking: negative mean max-softmax per pseudo-class."""
    Xg = Xt - dbar
    S = scores(Xg, W, b)
    Z = S - S.max(1, keepdims=True)
    P = np.exp(Z); P = P / P.sum(1, keepdims=True)
    pl = P.argmax(1); conf = P.max(1)
    out = np.zeros(K)
    for c in range(K):
        m = (pl == c)
        out[c] = -float(conf[m].mean()) if m.any() else 0.0
    return out


def selective_corrector(Xt, yt, Xs, ys, W, b, K, dbar, B, rng,
                        n_min=8, score=None, floor=0.0):
    """Certificate-driven selective per-class correction under a label budget B.
    Labels are revealed only for the B queried target points."""
    Xg = Xt - dbar
    pl = predict(Xg, W, b)
    MS = class_means(Xs, ys, K)
    if score is None:
        score, _ = plugin_certificate(Xt, Xs, ys, W, b, K, dbar)
    order = np.argsort(-np.asarray(score))
    cand = [int(c) for c in order if score[c] > floor]
    if len(cand) == 0:
        return Xg.copy(), []
    m = int(min(len(cand), max(1, B // n_min)))
    Ssel = cand[:m]
    per = max(1, B // m)
    Xc = Xg.copy()
    used = []
    for c in Ssel:
        idx = np.where(pl == c)[0]
        if idx.size == 0:
            continue
        q = rng.choice(idx, size=int(min(per, idx.size)), replace=False)
        keep = q[yt[q] == c]
        if keep.size < 2:
            continue
        dhat_c = Xt[keep].mean(0) - MS[c]
        mask = (pl == c)
        Xc[mask] = Xt[mask] - dhat_c
        used.append(c)
    return Xc, used


LAMS_CV = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def ridge_cv(XB, yB, K, folds=3, seed=0):
    """Ridge probe with the penalty chosen by folds-fold cross-validation on the
    labeled budget alone.  No target label outside the budget is used to pick the
    penalty, so the refitted probe is deployable."""
    XB = np.asarray(XB, dtype=np.float64)
    n, d = XB.shape
    Y = np.zeros((n, K)); Y[np.arange(n), yB] = 1.0
    Xa = np.c_[XB, np.ones(n)]
    I = np.eye(d + 1)
    idx = np.random.default_rng(seed).permutation(n)
    sc = {l: [0, 0] for l in LAMS_CV}
    for f in range(folds):
        te = idx[f::folds]
        mask = np.zeros(n, bool); mask[te] = True
        tr = ~mask
        if tr.sum() < 2 or mask.sum() < 1:
            continue
        A0 = Xa[tr].T @ Xa[tr]; B0 = Xa[tr].T @ Y[tr]
        for l in LAMS_CV:
            try:
                Wa = np.linalg.solve(A0 + l * I, B0)
            except Exception:
                continue
            pr = (XB[mask] @ Wa[:d] + Wa[d]).argmax(1)
            sc[l][0] += int((pr == yB[mask]).sum()); sc[l][1] += int(mask.sum())
    cand = [l for l in LAMS_CV if sc[l][1] > 0]
    best = max(cand, key=lambda l: sc[l][0] / sc[l][1]) if cand else 1e-2
    A = Xa.T @ Xa; Bm = Xa.T @ Y
    Wa = np.linalg.solve(A + best * I, Bm)
    return Wa[:d].T, Wa[d], float(best)


def recovery(a_m, a_g, a_o, eps=1e-9):
    den = a_o - a_g
    if den <= eps:
        return float('nan')
    return float((a_m - a_g) / den)


def ci95(v):
    v = np.asarray(v, dtype=float); v = v[np.isfinite(v)]
    if v.size < 2:
        return (float(v.mean()) if v.size else float('nan'), 0.0)
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(v.size))


# ---------------- controlled generator ----------------

def _sample_class(rng, mu, csqrt, n, df=None):
    z = rng.standard_normal((n, len(mu)))
    if df is not None:
        g = rng.chisquare(df, size=(n, 1)) / df
        z = z / np.sqrt(g)
    return mu + z @ csqrt


def make_source(rng, d=50, K=8, n=2000, sep=3.0, alpha=0.0, df=None):
    """alpha>0 gives a power-law (anisotropic) spectrum; df sets multivariate-t tails."""
    mus = rng.standard_normal((K, d))
    mus = mus / np.linalg.norm(mus, axis=1, keepdims=True) * sep
    lam = np.arange(1, d + 1) ** (-float(alpha))
    lam = lam / lam.mean()
    if alpha > 0:
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    else:
        Q = np.eye(d)
    csqrt = (Q * np.sqrt(lam)) @ Q.T
    X = []; y = []
    for c in range(K):
        X.append(_sample_class(rng, mus[c], csqrt, n, df))
        y.append(np.full(n, c))
    return np.vstack(X), np.concatenate(y), mus, csqrt


def disc_basis(W, tol=1e-8):
    Wc = W - W.mean(0)
    U, S, Vt = np.linalg.svd(Wc, full_matrices=False)
    r = int((S > tol * max(S[0], 1e-30)).sum())
    return Vt[:r]


def make_residuals(rng, W, pis, r_norm, theta):
    """Residuals with RMS norm r_norm and exact angle theta to the discriminative
    subspace D = span{w_j - w_k}. Satisfies sum_c pi_c rho_c = 0 by construction."""
    K, d = W.shape
    Vd = disc_basis(W)
    P = Vd.T @ Vd

    def fam(inside):
        G = rng.standard_normal((K, d))
        A = G @ P if inside else G - G @ P
        A = A - np.sum(pis[:, None] * A, axis=0, keepdims=True)
        s = np.sqrt(np.sum(pis * np.sum(A * A, axis=1)))
        return A / (s + 1e-12)

    A = fam(True)
    O = fam(False)
    return r_norm * (np.cos(theta) * A + np.sin(theta) * O)


def shift_target(rng, Xs, ys, K, deltas, pis_t=None, n_t=None):
    """Build a target sample by translating each class and optionally reweighting."""
    idx = []
    for c in range(K):
        pool = np.where(ys == c)[0]
        m = len(pool) if pis_t is None else int(round(pis_t[c] * n_t))
        m = max(2, min(m, len(pool)))
        idx.append(rng.choice(pool, size=m, replace=False))
    idx = np.concatenate(idx)
    yt = ys[idx]
    Xt = Xs[idx] + deltas[yt]
    return Xt, yt


def split_by_class(X, y, K, n_first, rng):
    a = []; bidx = []
    for c in range(K):
        idx = np.where(y == c)[0]
        idx = rng.permutation(idx)
        a.append(idx[:n_first]); bidx.append(idx[n_first:])
    a = np.concatenate(a); bidx = np.concatenate(bidx)
    return X[a], y[a], X[bidx], y[bidx]


def _adapt_affine(Xt, W, b, mode='tent', steps=150, lr=1e-3):
    """Faithful frozen-encoder analogue of TENT/SHOT: adapt a feature-wise affine
    transform (scale, shift) by entropy (plus diversity for SHOT); probe stays fixed."""
    import torch
    d = Xt.shape[1]
    g = torch.ones(d, dtype=torch.float64, requires_grad=True)
    bt = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    X = torch.tensor(Xt, dtype=torch.float64)
    Wt = torch.tensor(W, dtype=torch.float64)
    b0 = torch.tensor(b, dtype=torch.float64)
    opt = torch.optim.Adam([g, bt], lr=lr)

    def obj():
        S = (X * g + bt) @ Wt.T + b0
        logp = torch.log_softmax(S, 1)
        p = logp.exp()
        L = -(p * logp).sum(1).mean()
        if mode == 'shot':
            pb = p.mean(0)
            L = L + (pb * torch.log(pb + 1e-12)).sum()
        return L

    for _ in range(steps):
        opt.zero_grad(); L = obj(); L.backward(); opt.step()
    with torch.no_grad():
        v = float(obj())
    return (Xt * g.detach().numpy() + bt.detach().numpy()), v


def tta_baseline(Xt, W, b, mode='tent', steps=150, lrs=(1e-4, 1e-3, 1e-2)):
    """Best of head-adaptation and affine-adaptation, chosen by the method's own
    label-free objective. Returns (features, W, b) to score with."""
    best = None
    for lr in lrs:
        Xa, o = _adapt_affine(Xt, W, b, mode, steps, lr)
        if best is None or o < best[0]:
            best = (o, Xa, W, b)
    return best[1], best[2], best[3]


def tta_best_oracle(Xt, yt, W, b, mode='tent', steps=150, lrs=(1e-4, 1e-3, 1e-2)):
    """Upper bound for a TTA baseline: best accuracy over {head, affine} x lrs,
    selected using target labels. Deliberately generous to the baseline."""
    best = 0.0
    for lr in lrs:
        Wn, bn, _ = _adapt_head(Xt, W, b, mode, steps, lr)
        best = max(best, acc(Xt, yt, Wn, bn))
        Xa, _ = _adapt_affine(Xt, W, b, mode, steps, lr)
        best = max(best, acc(Xa, yt, W, b))
    return float(best)


# ---------------- label-free per-class rules and the residual conflict ----------------

def displacement_scores(W, D):
    """A[k, j] = <w_j, delta_k>: the score offset of class j at a point corrected by
    the displacement of class k.  Scores of x - delta_k are scores(x) - A[k]."""
    return D @ W.T


def per_hypothesis_predict(X, W, b, D):
    """argmax_j <w_j, x - delta_j> + b_j: every hypothesis is scored at its own
    corrected point.  For a linear head this is a class-dependent bias change."""
    S = scores(X, W, b)
    return (S - np.einsum('jd,jd->j', W, D)[None, :]).argmax(1)


def fixed_point_predict(X, W, b, D, r0, iters=100):
    """Iterate r <- h(x - delta_r) from r0.  A fixed point is a self-consistent label:
    the corrector it selects predicts that same label.  Returns the final prediction
    h(x - delta_r), the fraction of points at a fixed point, and the iterations used."""
    S = scores(X, W, b)
    A = displacement_scores(W, D)
    r = np.asarray(r0).copy()
    it = 0
    for it in range(1, iters + 1):
        rn = (S - A[r]).argmax(1)
        if np.array_equal(rn, r):
            break
        r = rn
    pred = (S - A[r]).argmax(1)
    return pred, float(np.mean(pred == r)), it


def em_per_class(X, W, b, MS, dbar, iters=30):
    """Label-free per-class translation by alternating assignment and estimation,
    in the manner of k-means: displacements are re-estimated from the current
    pseudo-classes and every point is re-assigned by r <- h(x - delta_hat_r)."""
    K = W.shape[0]
    S = scores(X, W, b)
    r = (S - (W @ dbar)[None, :]).argmax(1)
    D = np.tile(dbar, (K, 1))
    for _ in range(iters):
        for k in range(K):
            m = (r == k)
            D[k] = X[m].mean(0) - MS[k] if m.sum() >= 2 else dbar
        rn = (S - displacement_scores(W, D)[r]).argmax(1)
        if np.array_equal(rn, r):
            break
        r = rn
    return (S - displacement_scores(W, D)[r]).argmax(1)


def consistent_sets(X, W, b, D):
    """Boolean matrix F[i, k] = 1 when h(x_i - delta_k) = k, the set of labels that a
    self-consistent per-class corrector may return at x_i."""
    S = scores(X, W, b)
    A = displacement_scores(W, D)
    F = np.zeros(S.shape, dtype=bool)
    for k in range(W.shape[0]):
        F[:, k] = (S - A[k]).argmax(1) == k
    return F


def residual_conflict(W, R):
    """kappa[c, j] = <w_j - w_c, rho_c - rho_j>, symmetric in (c, j)."""
    P = R @ W.T
    d = np.diag(P)
    return P + P.T - d[:, None] - d[None, :]


def conflict_band_mass(X, y, W, b, D, R):
    """Fraction of target points that are correct under the oracle and lie in the
    conflict band 0 < gamma_k(z) <= kappa[y, k] for some k, with z = x - delta_y."""
    Z = X - D[y]
    S = scores(Z, W, b)
    idx = np.arange(len(y))
    G = S[idx, y][:, None] - S
    G[idx, y] = np.inf
    kap = residual_conflict(W, R)[y]
    kap[idx, y] = -np.inf
    ok = G.min(1) > 0
    return float(np.mean(ok & (G <= kap).any(1)))


def tta_standard(Xt, W, b, mode='tent', optimizer='adam', lr=None, batch=None,
                 epochs=1, seed=0, filt=None):
    """TENT and SHOT in a standard episodic configuration on frozen features: a
    feature-wise affine transform (the analogue of normalization-layer parameters) is
    adapted over one pass of shuffled mini-batches, with no selection of
    hyperparameters by the method's own objective.  optimizer='adam' uses learning
    rate 1e-3 and batches of 128; optimizer='sgd' uses the published TENT setting of
    SGD with momentum 0.9, learning rate 2.5e-4 and batches of 64.  filt, when given,
    keeps in each batch only the points whose prediction entropy is below
    filt * log K, the reliable-sample filter of SAR."""
    if lr is None:
        lr = 1e-3 if optimizer == 'adam' else 2.5e-4
    if batch is None:
        batch = 128 if optimizer == 'adam' else 64
    import torch
    torch.manual_seed(seed)
    d = Xt.shape[1]
    g = torch.ones(d, dtype=torch.float64, requires_grad=True)
    sh = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    X = torch.tensor(Xt, dtype=torch.float64)
    Wt = torch.tensor(W, dtype=torch.float64)
    b0 = torch.tensor(b, dtype=torch.float64)
    if optimizer == 'adam':
        opt = torch.optim.Adam([g, sh], lr=lr)
    else:
        opt = torch.optim.SGD([g, sh], lr=lr, momentum=0.9)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for s in range(0, len(Xt), batch):
            idx = torch.tensor(rng.permutation(len(Xt))[:batch])
            Sb = (X[idx] * g + sh) @ Wt.T + b0
            logp = torch.log_softmax(Sb, 1)
            p = logp.exp()
            ent = -(p * logp).sum(1)
            if filt is not None:
                keep = ent < filt * np.log(W.shape[0])
                if int(keep.sum()) == 0:
                    continue
                ent = ent[keep]; p = p[keep]
            L = ent.mean()
            if mode == 'shot':
                pb = p.mean(0)
                L = L + (pb * torch.log(pb + 1e-12)).sum()
            opt.zero_grad(); L.backward(); opt.step()
    with torch.no_grad():
        Xa = (X * g + sh).numpy()
    return Xa


TTA_GRID = [(opt, lr, filt) for opt in ('adam', 'sgd')
            for lr in (1e-4, 3e-4, 1e-3, 3e-3) for filt in (None, 0.4)]


def tta_grid(Xt, yt, W, b, mode):
    """Accuracy of TENT or SHOT at every configuration of TTA_GRID.  The scores are
    stored per configuration so that a configuration can be selected on held-out
    shifts, never on the shift it is evaluated on."""
    out = {}
    for opt, lr, filt in TTA_GRID:
        Xa = tta_standard(Xt, W, b, mode=mode, optimizer=opt, lr=lr, filt=filt)
        out[(opt, lr, filt)] = acc(Xa, yt, W, b)
    return out


def select_held_out(rows, key, group):
    """Leave-one-group-out selection: for each row, the configuration with the best
    mean accuracy over the rows of the other groups, scored on the row itself."""
    res = []
    for r in rows:
        oth = [q for q in rows if q[group] != r[group]]
        best = max(TTA_GRID, key=lambda c: np.mean([q[key][c] for q in oth]))
        res.append(r[key][best])
    return np.array(res)
