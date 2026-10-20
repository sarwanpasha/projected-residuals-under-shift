"""Additional routines: spectral dimensions, shrinkage refits, the LAME baseline,
paired inference, confidence bands, and the decomposition of an affine corrector.

The probe convention follows prlb.core: a probe is a pair (W, b) with W of shape
(K, d) and b of shape (K,), and its scores are X @ W.T + b.
"""
import numpy as np


# --------------------------------------------------------------------- spectra

def spectrum(X):
    """Descending eigenvalues of the centred second moment of X."""
    Xc = X - X.mean(0, keepdims=True)
    n, d = Xc.shape
    if d <= n:
        lam = np.linalg.eigvalsh(Xc.T @ Xc / n)
    else:
        lam = np.linalg.eigvalsh(Xc @ Xc.T / n)
    return np.clip(lam[::-1].astype(np.float64), 0.0, None)


def pr_dim(lam):
    """Participation ratio (sum l)^2 / sum l^2.  Equals d exactly for a flat spectrum
    and collapses toward 1 for a spectrum dominated by one direction."""
    s1 = float(lam.sum()); s2 = float((lam ** 2).sum())
    return s1 * s1 / max(s2, 1e-300)


def entropy_dim(lam):
    """exp of the Shannon entropy of the normalised spectrum, the effective rank."""
    p = lam / max(float(lam.sum()), 1e-300)
    p = p[p > 1e-300]
    return float(np.exp(-(p * np.log(p)).sum()))


def ridge_dim(lam, lam_star):
    """Ridge effective dimension sum_i l_i / (l_i + lam_star).  This is the quantity
    that replaces the ambient dimension in the corrected Route A requirement."""
    return float((lam / (lam + max(float(lam_star), 1e-300))).sum())


def eff_class_count(pi):
    """s(pi) = (sum_c sqrt(pi_c (1 - pi_c)))^2, the effective number of cells that the
    multinomial l1 error pays for.  Exactly K - 1 at the uniform marginal, and much
    smaller for a peaked marginal.  This is the quantity that replaces the raw class
    count in the corrected Route B requirement."""
    p = np.asarray(pi, dtype=np.float64)
    return float(np.sum(np.sqrt(np.clip(p * (1.0 - p), 0.0, None))) ** 2)


def pca_truncate(Xs, Xt_list, d):
    """Project every array onto the top d principal directions of the source array.
    Returns the truncated source, the truncated targets, and the source spectrum in
    the truncated basis."""
    mu = Xs.mean(0, keepdims=True)
    Xc = Xs - mu
    C = Xc.T @ Xc / Xc.shape[0]
    w, V = np.linalg.eigh(C)
    V = V[:, ::-1][:, :d]
    return (Xc @ V, [(A - mu) @ V for A in Xt_list], np.clip(w[::-1][:d], 0.0, None))


# ------------------------------------------------------------ Route A refits

def ridge_to_source(Xb, yb, K, W_S, b_S, lam):
    """One versus rest least squares on B target labels, shrunk toward the source probe:

        argmin_T  ||Z T - Y||_F^2 / B  +  lam * ||T - T_S||_F^2 ,   Z = [Xb, 1].

    lam to 0 recovers the unregularized refit, and lam to infinity recovers the
    source probe unchanged.  In the intermediate regime the label requirement is
    governed by the ridge effective dimension rather than the ambient dimension."""
    B, d = Xb.shape
    Z = np.concatenate([Xb, np.ones((B, 1))], 1)
    Y = np.zeros((B, K)); Y[np.arange(B), yb] = 1.0
    T_S = np.concatenate([W_S.T, b_S.reshape(1, -1)], 0)
    A = Z.T @ Z / B + lam * np.eye(d + 1)
    T = np.linalg.solve(A, Z.T @ Y / B + lam * T_S)
    return T[:d].T, T[d]


def ridge_to_source_cv(Xb, yb, K, W_S, b_S, lams, rng, folds=5):
    """Select the shrinkage level using the B target labels only, by cross validated
    0-1 error, breaking ties toward more shrinkage.  No target test data, no target
    label beyond the purchased budget, and no oracle is consulted, so the selected
    probe is deployable."""
    B = Xb.shape[0]
    nf = int(min(folds, max(2, B // 2)))
    perm = rng.permutation(B)
    fold = np.zeros(B, dtype=np.int64); fold[perm] = np.arange(B) % nf
    err = np.zeros(len(lams))
    for f in range(nf):
        tr = fold != f; te = ~tr
        if tr.sum() < 2 or te.sum() < 1:
            continue
        for j, lm in enumerate(lams):
            W, b = ridge_to_source(Xb[tr], yb[tr], K, W_S, b_S, lm)
            err[j] += float((np.argmax(Xb[te] @ W.T + b, 1) != yb[te]).sum())
    best = float(err.min())
    j = int(np.max(np.where(err <= best + 1e-12)[0]))
    lm = lams[j]
    W, b = ridge_to_source(Xb, yb, K, W_S, b_S, lm)
    return W, b, float(lm)


def proto_refit(Xb, yb, K, W_S, b_S, tau=1.0, mu_S=None):
    """Few shot prototype head.  Class means of the purchased labels are shrunk toward
    the source class means with weight tau, and any class that received no label keeps
    its source prototype, so the head is defined at every budget."""
    d = Xb.shape[1]
    if mu_S is None:
        mu_S = W_S.copy()
    M = np.zeros((K, d))
    for c in range(K):
        m = yb == c
        n_c = int(m.sum())
        if n_c == 0:
            M[c] = mu_S[c]
        else:
            M[c] = (n_c * Xb[m].mean(0) + tau * mu_S[c]) / (n_c + tau)
    return M, -0.5 * (M ** 2).sum(1)


# ------------------------------------------------------------ LAME baseline

def knn_index(X, k):
    """Indices of the k nearest neighbours by cosine similarity, self excluded."""
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    n = Xn.shape[0]
    idx = np.zeros((n, k), dtype=np.int64)
    ch = 1024
    for s in range(0, n, ch):
        S = Xn[s:s + ch] @ Xn.T
        for r in range(S.shape[0]):
            S[r, s + r] = -np.inf
        idx[s:s + ch] = np.argpartition(-S, k, axis=1)[:, :k]
    return idx


def lame(Xt, W, b, K, knn=5, alpha=1.0, iters=100, tol=1e-6, idx=None):
    """LAME style label free adaptation on frozen embeddings.  The assignment
    distributions z_i maximise the source probe log likelihood plus a Laplacian
    agreement term over a kNN graph on the target embeddings, and the concave convex
    procedure gives the fixed point z_i proportional to p_i exp(alpha sum_j w_ij z_j).
    This is the natural frozen encoder competitor that is aware of the target label
    structure without purchasing any label."""
    S = Xt @ W.T + b
    S = S - S.max(1, keepdims=True)
    P = np.exp(S); P /= np.maximum(P.sum(1, keepdims=True), 1e-300)
    if idx is None:
        idx = knn_index(Xt, knn)
    n = Xt.shape[0]
    rows = np.repeat(np.arange(n), idx.shape[1]); cols = idx.ravel()
    Z = P.copy()
    for _ in range(iters):
        M = np.zeros_like(Z)
        np.add.at(M, rows, Z[cols])
        np.add.at(M, cols, Z[rows])
        Zn = P * np.exp(np.clip(alpha * M / max(idx.shape[1], 1), -30, 30))
        Zn /= np.maximum(Zn.sum(1, keepdims=True), 1e-300)
        if np.abs(Zn - Z).max() < tol:
            Z = Zn; break
        Z = Zn
    return Z.argmax(1)


# ------------------------------------------------------------ paired inference

def paired_perm(a, b, n=20000, rng=None):
    """Two sided paired sign flip permutation test on the mean of a minus b.  Returns
    the mean difference, the p value, and a percentile bootstrap interval."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    m = int(d.shape[0]); obs = float(d.mean())
    if rng is None:
        rng = np.random.default_rng(0)
    Sg = rng.integers(0, 2, size=(n, m)).astype(np.float64) * 2.0 - 1.0
    null = (Sg * d[None, :]).mean(1)
    p = float((np.sum(np.abs(null) >= abs(obs) - 1e-15) + 1) / (n + 1))
    bs = d[rng.integers(0, m, size=(n, m))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return obs, p, float(lo), float(hi)


def boot_reg(Xp, y, groups=None, n=4000, rng=None):
    """Ordinary least squares of y on the design [Xp, 1] with a cluster bootstrap over
    the supplied grouping.  Returns the coefficients, a 95 percent percentile interval
    for each, and R2."""
    Xp = np.atleast_2d(np.asarray(Xp, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64)
    if Xp.shape[0] != y.shape[0]:
        Xp = Xp.T
    A = np.concatenate([Xp, np.ones((Xp.shape[0], 1))], 1)
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    r = y - A @ beta
    r2 = 1.0 - float(r @ r) / max(float(((y - y.mean()) ** 2).sum()), 1e-300)
    if rng is None:
        rng = np.random.default_rng(0)
    if groups is None:
        groups = np.arange(y.shape[0])
    groups = np.asarray(groups)
    uq = np.unique(groups)
    idx_by_g = [np.where(groups == g)[0] for g in uq]
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(uq), size=len(uq))
        sel = np.concatenate([idx_by_g[j] for j in pick])
        try:
            draws.append(np.linalg.lstsq(A[sel], y[sel], rcond=None)[0])
        except np.linalg.LinAlgError:
            continue
    D = np.array(draws)
    return beta, np.percentile(D, 2.5, axis=0), np.percentile(D, 97.5, axis=0), r2


# -------------------------------------------------------- tight certificate bands

def cp_upper(k, n, delta=0.05):
    """One sided Clopper Pearson upper confidence limit for a binomial proportion.
    For a 0-1 loss this is exact and is the right band for the certificate, whereas a
    distribution free Kolmogorov band ignores the smallness of the loss."""
    from scipy.stats import beta as _beta
    if n <= 0 or k >= n:
        return 1.0
    return float(_beta.ppf(1.0 - delta, k + 1, n - k))


def eb_band(v, delta=0.05):
    """Empirical Bernstein deviation for the mean of [0,1] valued observations.  The
    band shrinks with the observed variance, which is why it is operational at sample
    sizes where a variance free band is not."""
    v = np.asarray(v, dtype=np.float64); n = int(v.shape[0])
    if n < 2:
        return 1.0
    s2 = float(v.var(ddof=1)); L = float(np.log(2.0 / delta))
    return float(np.sqrt(2.0 * s2 * L / n) + 7.0 * L / (3.0 * (n - 1)))


def kt_pi(counts, K):
    """Krichevsky Trofimov add one half marginal estimate.  Strictly positive for every
    class at every budget, so Route B stays defined when a class receives no label and
    the per class minimum count condition is not needed."""
    c = np.asarray(counts, dtype=np.float64)
    return (c + 0.5) / (c.sum() + 0.5 * K)


# ------------------------------------------------------------------ utilities

def budget_threshold(Bs, vals, target):
    """Smallest budget at which the running maximum of a curve reaches the target,
    with linear interpolation between grid points.  NaN if it never reaches it."""
    Bs = np.asarray(Bs, dtype=np.float64); vals = np.asarray(vals, dtype=np.float64)
    run = np.maximum.accumulate(vals)
    hit = np.where(run >= target)[0]
    if hit.size == 0:
        return np.nan
    i = int(hit[0])
    if i == 0:
        return float(Bs[0])
    v0, v1 = run[i - 1], run[i]
    if v1 <= v0:
        return float(Bs[i])
    return float(Bs[i - 1] + (target - v0) / (v1 - v0) * (Bs[i] - Bs[i - 1]))


def acc_from(X, y, W, b):
    return float((np.argmax(X @ W.T + b, 1) == y).mean())


# ------------------------------------- decomposing an affine label free corrector

def scale_align(Xt, Xs):
    """Match per coordinate scale without moving the mean.  This is the complement of
    the common mean correction inside a diagonal affine corrector, and isolating it lets
    the location component and the scale component be credited separately."""
    mt = Xt.mean(0, keepdims=True)
    st = Xt.std(0, keepdims=True); ss = Xs.std(0, keepdims=True)
    return (Xt - mt) * (ss / np.maximum(st, 1e-8)) + mt


def mean_align(Xt, Xs):
    """Common mean correction written in the same form as the scale variant, so the two
    components of a diagonal affine corrector are measured on the same footing."""
    return Xt - Xt.mean(0, keepdims=True) + Xs.mean(0, keepdims=True)


def noisy_assign(y, alpha, K, rng):
    """Assignments that agree with the truth on a fraction alpha of the points and are
    uniform over the other K - 1 labels elsewhere.  Sweeping alpha interpolates between
    the per class oracle and a blind assignment and locates the point at which a
    deployable per class corrector loses its advantage."""
    y = np.asarray(y)
    out = y.copy()
    flip = rng.random(y.shape[0]) > alpha
    if flip.any():
        shift = rng.integers(1, K, size=int(flip.sum()))
        out[flip] = (y[flip] + shift) % K
    return out


def tv_lower_bound(A, B, rng, frac=0.5, lam=1e-2):
    """Lower bound on the total variation distance between two samples, obtained by
    fitting a linear discriminator on one half and reading its balanced accuracy on the
    held out half:  TV >= 2 (balanced accuracy - 1/2)."""
    nA, nB = A.shape[0], B.shape[0]
    if nA < 8 or nB < 8:
        return 0.0
    ia = rng.permutation(nA); ib = rng.permutation(nB)
    ca = int(frac * nA); cb = int(frac * nB)
    Xtr = np.concatenate([A[ia[:ca]], B[ib[:cb]]], 0)
    ytr = np.concatenate([np.zeros(ca, dtype=np.int64), np.ones(cb, dtype=np.int64)])
    Xte_a = A[ia[ca:]]; Xte_b = B[ib[cb:]]
    d = Xtr.shape[1]
    Z = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], 1)
    Y = np.zeros((Xtr.shape[0], 2)); Y[np.arange(Xtr.shape[0]), ytr] = 1.0
    T = np.linalg.solve(Z.T @ Z + lam * np.eye(d + 1), Z.T @ Y)
    W, b = T[:d].T, T[d]
    pa = float((np.argmax(Xte_a @ W.T + b, 1) == 0).mean())
    pb = float((np.argmax(Xte_b @ W.T + b, 1) == 1).mean())
    return float(max(0.0, 2.0 * (0.5 * (pa + pb) - 0.5)))


# ------------------------------------------------------------ few-shot baselines

def _folds(B, rng, folds=5):
    nf = int(min(folds, max(2, B // 2)))
    fold = np.zeros(B, dtype=np.int64); fold[rng.permutation(B)] = np.arange(B) % nf
    return fold, nf


def _ovr(Xb, yb, K, lam):
    n, d = Xb.shape
    Y = np.zeros((n, K)); Y[np.arange(n), yb] = 1.0
    Xa = np.c_[Xb, np.ones(n)]
    Wa = np.linalg.solve(Xa.T @ Xa + lam * np.eye(d + 1), Xa.T @ Y)
    return Wa[:d].T, Wa[d]


def wise_cv(Xb, yb, K, W_S, b_S, rng, thetas=np.linspace(0.0, 1.0, 11), lam=1e-2):
    """Weight-space interpolation in the manner of WiSE-FT: theta * (refit from
    scratch) + (1 - theta) * (source probe), with theta chosen by cross-validated 0-1
    error on the budget alone, ties broken toward the source probe."""
    B = Xb.shape[0]
    fold, nf = _folds(B, rng)
    err = np.zeros(len(thetas))
    for f in range(nf):
        tr = fold != f; te = ~tr
        if tr.sum() < 2 or te.sum() < 1:
            continue
        Wr, br = _ovr(Xb[tr], yb[tr], K, lam)
        for j, t in enumerate(thetas):
            S = Xb[te] @ (t * Wr + (1 - t) * W_S).T + t * br + (1 - t) * b_S
            err[j] += float((S.argmax(1) != yb[te]).sum())
    j = int(np.min(np.where(err <= err.min() + 1e-12)[0]))
    t = float(thetas[j])
    Wr, br = _ovr(Xb, yb, K, lam)
    return t * Wr + (1 - t) * W_S, t * br + (1 - t) * b_S, t


def pro2_basis(Xs, ys, K, rmax, lam=1e-2):
    """Ordered orthonormal source-predictive directions in the manner of Pro2: a ridge
    probe is fitted on the source, its discriminative directions are kept in order of
    singular value, the features are projected off the span found so far, and the
    step repeats until rmax directions are collected."""
    Xc = Xs - Xs.mean(0)
    d = Xs.shape[1]
    Q = np.zeros((d, 0))
    while Q.shape[1] < min(rmax, d):
        Xr = Xc - (Xc @ Q) @ Q.T
        W, _ = _ovr(Xr, ys, K, lam)
        Wd = W - W.mean(0)
        Wd = Wd - (Wd @ Q) @ Q.T
        U, s, Vt = np.linalg.svd(Wd, full_matrices=False)
        keep = s > 1e-8 * max(s.max(), 1e-30)
        if not keep.any():
            break
        Q = np.concatenate([Q, Vt[keep].T], 1)
        Q, _ = np.linalg.qr(Q)
    return Q[:, :min(rmax, Q.shape[1])]


def pro2_cv(Xb, yb, K, Q, rng, ranks, lams=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Probe on the first r source-predictive directions, with r and the ridge penalty
    chosen by cross-validated 0-1 error on the budget alone.  Returns a probe on the
    full features."""
    B = Xb.shape[0]
    fold, nf = _folds(B, rng)
    cand = [(r, l) for r in ranks if r <= Q.shape[1] for l in lams]
    err = np.zeros(len(cand))
    for f in range(nf):
        tr = fold != f; te = ~tr
        if tr.sum() < 2 or te.sum() < 1:
            continue
        for j, (r, l) in enumerate(cand):
            P = Q[:, :r]
            W, b = _ovr(Xb[tr] @ P, yb[tr], K, l)
            err[j] += float(((Xb[te] @ P) @ W.T + b).argmax(1).__ne__(yb[te]).sum())
    j = int(np.argmin(err))
    r, l = cand[j]
    P = Q[:, :r]
    W, b = _ovr(Xb @ P, yb, K, l)
    return W @ P.T, b, r
