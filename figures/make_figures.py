"""Draw Figures 1 to 4 from the saved experiment outputs.

Inputs:
    results/synthetic/rotation.pkl                 Figure 1
    results/cifar10c/shift_pairs_{clip,dinov2}.pkl Figure 2
    results/domainnet/dnet_{clip,dino}.pkl         Figure 3
    results/cifar10c/budget_{clip,dinov2}.pkl      Figure 4
Outputs: results/figures/f1_rotation.pdf, f2_realbound.pdf, f3_domainnet.pdf,
f4_allocation.pdf.  In each two-panel figure the left panel is CLIP and the right panel
is DINOv2.
"""
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from prlb.paths import SYN_OUT, CIFAR_OUT, DNET_OUT, FIG_OUT, ensure

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'xtick.labelsize': 11,
                     'ytick.labelsize': 11, 'legend.fontsize': 9.5, 'axes.grid': True,
                     'grid.alpha': 0.3, 'figure.dpi': 200, 'savefig.bbox': 'tight',
                     'pdf.fonttype': 42, 'font.family': 'STIXGeneral',
                     'mathtext.fontset': 'stix', 'lines.linewidth': 1.6,
                     'axes.linewidth': 0.8, 'legend.frameon': False})
C = {'gap': '#1f3b73', 'ours': '#c0392b', 'cheb': '#2e8b57', 'subg': '#8e44ad'}


def load(p):
    return pickle.load(open(p, 'rb'))


def ci(v):
    v = np.asarray(v, float)
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def fig_rotation():
    rows = load(SYN_OUT / 'rotation.pkl')
    th = sorted(set(r['theta'] for r in rows))
    g = np.array([ci([r['gap'] for r in rows if r['theta'] == t]) for t in th])
    b = np.array([ci([r['bound'] for r in rows if r['theta'] == t]) for t in th])
    ch = [np.mean([r['cheb'] for r in rows if r['theta'] == t]) for t in th]
    sg = [np.mean([r['subg'] for r in rows if r['theta'] == t]) for t in th]
    f, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.plot(th, ch, '^-', ms=4, color=C['cheb'], label='Chebyshev')
    ax.plot(th, sg, 'v-', ms=4, color=C['subg'], label='sub-Gaussian')
    ax.errorbar(th, b[:, 0], yerr=b[:, 1], fmt='s--', ms=4, capsize=2, color=C['ours'], label='ours')
    ax.errorbar(th, g[:, 0], yerr=g[:, 1], fmt='o-', ms=4, capsize=2, color=C['gap'], label='measured')
    ax.set_yscale('symlog', linthresh=1e-3)
    ax.set_xlabel(r'$\theta$ (degrees)'); ax.set_ylabel('accuracy gap')
    ax.set_xlim(-3, 93); ax.set_xticks([0, 30, 60, 90]); ax.legend(loc='lower left')
    f.savefig(FIG_OUT / 'f1_rotation.pdf'); plt.close(f)


def fig_realbound():
    law = [('bound_t', 'o', '#1f3b73', 'oracle-corrected'),
           ('bound_h', 's', '#2e8b57', 'pseudo-labeled'),
           ('bound', '^', '#c0392b', 'source')]
    f, axs = plt.subplots(1, 2, figsize=(7.16, 2.9))
    for ax, k in zip(axs, ['clip', 'dinov2']):
        rs = load(CIFAR_OUT / ('shift_pairs_%s.pkl' % k))
        gp = np.array([r['gap'] for r in rs])
        hi = max(gp.max(), max(np.array([r[q] for r in rs]).max() for q, _, _, _ in law)) * 1.06
        ax.plot([0, hi], [0, hi], 'k--', lw=1, zorder=1, label='equality')
        for q, mk, col, lab in law:
            ax.scatter(gp, [r[q] for r in rs], s=20, marker=mk, c=col, zorder=3,
                       alpha=0.8, linewidths=0, label=lab)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_xlabel('measured gap'); ax.set_ylabel('bound')
    axs[0].legend(loc='lower right', handletextpad=0.4)
    f.tight_layout(); f.savefig(FIG_OUT / 'f2_realbound.pdf'); plt.close(f)


def fig_domainnet():
    D = {e: load(DNET_OUT / ('dnet_%s.pkl' % e)) for e in ['clip', 'dino']}
    KS = sorted(set(r['K'] for rs in D.values() for r in rs))
    f, ax = plt.subplots(1, 3, figsize=(7.16, 2.5))
    rs0 = D['clip']
    md = [np.median([r['drift'] for r in rs0 if r['K'] == K]) for K in KS]
    lo = [np.percentile([r['drift'] for r in rs0 if r['K'] == K], 10) for K in KS]
    hi = [np.percentile([r['drift'] for r in rs0 if r['K'] == K], 90) for K in KS]
    ax[0].errorbar(KS, md, yerr=[np.array(md) - lo, np.array(hi) - md], fmt='o-', ms=4,
                   capsize=3, color='#1f3b73')
    ax[0].set_xscale('log'); ax[0].set_xticks(KS)
    ax[0].set_xticklabels([str(k) for k in KS]); ax[0].set_xlabel(r'$K$')
    ax[0].set_ylabel(r'$\|\pi^T-\pi^S\|_1$')
    for e, rs in D.items():
        gp = np.array([r['gap'] for r in rs]); m = gp > 0
        for q, mk, col in [('b_orc', 'o', '#1f3b73'), ('b_ps', 's', '#2e8b57'), ('b_src', '^', '#c0392b')]:
            ax[1].plot(gp[m], np.array([r[q] for r in rs])[m], mk, ms=2.6, alpha=0.55, color=col)
    lim = [1e-4, 1.0]
    ax[1].plot(lim, lim, 'k--', lw=1)
    ax[1].set_xscale('log'); ax[1].set_yscale('log')
    ax[1].set_xlabel('measured gap'); ax[1].set_ylabel('bound')
    for e, mk, col in [('clip', 'o-', '#1f3b73'), ('dino', 's--', '#c0392b')]:
        rs = D[e]
        ax[2].plot(KS, [np.median([r['npos'] / float(K) for r in rs if r['K'] == K]) for K in KS],
                   mk, ms=4, color=col)
        ax[2].plot(KS, [np.median([r['nposhat'] / float(K) for r in rs if r['K'] == K]) for K in KS],
                   mk[0] + ':', ms=4, lw=1.2, alpha=0.75, color=col)
    ax[2].set_xscale('log'); ax[2].set_xticks(KS)
    ax[2].set_xticklabels([str(k) for k in KS]); ax[2].set_xlabel(r'$K$')
    ax[2].set_ylabel(r'fraction $\Gamma_c>0$'); ax[2].set_ylim(-0.04, 1.08)
    f.tight_layout(); f.savefig(FIG_OUT / 'f3_domainnet.pdf'); plt.close(f)


def fig_allocation():
    bud = [50, 100, 200, 400, 800]
    f, axs = plt.subplots(1, 2, figsize=(7.16, 2.9))
    for ax, k in zip(axs, ['clip', 'dinov2']):
        rs = [r for r in load(CIFAR_OUT / ('budget_%s.pkl' % k)) if r['alpha'] == 1.0]
        for tag, mk, col, lab in [('refit', 's-', '#c0392b', 'refit the probe'),
                                  ('cnt', 'o-', '#1f3b73', 'count the marginal'),
                                  ('cnd', '^-', '#2e8b57', 'confusion conditional')]:
            y = np.array([ci([r['a_%s%d' % (tag, B)] for r in rs]) for B in bud])
            ax.errorbar(bud, y[:, 0], yerr=y[:, 1], fmt=mk, ms=4, capsize=2, color=col, label=lab)
        for tag, st, col, lab in [('orcb', '--', 'k', 'oracle marginal'),
                                  ('naive', '-.', '0.45', 'common-mean'),
                                  ('none', ':', '0.45', 'no correction')]:
            ax.axhline(ci([r['a_' + tag] for r in rs])[0], ls=st, lw=1.2, color=col, label=lab)
        ax.set_xscale('log'); ax.set_xticks(bud)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(r'$B$'); ax.set_ylabel('accuracy')
    h, l = axs[0].get_legend_handles_labels()
    f.legend(h, l, loc='lower center', ncol=3, handletextpad=0.4, columnspacing=1.4,
             bbox_to_anchor=(0.5, -0.13))
    f.tight_layout(); f.savefig(FIG_OUT / 'f4_allocation.pdf'); plt.close(f)


def main():
    ensure(FIG_OUT)
    for name, fn in [('f1', fig_rotation), ('f2', fig_realbound),
                     ('f3', fig_domainnet), ('f4', fig_allocation)]:
        try:
            fn()
            print(name, 'written', flush=True)
        except FileNotFoundError as e:
            print(name, 'skipped, missing input:', e.filename, flush=True)


if __name__ == '__main__':
    main()
