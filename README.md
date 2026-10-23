# Projected residuals and label budgets for linear probes under shift

The repository computes, from frozen foundation-model embeddings and a linear probe,

* the **projected residual** Γ_c = max_{j≠c} ⟨ρ_c, w_j − w_c⟩ of each class, where ρ_c is the
  part of the class displacement that a common-mean correction cannot remove;
* the **gap bound**: the accuracy lost by the common-mean correction relative
  to the per-class oracle is at most Σ_c π_c P(0 < γ ≤ Γ_c), evaluated with source,
  oracle-corrected or pseudo-labeled margins;
* the **label-free certificate** Γ̂_c, its screening rule and the budget-safe selective
  corrector;
* the three uses of a target **label budget** B: refitting the probe (Route A), counting
  the target label marginal (Route B, post-stratified counting) and estimating the
  confusion conditional (Route C), with their label requirements and crossover budget;
* label-free baselines (BN alignment, CORAL, T3A, TENT, SHOT, LAME) with paired
  sign-flip permutation tests;
* the refit anchored to the source probe, weight-space interpolation and a projected
  probe as uses of the budget, and supervised target references.

## Repository layout

```
prlb/                       library (installed as the package "prlb")
  core.py                   ridge probe, margins, Γ_c, gap bound, correctors,
                            baselines, plug-in certificate, selective corrector,
                            controlled data generator
  extra.py                  spectral dimensions, shrinkage refits, LAME, paired
                            permutation test, cluster bootstrap, confidence bands,
                            scale/location decomposition of an affine corrector
  paths.py                  data and result locations (overridable by environment)
data_prep/
  download_cifar.py         CIFAR-10 and CIFAR-10-C
  encode_cifar.py           CLIP ViT-B/32 and DINOv2 ViT-B/14 embeddings of CIFAR-10(-C)
  embed_domainnet.py        CLIP and DINOv2 embeddings of one DomainNet domain
  text_embed.py             CLIP text prototypes for the zero-shot head
experiments/
  synthetic/                controlled experiments (no download needed)
  cifar10c/                 CIFAR-10 → CIFAR-10-C, 19 corruptions × 5 severities
  domainnet/                DomainNet, 6 domains, 30 ordered pairs, K ∈ {10,25,50,100,345}
  zeroshot/                 prototype shifting against feature translation, zero-shot CLIP head
figures/make_figures.py     summary figures
scripts/run_all.sh          the full pipeline in order
scripts/run.sbatch          generic Slurm wrapper for any single step
```

## Installation

Python 3.9 or newer. From the repository root:

```bash
pip install -r requirements.txt
pip install -e .
```

`numpy`, `scipy` and `matplotlib` are enough for the synthetic experiments, for every
script that only reads saved results, and for the figures. `torch`, `transformers`,
`huggingface_hub`, `pyarrow` and `pillow` are needed for the embedding scripts and for
the TENT, SHOT and temperature-scaling arms. A GPU is recommended for embedding; every
analysis script runs on CPU.

By default data go to `./data` and outputs to `./results`. To place them elsewhere:

```bash
export PRLB_DATA=/path/to/data
export PRLB_RESULTS=/path/to/results
```

All scripts are deterministic: generator seeds are derived from the cell index, and the
run scripts set `PYTHONHASHSEED=0`.

## Data

**CIFAR-10 and CIFAR-10-C.**

```bash
python data_prep/download_cifar.py
python data_prep/encode_cifar.py clip
python data_prep/encode_cifar.py dinov2
```

CIFAR-10 is read from the Hugging Face dataset `uoft-cs/cifar10`, and CIFAR-10-C
(Hendrycks and Dietterich, 2019) from Zenodo record 2535967. Embeddings are written
to `data/cifar10_emb/` as `<enc>_train.npy`, `<enc>_clean.npy` and
`<enc>_<corruption>.npz` with keys `s1`–`s5`.

**DomainNet.**

```bash
for d in clipart infograph painting quickdraw real sketch; do
  python data_prep/embed_domainnet.py $d
done
```

The official train/test split lists are fetched automatically. Each archive is read
directly from the DomainNet server with HTTP range requests, so the zip files never
need to be stored; a local copy placed at `data/domainnet/zips/<domain>.zip` is used
instead when present. For clipart and painting the cleaned archives
(`groundtruth/clipart.zip`, `groundtruth/painting.zip`) are used, which match the split
lists. Both encoders are applied in a single pass, giving `<domain>_clip.npy`,
`<domain>_dino.npy`, `<domain>_y.npy`, `<domain>_ok.npy` and `classes.txt` in
`data/domainnet/emb/`, 586,575 images in total.

Encoder identifiers are `clip` and `dinov2` for CIFAR-10-C and `clip` and `dino` for
DomainNet.

## Reproducing the results

`bash scripts/run_all.sh` runs everything in order and writes one log per step to
`results/logs/`. Each step can also be run on its own, or submitted to a Slurm
cluster through the wrapper, for example

```bash
mkdir -p results/logs
sbatch --account="$ACCOUNT" --partition="$PARTITION" scripts/run.sbatch \
       python experiments/cifar10c/shift_pairs.py clip
```

### Controlled experiments (`experiments/synthetic`)

| Script | Result | Output |
|---|---|---|
| `rotation.py` | Rotation sweep, 19 angles × 30 seeds = 570 cells; the gap bound against Chebyshev and sub-Gaussian bounds | `synthetic/rotation.pkl` |
| `high_dim.py` | 720 cells over d ∈ {50, 256, 512, 1024}, two spectra and three tail families; zero violations | `synthetic/high_dim.pkl` |
| `frontier.py` | Margin-quantile frontier with calibration/test split and DKW band (600 cells) | `synthetic/frontier.pkl` |
| `certificate.py` | Plug-in certificate against the mean-confidence null; decision rule at q = 0.02 | `synthetic/certificate.pkl` |
| `certificate_roc.py` | AUROC, AUPRC and precision at matched recall for the certificate and the null | console |

### CIFAR-10 → CIFAR-10-C (`experiments/cifar10c`)

Run each script with `clip` and with `dinov2`.

| Script | Result | Output |
|---|---|---|
| `shift_pairs.py` | Collinearity, residual rank, the gap bound with three margin laws, certificate rank agreement, label-free baselines, refit (fixed and cross-validated penalty) and selective corrector at B ≤ 400 | `cifar10c/shift_pairs_<enc>.pkl` |
| `leaderboard.py` | 12 correctors over 95 paired shifts with sign-flip permutation tests and bootstrap intervals | `cifar10c/leaderboard_<enc>.pkl` |
| `budget_allocation.py` | Routes A, B and C at B ∈ {50, …, 800} under label drift, with the cross-validated Route A check | `cifar10c/budget_<enc>.pkl` |
| `label_drift.py` | Label drift and the BBSE-debiased common-displacement estimate | `cifar10c/label_drift_<enc>.pkl` |
| `dim_sweep.py` | Crossover budget B* as a function of d by PCA truncation | `cifar10c/dim_sweep_<enc>.pkl` |
| `route_b_grid.py` | Route B requirement B99 on the d × K grid and its two-exponent fit | `cifar10c/route_b_<enc>.pkl` |
| `route_a_grid.py` | Route A requirement B_ε on the same grid and its two-exponent fit | `cifar10c/route_a_<enc>.pkl` |
| `obstruction.py` | Per-hypothesis and fixed-point correctors with exact displacements, label-free EM per-class translation, standard TENT/SHOT, consistent-set and residual-conflict statistics | `cifar10c/obstruction_<enc>.pkl` |
| `budget_anchored.py` | Source-anchored Route A (ridge toward the source probe, cross-validated) paired with counting on the cells of `budget_allocation.py` | `cifar10c/budget_anchored_<enc>.pkl` |
| `cosine_head.py` | The gap bound and correctors for a probe on L2-normalized features | `cifar10c/cosine_head_<enc>.pkl` |
| `spectrum.py` | Ridge effective dimension of the PCA-truncated features | console |
| `fewshot.py` | Weight-space interpolation (WiSE-FT style) and the Pro2-style projected probe on the labeled draws of `budget_allocation.py` | `cifar10c/fewshot_<enc>.pkl` |
| `reference.py` | Target-supervised references (target probe, anchored refit with all labels, 20-NN) on held-out halves, and TENT/SHOT over a 16-configuration grid selected on the other corruptions | `cifar10c/reference_<enc>.pkl` |

After both encoders have run:

| Script | Result |
|---|---|
| `bound_tightness.py` | Ratio bound/gap for the three margin laws, overall and by severity |
| `certificate_pairs.py` | Per-pair paired test of certificate against null rank agreement |
| `separation.py` | Separation exponent log(B_A/B_B) and the parameter-count fits |

### DomainNet (`experiments/domainnet`)

Run each analysis with `clip` and with `dino` and the class counts
`10,25,50,100,345`. `selective.py`, `tta.py` and `tta_aux.py` take a third argument that
names the output file, so several class-count lists can run in parallel and are merged
by the report scripts.

| Script | Result | Output |
|---|---|---|
| `core.py` | Drift, gap, the gap bound with three margin laws, Γ_c > 0 counts, label-shift estimators (BBSE, RLLS, MLLS with bias-corrected temperature scaling), Routes A, B, C at B ∈ {5K, 10K, 25K} | `domainnet/dnet_<enc>.pkl` |
| `decision_rule.py` | Decision rule on natural shift and cross-validated Route A | `domainnet/dnet_extra_<enc>.pkl` |
| `selective.py` | Budget safety of the selective corrector, B ∈ {50, …, 1600} | `domainnet/dnet_sel_<enc>_<tag>.pkl` |
| `tta.py` | TENT, SHOT, LAME (default and oracle-tuned), common-mean + LAME | `domainnet/dnet_tta_<enc>_<tag>.pkl` |
| `tta_aux.py` | BN, CORAL, T3A, scale alignment, per-class with predicted labels | `domainnet/dnet_tta2_<enc>_<tag>.pkl` |
| `obstruction.py` | Per-hypothesis, fixed-point and EM per-class correctors, standard TENT/SHOT, consistent-set statistics | `domainnet/dnet_obstruction_<enc>_<tag>.pkl` |
| `reference.py` | Target-supervised references, TENT/SHOT over the configuration grid, and five uses of a budget (counting, refit from scratch, anchored refit, interpolation, projected probe) at natural label drift | `domainnet/dnet_reference_<enc>_<tag>.pkl` |
| `core_report.py`, `decision_rule_report.py`, `selective_report.py`, `tta_report.py` | Summaries with paired tests | console and `.txt` |
| `subsample_check.py` | Effect of the 4,000-point target subsample used by the transductive baselines | console |

### Zero-shot head (`experiments/zeroshot`)

After `python data_prep/text_embed.py`:

| Script | Result | Output |
|---|---|---|
| `tps_views.py` | Test-time prototype shifting in its own protocol on the zero-shot CLIP head: 63 augmented views per CIFAR-10-C image, confident-view selection, one Adam step on per-class or shared prototype shifts, against the view ensemble, the common-mean correction and the oracle translation | `zeroshot/tps_views.pkl` |
| `tps.py cifar`, `tps.py domainnet <K list> [tag]` | Transductive prototype shifting with an information-maximization objective, on the zero-shot head and on the source probe, beside the feature translations | `zeroshot/tps_cifar.pkl`, `zeroshot/tps_domainnet_<tag>.pkl` |

### Figures

```bash
python figures/make_figures.py
```

writes `results/figures/f1_rotation.pdf`, `f2_realbound.pdf`, `f3_domainnet.pdf` and
`f4_allocation.pdf`. Any figure whose inputs are missing is skipped.

## Conventions

* A probe is a pair `(W, b)` with `W` of shape `(K, d)`; scores are `X @ W.T + b`.
  The source probe is a one-versus-rest ridge regression with penalty 1e-2.
* CIFAR features are divided by the mean training-set feature norm before fitting.
* The common displacement is the difference of target and source means; the per-class
  oracle subtracts the true class-conditional displacement and is used only as a
  reference.
* Accuracies are reported as fractions in [0, 1]; intervals are 95 percent normal
  intervals over cells unless a script states a bootstrap or permutation procedure.

## License

MIT. See `LICENSE`.
