#!/bin/bash
# Reproduce every result in order, on one machine.  Each step writes its pickle to
# results/ and its console summary to results/logs/.  Steps that only read pickles run
# in seconds; the encoders and DomainNet steps need a GPU and several hours.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
export PYTHONHASHSEED=0 MPLBACKEND=Agg
run() { local name=$1; shift; echo "[$(date +%H:%M:%S)] $name"; "$@" > "results/logs/$name.log" 2>&1; }

run download_cifar      python data_prep/download_cifar.py
for e in clip dinov2; do run encode_cifar_$e python data_prep/encode_cifar.py $e; done
for d in clipart infograph painting quickdraw real sketch; do
  run embed_domainnet_$d python data_prep/embed_domainnet.py $d
done

run rotation            python experiments/synthetic/rotation.py
run frontier            python experiments/synthetic/frontier.py
run high_dim            python experiments/synthetic/high_dim.py
run certificate         python experiments/synthetic/certificate.py
run certificate_roc     python experiments/synthetic/certificate_roc.py

for e in clip dinov2; do
  run shift_pairs_$e    python experiments/cifar10c/shift_pairs.py $e
  run leaderboard_$e    python experiments/cifar10c/leaderboard.py $e
  run label_drift_$e    python experiments/cifar10c/label_drift.py $e
  run budget_$e         python experiments/cifar10c/budget_allocation.py $e
  run dim_sweep_$e      python experiments/cifar10c/dim_sweep.py $e
  run route_b_$e        python experiments/cifar10c/route_b_grid.py $e
  run route_a_$e        python experiments/cifar10c/route_a_grid.py $e
  run obstruction_$e    python experiments/cifar10c/obstruction.py $e
  run anchored_$e       python experiments/cifar10c/budget_anchored.py $e
  run cosine_head_$e    python experiments/cifar10c/cosine_head.py $e
  run fewshot_$e        python experiments/cifar10c/fewshot.py $e
  run reference_$e      python experiments/cifar10c/reference.py $e
done
run bound_tightness     python experiments/cifar10c/bound_tightness.py
run certificate_pairs   python experiments/cifar10c/certificate_pairs.py
run separation          python experiments/cifar10c/separation.py
run spectrum            python experiments/cifar10c/spectrum.py

for e in clip dino; do
  run dnet_core_$e      python experiments/domainnet/core.py $e 10,25,50,100,345
  run dnet_rule_$e      python experiments/domainnet/decision_rule.py $e 10,25,50,100,345
  run dnet_sel_$e       python experiments/domainnet/selective.py $e 10,25,50,100,345 all
  run dnet_tta_$e       python experiments/domainnet/tta.py $e 10,25,50,100,345 all
  run dnet_tta_aux_$e   python experiments/domainnet/tta_aux.py $e 10,25,50,100,345 all
  run dnet_obstr_$e     python experiments/domainnet/obstruction.py $e 10,25,50,100,345 all
  run dnet_ref_$e       python experiments/domainnet/reference.py $e 10,25,50,100,345 all
done
run text_embed          python data_prep/text_embed.py
run tps_views           python experiments/zeroshot/tps_views.py
run tps_cifar           python experiments/zeroshot/tps.py cifar
run tps_domainnet       python experiments/zeroshot/tps.py domainnet 10,25,50,100,345 all
run dnet_core_report    python experiments/domainnet/core_report.py
run dnet_rule_report    python experiments/domainnet/decision_rule_report.py
run dnet_sel_report     python experiments/domainnet/selective_report.py
run dnet_tta_report     python experiments/domainnet/tta_report.py
run dnet_subsample      python experiments/domainnet/subsample_check.py

run figures             python figures/make_figures.py
echo "[$(date +%H:%M:%S)] done"
