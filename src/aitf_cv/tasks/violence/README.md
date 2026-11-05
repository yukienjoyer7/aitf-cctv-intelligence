# Violence event recognition

Status: exploratory baseline trained on unreviewed source labels; not production ready. Classify `non_violent` versus `violent` actions across a time window; define physical aggression including fighting, punching and kicking in the protocol and explicitly handle ambiguity. A pretrained R3D-18 and clip sampler are candidate baseline settings, not production specifications. Compare full frames, interaction regions and multi-person temporal crops.

Use public real videos after auditing license, recording/actor duplication and CCTV domain match. Hard negatives include hugging, sports, joking, crowds, fast movement, falls and non-violent confrontations. Evaluate at video/event level on held-out recordings, including time-to-detection and false alarms when timestamps allow. Synthetic stress tests provide scenario coverage, not field reliability.

A local candidate is the [Surveillance Camera Fight Dataset](https://github.com/seymanurakti/fight-detection-surv-dataset). Audit its source index without creating a split:

```bash
.venv/bin/python src/aitf_cv/tasks/violence/audit_sources.py data/violence/fight-detection-surv-dataset
```

The available source index groups multiple clips from the same recording and contains ambiguous or missing source mappings; do not randomly split clips. Resolve or conservatively exclude ambiguous groups, check near-duplicates and review upstream video rights. Its repository license does not by itself establish rights to all source footage. Dataset details remain a candidate in [the registry](../../../../configs/datasets/surveillance_fight.yaml).

At repository revision `ff83b7c521e5ca4eb67a212cc42dbc54f39d6290`, the audit found 300 clips: 269 with one indexed source, 30 without an individual source mapping, and `nofi022` listed under two sources. Two source-dataset ranges overlap at `nofi076`. There were no exact duplicate files by SHA-256; this does not rule out near-duplicate clips. The 269 unambiguous clips span 44 indexed sources, with up to 20 clips from one source. These observations are an audit, not a training split or performance result.

## Provisional grouped split

Prepare seed-42 clip manifests from the locally downloaded dataset:

```bash
PYTHONPATH=src .venv/bin/python -m aitf_cv.tasks.prepare_splits violence data/splits/violence_v1
```

The command requires a **new** output directory and writes `train.csv`, `val.csv`, `test.csv`, `excluded.csv`, and `metadata.json`. Each manifest references original local clips without copying video. Clips with no unique source recording mapping or unreadable sampled frames are excluded. Clips sharing a source URL, exact file hash, or matching three sampled-frame dHashes are kept in the same group; sampled-frame screening can miss related videos and near-duplicates. Source labels remain unreviewed, and repository license metadata does not settle underlying video rights. Review labels, recording/actor overlap, source rights, and the held-out test material before training or reporting results.

The planned first classifier is MoViNet-A0 Base, using the pinned community PyTorch implementation. The grouped manifests and source labels are provisional; no claim of operational accuracy follows from these short clips.

An exploratory run can be started after installing `requirements-training.txt` with:

```bash
PYTHONPATH=src .venv/bin/python -m aitf_cv.tasks.train_baseline violence artifacts/violence/baseline_v1 --allow-unreviewed
```

Choose a new output directory for each run. The pinned PyTorch implementation and Kinetics weights are separate from the benchmark dependencies. Test metrics are preliminary because the source clips and labels are unreviewed.
