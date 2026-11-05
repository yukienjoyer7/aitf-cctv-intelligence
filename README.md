# AITF CCTV Intelligence

Computer vision research for CCTV model development. The CV engineer develops and evaluates model weights and their input/output contracts. API serving, scheduling, tracking, event aggregation and deployment belong to the integration team.

| Task | Status | Scope |
| --- | --- | --- |
| [Person attributes and static objects](src/aitf_cv/tasks/person_attributes/README.md) | benchmark only | Historical detection mappings, not a student classifier |
| [Smoking](src/aitf_cv/tasks/smoking/README.md) | exploratory baseline | Observable smoking behavior |
| [Violence](src/aitf_cv/tasks/violence/README.md) | exploratory baseline | Actions over time |
| [Indonesian flag](src/aitf_cv/tasks/flag/README.md) | planned, not trained | Flag detection and conditional recognition |

The [CCTV-SR benchmark](benchmarks/cctv_sr/README.md) compares YOLO26n, YOLO26s and RT-DETR-L. Its [recorded evidence](results/cctv_sr/) is preliminary: one seed, with outdoor gender, posture and allclass missing fold 4. Two fixed cameras do not represent deployments. Gender labels are annotations, not reliable evidence of gender identity.

To reproduce from the repository root after separately obtaining permitted source exports and dependencies:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# Place the original exports in data/indoor and data/outdoor.
.venv/bin/python benchmarks/cctv_sr/prep_sr.py
benchmarks/cctv_sr/run_sr.sh
.venv/bin/python benchmarks/cctv_sr/aggregate_sr.py
.venv/bin/python benchmarks/cctv_sr/make_figures_sr.py
```

Training is a long GPU job and may download pretrained weights. The runner resumes completed jobs and deletes its checkpoints. Aggregation and figure generation overwrite corresponding outputs, so preserve archived evidence before reproducing. New task settings belong in [configs/tasks](configs/tasks/); smoking and violence have local exploratory checkpoints and validation-selected thresholds, but no operationally validated model or threshold. These artifacts are not committed. See the [offline demos](examples/synthetic_demo/README.md). See the [data protocol](docs/DATA_PROTOCOL.md), [architecture](docs/ARCHITECTURE.md), [handoff contract](docs/MODEL_HANDOFF.md) and [migration notes](docs/MIGRATION.md).
