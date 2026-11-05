# Migration

All ten scripts moved from `src/` to `benchmarks/cctv_sr/`; all historical `results/sr/`, CSVs, latency JSON, `fig_sr/` and slide notes moved under `results/cctv_sr/`. The local ignored raw-frame examples moved with their folder and remain ignored. Readers now resolve benchmark results there; archived measurements and slide text remain unchanged, so paths written inside them are historical. See the [recorded results](../results/cctv_sr/) and [benchmark protocol](../benchmarks/cctv_sr/README.md).

Before migration, a local SHA-256 snapshot recorded 107 results files, including 96 job JSONs and two ignored examples. All 107 matched after relocation. The runner keeps its original job order and resume logic, with its root and script paths repaired. The old `python src/<script>` commands become `python benchmarks/cctv_sr/<script>`; runner commands change analogously. The separate untracked `docs/training_pipeline.md` was preserved as found and may still describe old paths.

Offline validation and environment limitations are recorded below. No GPU training, data downloads, synthetic generation or evaluation on the original data occurred during migration.

Validation used a Python 3.12 `.venv` with NumPy, pandas, SciPy, PyYAML, pytest and Ruff: `.venv/bin/pytest -q` passed 7 tests, including temporary-directory aggregation comparison with both archived CSVs; `.venv/bin/ruff check benchmarks src tests` and `.venv/bin/ruff format --check benchmarks src tests` passed. `bash -n benchmarks/cctv_sr/run_sr.sh`, relative links in newly written documentation, and both staged and unstaged `git diff --check` passed. The evaluator `--help` was checked from another working directory without importing the training framework.

The pinned requirements were installed into the Python 3.12 environment after retrying the large PyTorch/CUDA downloads over IPv4. Import checks succeeded for PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130 and Ultralytics 8.4.165, and the offline tests and Ruff checks were rerun successfully. No original data, GPU availability, GPU training or evaluation was tested.
