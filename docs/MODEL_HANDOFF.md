# Model handoff contract

Exploratory smoking and violence checkpoints and validation-selected thresholds exist locally, but are not committed or validated for deployment. No production checkpoint or operationally calibrated threshold is supplied. For each future model stage, deliver weights plus a machine-readable manifest to the integration teammate. Suggested local layout is `artifacts/<task>/<version>/model.pt` and `manifest.json`; a flag version may contain `detector.pt` and `classifier.pt`. Artifacts are excluded from Git. A shared person detector is optional and must have its own contract.

The manifest must specify:

- `task_id`, `stage_id`, architecture/family and model version; artifact location, framework, checkpoint format and SHA-256 checksum. Declare ONNX only if actually exported and include tested runtime/dependency versions.
- Input semantics and exact color order, resize/letterbox/crop, channel order, normalization, tensor shape, and for temporal models clip length and sampling rate.
- Ordered class labels, unit of prediction, output tensor shapes and semantics, postprocessing, confidence interpretation, ambiguity and abstention behavior. Flag binary labels apply only to verified flag crops; no flag is separate.
- Thresholds calibrated on validation data and calibration provenance, or `null` and `uncalibrated` until available.
- Training dataset IDs, versions and licenses, split manifest hashes, training commit and seed; evaluation metrics with dataset/split provenance, ambiguity handling and model limitations.

This specifies the model-side handoff only. API and event management remain with integration. Historical benchmark runs delete their checkpoints and are not production weight sources.
