# Model research architecture

The implemented component is the isolated [CCTV-SR benchmark](../benchmarks/cctv_sr/README.md). The remaining task paths contain plans, not inference implementations or trained weights. A person detector is an optional upstream primitive for ROI experiments; YOLO26s is a provisional candidate from the benchmark accuracy/cost tradeoff, not a mandated dependency.

Smoking research can compare cue detection with a person or upper-body ROI classifier and temporal confirmation. Violence requires temporal classification across full frames, interaction regions or multi-person crops; a pretrained R3D-18 is a candidate. Flag research can compare a class-agnostic detector plus verified-crop classifier with a single-stage multiclass detector. No flag and ambiguous/unknown are separate from the binary crop head. Exact preprocessing and outputs remain experimental until measured and documented in the [handoff](MODEL_HANDOFF.md).

CV owns model development, evaluation and model-side contracts. The integration team owns video buffers, tracking, scheduling, event management, APIs and deployment. No such runtime is implemented here.
