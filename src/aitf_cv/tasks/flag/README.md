# Indonesian flag recognition

Status: planned, not trained. Detect a flag and classify verified flag crops as `indonesian` or `other`. The overall pipeline also returns `no_flag` and `unknown` for ambiguous evidence; a no-flag frame is never a negative flag crop. Compare a class-agnostic detector plus crop classifier with a single-stage multiclass detector.

License-reviewed real data should include Monaco (same red-over-white layout), Poland (white over red), other flags, banners and cropped, occluded or waving fabric. Perspective and waving can make proportions inadequate to distinguish Indonesia from Monaco, so abstention is required. Report detection mAP, conditional classification macro-F1 and confusion matrix separately, plus end-to-end accuracy on complete frames/videos. Synthetic stress tests are supplementary.
