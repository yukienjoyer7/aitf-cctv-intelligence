"""Find people boxes that overlap almost completely (IoU >= 0.8) but carry different labels.

Two boxes that cover the same person cannot both be right if one says Female and the other Male. In the outdoor
dataset this happens a lot, which caps gender/posture accuracy no matter how good the model is. This script is
the source of the "723 conflicting pairs" figure in the deck.

Counts are over PAIRS of boxes (three stacked boxes = 3 pairs), across all provided splits.

Usage: python src/check_label_conflicts.py            (checks data/indoor and data/outdoor)
"""

from itertools import combinations

from common import DATA, DATASETS, class_names, iou, is_person, read_label_file, xywh_to_xyxy

DUPLICATE_IOU = 0.8


def main() -> None:
    for dataset in DATASETS:
        names = class_names(DATA / dataset)
        n_people = n_pairs = n_gender = n_posture = 0
        frames_with_pairs = set()
        for path in (DATA / dataset).glob("*/labels/*.txt"):
            people = [(names[c], xywh_to_xyxy(b)) for c, b in read_label_file(path) if is_person(names[c])]
            n_people += len(people)
            for (name_a, box_a), (name_b, box_b) in combinations(people, 2):
                if iou(box_a, box_b) < DUPLICATE_IOU:
                    continue
                n_pairs += 1
                frames_with_pairs.add(path.name)
                gender_a, posture_a = name_a.split("_")
                gender_b, posture_b = name_b.split("_")
                n_gender += gender_a != gender_b
                n_posture += posture_a != posture_b
        print(
            f"{dataset}: people boxes {n_people} | pairs with IoU >= {DUPLICATE_IOU}: {n_pairs} "
            f"in {len(frames_with_pairs)} frames | different gender: {n_gender} | different posture: {n_posture}"
        )


if __name__ == "__main__":
    main()
