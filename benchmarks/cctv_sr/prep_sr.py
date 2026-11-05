"""Build leakage-aware cross-validation folds for the CCTV-SR indoor/outdoor datasets.

Why: the Roboflow train/valid/test split is random over consecutive frames of ONE fixed camera, so every
test frame has near-identical neighbours in train. Instead, all frames are pooled, sorted by frame id and cut
into K contiguous time blocks. Each block is the test set once, and train frames within BUFFER frame ids of
that block are dropped so the model never trains on near-copies of test frames.

Tasks are different class mappings over the SAME boxes:
    indoor : person (all people) | gender (female/male) | allclass (3 original classes, incl. windows)
    outdoor: person | gender (posture merged) | posture (sitting/standing/walking, gender merged) | allclass (9)
Images with no box for a task are kept as background images.

Output (images are symlinks to the originals, labels are rewritten):
    data/sr/<dataset>/<task>/fold<k>/{train,test}/{images,labels}/  and  fold<k>/data.yaml
    data/sr/manifest.json   fold sizes and test frame-id ranges

Usage: python benchmarks/cctv_sr/prep_sr.py     (expects the unzipped exports in data/indoor and data/outdoor)
"""

import json
import re
import shutil

import numpy as np
from common import DATA, DATASETS, SPLITS, class_names, is_person, parse_label_line

OUT = DATA / "sr"
N_FOLDS = 5
# Frames (by frame id) removed from train around each test block. Outdoor uses a wider buffer because it
# has many more frames per second of recording.
BUFFER = {"indoor": 5, "outdoor": 10}


def task_mappings(names: list[str]) -> dict[str, tuple[dict[str, int], list[str]]]:
    """For every task: ({original class name -> new class id}, [new class names]).

    Original classes missing from the mapping (e.g. windows in the person task) are dropped.
    """
    people = [n for n in names if is_person(n)]
    tasks = {
        "person": ({n: 0 for n in people}, ["person"]),
        "gender": ({n: 0 if n.startswith("Female") else 1 for n in people}, ["female", "male"]),
        "allclass": ({n: i for i, n in enumerate(names)}, list(names)),
    }
    postures = sorted({n.split("_")[1] for n in people})  # Sitting / Standing / Walking
    if len(postures) > 1:  # indoor only has "Sitting", so it has no posture task
        tasks["posture"] = ({n: postures.index(n.split("_")[1]) for n in people}, [p.lower() for p in postures])
    return tasks


def load_frames(dataset: str, names: list[str]):
    """Pool the original train/valid/test frames.

    Returns [(frame_id, image_path, [(class_name, box)])] sorted by frame id.
    """
    src = DATA / dataset
    frames = []
    for split in SPLITS:
        for image in (src / split / "images").glob("*.jpg"):
            frame_id = int(re.search(r"frame_0*(\d+)", image.name).group(1))
            label_file = src / split / "labels" / (image.stem + ".txt")
            boxes = []
            for line in label_file.read_text().splitlines():
                class_id, box = parse_label_line(line)
                boxes.append((names[class_id], box))
            frames.append((frame_id, image, boxes))
    frames.sort(key=lambda f: f[0])
    return frames


def write_fold(task_dir, fold, split, items, mapping):
    """Write one split of one fold: image symlinks plus label files remapped to the task's classes."""
    folder = task_dir / f"fold{fold}" / split
    (folder / "images").mkdir(parents=True)
    (folder / "labels").mkdir(parents=True)
    for frame_id, image, boxes in items:
        (folder / "images" / f"f{frame_id:04d}.jpg").symlink_to(image.resolve())
        lines = "".join(f"{mapping[name]} " + " ".join(f"{x:.6f}" for x in box) + "\n" for name, box in boxes if name in mapping)
        (folder / "labels" / f"f{frame_id:04d}.txt").write_text(lines)


def main() -> None:
    manifest = {}
    for dataset in DATASETS:
        names = class_names(DATA / dataset)
        tasks = task_mappings(names)
        frames = load_frames(dataset, names)
        blocks = np.array_split(np.arange(len(frames)), N_FOLDS)  # contiguous blocks in time order

        # Which frames go to train/test in each fold (the same for every task).
        folds = []
        for block in blocks:
            test = [frames[i] for i in block]
            lo = frames[block[0]][0] - BUFFER[dataset]
            hi = frames[block[-1]][0] + BUFFER[dataset]
            train = [f for f in frames if not lo <= f[0] <= hi]  # drop test block + buffer from train
            folds.append((train, test))
        manifest[dataset] = {
            "n": len(frames),
            "folds": {
                k: {"n_train": len(tr), "n_test": len(te), "test_ids": [te[0][0], te[-1][0]]} for k, (tr, te) in enumerate(folds)
            },
        }

        for task, (mapping, task_classes) in tasks.items():
            task_dir = OUT / dataset / task
            if task_dir.exists():
                shutil.rmtree(task_dir)
            for k, (train, test) in enumerate(folds):
                write_fold(task_dir, k, "train", train, mapping)
                write_fold(task_dir, k, "test", test, mapping)
                (task_dir / f"fold{k}" / "data.yaml").write_text(
                    f"path: {task_dir / f'fold{k}'}\ntrain: train/images\nval: test/images\ntest: test/images\n"
                    f"nc: {len(task_classes)}\nnames: {task_classes}\n"
                )
        sizes = [(v["n_train"], v["n_test"]) for v in manifest[dataset]["folds"].values()]
        print(dataset, len(frames), "frames; tasks:", list(tasks), "; folds (train/test):", sizes)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
