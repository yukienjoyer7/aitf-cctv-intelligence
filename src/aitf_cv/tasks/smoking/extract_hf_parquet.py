"""Extract the downloaded smoking imagefolder export into labeled local directories."""

import argparse
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
LABELS = {0: "not_smoking", 1: "smoking"}


def extract(parquet_root: Path, output: Path) -> int:
    """Extract byte-identical images, rejecting unsafe paths and overwrites."""
    files = {split: sorted((parquet_root / split).glob("*.parquet")) for split in ("train", "test")}
    if not all(files.values()):
        raise ValueError("Both train and test Parquet files are required")
    output.mkdir(parents=True, exist_ok=False)
    count = 0
    for split, paths in files.items():
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=["image", "label"]):
                for row in batch.to_pylist():
                    image, label = row["image"], row["label"]
                    if label not in LABELS or not image or not image.get("bytes"):
                        raise ValueError(f"Unexpected or missing image/label in {path}")
                    name = image.get("path")
                    if not name or name in (".", "..") or Path(name).name != name or "\\" in name:
                        raise ValueError(f"Unsafe image filename in {path}")
                    folder = output / split / LABELS[label]
                    folder.mkdir(parents=True, exist_ok=True)
                    with (folder / name).open("xb") as handle:
                        handle.write(image["bytes"])
                    count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet_root", type=Path)
    parser.add_argument("output", type=Path, help="New directory inside data/smoking/")
    args = parser.parse_args()
    target = args.output.resolve()
    permitted = (ROOT / "data" / "smoking").resolve()
    if not target.is_relative_to(permitted) or target == permitted:
        parser.error("output must be a new directory inside data/smoking/")
    print(f"Extracted {extract(args.parquet_root, target)} labeled images to {target}")


if __name__ == "__main__":
    main()
