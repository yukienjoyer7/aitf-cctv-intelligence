"""Small deterministic helpers for grouping and splitting research examples."""

import random
from collections import Counter, defaultdict

SPLITS = ("train", "val", "test")
FRACTIONS = (0.7, 0.15, 0.15)


class Groups:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def choose_group_split(records, seed=42, trials=2000):
    """Choose a reproducible approximate stratified split without dividing any group."""
    grouped = defaultdict(Counter)
    for row in records:
        grouped[row["group_id"]][row["label"]] += 1
    labels = sorted({row["label"] for row in records})
    totals = Counter(row["label"] for row in records)
    if len(labels) < 2 or len(grouped) < 3:
        raise ValueError("At least two labels and three groups are required")
    group_items = sorted(grouped.items())
    rng = random.Random(seed)
    best, best_score = None, float("inf")
    for _ in range(trials):
        counts = {split: Counter() for split in SPLITS}
        allocation = {}
        for group, label_counts in group_items:
            split = rng.choices(SPLITS, weights=FRACTIONS, k=1)[0]
            allocation[group] = split
            counts[split].update(label_counts)
        if any(counts[split][label] == 0 for split in SPLITS for label in labels):
            continue
        score = sum(
            ((counts[split][label] - totals[label] * fraction) / max(totals[label] * fraction, 1)) ** 2
            for split, fraction in zip(SPLITS, FRACTIONS, strict=True)
            for label in labels
        )
        if score < best_score:
            best, best_score = allocation, score
    if best is None:
        raise ValueError("Cannot assign all labels to each split with the available groups")
    return best
