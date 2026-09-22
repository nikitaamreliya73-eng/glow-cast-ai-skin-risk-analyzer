import os
from collections import Counter

DATASET_DIR = "skin_dataset"

SPLITS = [
    "train",
    "validation",
    "test"
]

CLASSES = [
    "dry",
    "normal",
    "oily"
]

print("\n======================================")
print("GLOW CAST DATASET IMBALANCE ANALYSIS")
print("======================================")

overall_counts = Counter()

for split in SPLITS:

    split_path = os.path.join(
        DATASET_DIR,
        split
    )

    counts = Counter()

    for class_name in CLASSES:

        class_path = os.path.join(
            split_path,
            class_name
        )

        if not os.path.exists(class_path):
            continue

        for file in os.listdir(class_path):

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            ):
                counts[class_name] += 1
                overall_counts[class_name] += 1

    total = sum(counts.values())

    print("\n--------------------------------------")
    print(split.upper())
    print("--------------------------------------")

    print("Total images:", total)

    for class_name in CLASSES:

        count = counts[class_name]

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        print(
            f"{class_name.upper():8} : "
            f"{count:4} images | "
            f"{percentage:6.2f}%"
        )

    if total > 0:

        largest = max(counts.values())
        smallest = min(counts.values())

        imbalance_ratio = (
            largest / smallest
            if smallest > 0
            else 0
        )

        print(
            "\nLargest / Smallest ratio:",
            round(imbalance_ratio, 2)
        )

print("\n======================================")
print("OVERALL DATASET")
print("======================================")

overall_total = sum(
    overall_counts.values()
)

print("Total images:", overall_total)

for class_name in CLASSES:

    count = overall_counts[class_name]

    percentage = (
        count / overall_total * 100
        if overall_total > 0
        else 0
    )

    print(
        f"{class_name.upper():8} : "
        f"{count:4} images | "
        f"{percentage:6.2f}%"
    )

print("\n======================================")
print("ANALYSIS COMPLETE")
print("======================================")