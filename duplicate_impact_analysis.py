import os
import hashlib
from collections import defaultdict

DATASET_DIR = "skin_dataset"

CLASSES = [
    "dry",
    "normal",
    "oily"
]

TRAIN_DIR = os.path.join(
    DATASET_DIR,
    "train"
)

print("\n======================================")
print("GLOW CAST DUPLICATE IMPACT ANALYSIS")
print("======================================")

for class_name in CLASSES:

    class_path = os.path.join(
        TRAIN_DIR,
        class_name
    )

    hashes = defaultdict(list)

    total_images = 0

    for file in os.listdir(class_path):

        if not file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            continue

        path = os.path.join(
            class_path,
            file
        )

        total_images += 1

        with open(path, "rb") as f:
            file_hash = hashlib.md5(
                f.read()
            ).hexdigest()

        hashes[file_hash].append(file)

    duplicate_groups = {
        h: files
        for h, files in hashes.items()
        if len(files) > 1
    }

    duplicate_files = sum(
        len(files)
        for files in duplicate_groups.values()
    )

    extra_copies = sum(
        len(files) - 1
        for files in duplicate_groups.values()
    )

    unique_images = len(hashes)

    duplicate_percentage = (
        duplicate_files / total_images * 100
        if total_images > 0
        else 0
    )

    print("\n--------------------------------------")
    print(class_name.upper())
    print("--------------------------------------")

    print("Total images:", total_images)
    print("Unique images:", unique_images)
    print("Duplicate groups:", len(duplicate_groups))
    print("Files in duplicate groups:", duplicate_files)
    print("Extra duplicate copies:", extra_copies)

    print(
        "Files involved in duplicates:",
        f"{duplicate_percentage:.2f}%"
    )

print("\n======================================")
print("ANALYSIS COMPLETE")
print("======================================")