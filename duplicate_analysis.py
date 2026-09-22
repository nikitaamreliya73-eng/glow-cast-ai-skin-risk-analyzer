import os
import hashlib
from collections import defaultdict, Counter

DATASET_DIR = "skin_dataset"

SPLITS = [
    "train",
    "validation",
    "test"
]

print("\n======================================")
print("GLOW CAST DATASET DUPLICATE ANALYSIS")
print("======================================")

for split in SPLITS:

    split_path = os.path.join(DATASET_DIR, split)

    hashes = defaultdict(list)

    total_images = 0

    for root, dirs, files in os.walk(split_path):

        for file in files:

            if not file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            ):
                continue

            path = os.path.join(root, file)

            total_images += 1

            with open(path, "rb") as f:
                file_hash = hashlib.md5(
                    f.read()
                ).hexdigest()

            relative_path = os.path.relpath(
                path,
                split_path
            )

            hashes[file_hash].append(
                relative_path
            )

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

    print("\n--------------------------------------")
    print(split.upper())
    print("--------------------------------------")

    print("Total images:", total_images)
    print("Unique images:", len(hashes))
    print("Duplicate groups:", len(duplicate_groups))
    print("Files in duplicate groups:", duplicate_files)
    print("Extra duplicate copies:", extra_copies)

    class_duplicate_groups = Counter()

    for files in duplicate_groups.values():

        first_file = files[0]

        class_name = first_file.split(
            os.sep
        )[0]

        class_duplicate_groups[
            class_name
        ] += 1

    print("\nDuplicate groups by class:")

    for class_name in [
        "dry",
        "normal",
        "oily"
    ]:

        print(
            f"{class_name.upper():8}: "
            f"{class_duplicate_groups[class_name]}"
        )

print("\n======================================")
print("ANALYSIS COMPLETE")
print("======================================")