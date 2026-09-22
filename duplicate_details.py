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
print("GLOW CAST DUPLICATE DEPTH ANALYSIS")
print("======================================")

for split in SPLITS:

    split_path = os.path.join(
        DATASET_DIR,
        split
    )

    hashes = defaultdict(list)

    for root, dirs, files in os.walk(split_path):

        for file in files:

            if not file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            ):
                continue

            path = os.path.join(root, file)

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

    print("\n--------------------------------------")
    print(split.upper())
    print("--------------------------------------")

    if not duplicate_groups:
        print("No duplicates found.")
        continue

    depth_counter = Counter()

    for files in duplicate_groups.values():

        depth_counter[len(files)] += 1

    print(
        "Total duplicate groups:",
        len(duplicate_groups)
    )

    print("\nDuplicate depth:")

    for copies in sorted(depth_counter):

        groups = depth_counter[copies]

        print(
            f"{copies} copies : "
            f"{groups} groups"
        )

    print("\nSample duplicate groups:")

    count = 0

    for file_hash, files in duplicate_groups.items():

        print("\nHash:", file_hash)

        for file in files:
            print("  ", file)

        count += 1

        if count >= 5:
            break

print("\n======================================")
print("ANALYSIS COMPLETE")
print("======================================")