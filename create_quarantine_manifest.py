import os
import hashlib
import csv
from collections import defaultdict

DATASET_DIR = "skin_dataset"
QUARANTINE_DIR = "duplicate_quarantine"

SPLITS = [
    "train",
    "validation",
    "test"
]

print("\n======================================")
print("GLOW CAST SAFE QUARANTINE MANIFEST")
print("======================================")

manifest = []

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

            source_path = os.path.join(
                root,
                file
            )

            with open(source_path, "rb") as f:
                file_hash = hashlib.md5(
                    f.read()
                ).hexdigest()

            relative_path = os.path.relpath(
                source_path,
                split_path
            )

            hashes[file_hash].append(
                relative_path
            )

    for file_hash, files in hashes.items():

        if len(files) <= 1:
            continue

        # Keep the first copy in the dataset.
        # Quarantine the remaining duplicate copies.
        keep_file = files[0]

        for duplicate_file in files[1:]:

            source_path = os.path.join(
                split_path,
                duplicate_file
            )

            quarantine_path = os.path.join(
                QUARANTINE_DIR,
                split,
                duplicate_file
            )

            already_exists = os.path.exists(
                quarantine_path
            )

            manifest.append([
                split,
                os.path.dirname(duplicate_file),
                keep_file,
                duplicate_file,
                file_hash,
                "EXISTS"
                if already_exists
                else "NEW"
            ])

print("\nDuplicate copies identified:",
      len(manifest))

print("\n======================================")
print("BY SPLIT")
print("======================================")

for split in SPLITS:

    count = sum(
        1
        for row in manifest
        if row[0] == split
    )

    print(
        f"{split.upper():12}: {count}"
    )

manifest_file = "quarantine_manifest.csv"

with open(
    manifest_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "split",
        "class",
        "keep_file",
        "duplicate_file",
        "md5_hash",
        "quarantine_status"
    ])

    writer.writerows(manifest)

print("\nManifest saved:")
print(manifest_file)

print("\n======================================")
print("STATUS")
print("======================================")
print("NO FILES MOVED")
print("NO FILES DELETED")
print("NO DATASET CHANGED")
print("======================================")