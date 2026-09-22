import os
import shutil
import csv

MANIFEST_FILE = "quarantine_manifest.csv"
QUARANTINE_DIR = "duplicate_quarantine"

print("\n======================================")
print("GLOW CAST SAFE QUARANTINE COPY")
print("======================================")

with open(
    MANIFEST_FILE,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    rows = list(reader)

new_count = 0
existing_count = 0

for row in rows:

    split = row["split"]
    duplicate_file = row["duplicate_file"]

    source_path = os.path.join(
        "skin_dataset",
        split,
        duplicate_file
    )

    destination_path = os.path.join(
        QUARANTINE_DIR,
        split,
        duplicate_file
    )

    os.makedirs(
        os.path.dirname(destination_path),
        exist_ok=True
    )

    if os.path.exists(destination_path):

        existing_count += 1

        print(
            "ALREADY EXISTS:",
            destination_path
        )

        continue

    shutil.copy2(
        source_path,
        destination_path
    )

    new_count += 1

print("\n======================================")
print("COPY COMPLETE")
print("======================================")

print(
    "New files copied:",
    new_count
)

print(
    "Already existing:",
    existing_count
)

print(
    "Total manifest entries:",
    len(rows)
)

print("\nIMPORTANT:")
print("Original dataset files were NOT moved.")
print("Original dataset files were NOT deleted.")
print("======================================")