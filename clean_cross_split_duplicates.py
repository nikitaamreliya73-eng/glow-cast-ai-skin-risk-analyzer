import os
import hashlib
import shutil
from collections import defaultdict


# ==========================================
# GLOW CAST — SAFE DUPLICATE CLEANUP
# ==========================================

DATASET_DIR = "skin_dataset"

QUARANTINE_DIR = "duplicate_quarantine"

SPLITS = [
    "train",
    "validation",
    "test"
]

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)


# ==========================================
# 1. HASH FUNCTION
# ==========================================

def get_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ==========================================
# 2. COLLECT IMAGES
# ==========================================

hash_map = defaultdict(list)

print("\n==============================")
print("GLOW CAST SAFE CLEANUP")
print("==============================")


for split in SPLITS:

    split_path = os.path.join(
        DATASET_DIR,
        split
    )

    for root, _, files in os.walk(split_path):

        for file in files:

            if not file.lower().endswith(
                IMAGE_EXTENSIONS
            ):
                continue

            file_path = os.path.join(
                root,
                file
            )

            file_hash = get_file_hash(
                file_path
            )

            hash_map[file_hash].append(
                (
                    split,
                    file_path
                )
            )


# ==========================================
# 3. FIND CROSS-SPLIT DUPLICATES
# ==========================================

cross_split_groups = {}

for file_hash, files in hash_map.items():

    splits_found = set(
        split
        for split, _
        in files
    )

    if len(splits_found) > 1:

        cross_split_groups[
            file_hash
        ] = files


print(
    "\nCross-split duplicate groups:",
    len(cross_split_groups)
)


# ==========================================
# 4. DETERMINE FILES TO QUARANTINE
# ==========================================

files_to_move = []


for file_hash, files in cross_split_groups.items():

    # --------------------------------------
    # Keep TEST copy whenever available
    # --------------------------------------

    test_files = [
        path
        for split, path in files
        if split == "test"
    ]

    validation_files = [
        path
        for split, path in files
        if split == "validation"
    ]

    train_files = [
        path
        for split, path in files
        if split == "train"
    ]


    # --------------------------------------
    # If TEST exists
    # --------------------------------------

    if test_files:

        # Keep test
        # Remove/quarantine train + validation

        for path in train_files:
            files_to_move.append(path)

        for path in validation_files:
            files_to_move.append(path)


    # --------------------------------------
    # If no TEST but VALIDATION exists
    # --------------------------------------

    elif validation_files:

        # Keep validation
        # Quarantine train

        for path in train_files:
            files_to_move.append(path)


# ==========================================
# 5. DISPLAY PLAN
# ==========================================

print("\n==============================")
print("CLEANUP PLAN")
print("==============================")

print(
    "Files to quarantine:",
    len(files_to_move)
)

print(
    "Original files will NOT be deleted."
)


# ==========================================
# 6. SHOW FILES
# ==========================================

print("\nFiles that will be moved:\n")


for number, path in enumerate(
    files_to_move,
    start=1
):

    print(
        f"{number}. {path}"
    )


# ==========================================
# 7. CREATE QUARANTINE FOLDER
# ==========================================

os.makedirs(
    QUARANTINE_DIR,
    exist_ok=True
)


# ==========================================
# 8. MOVE FILES SAFELY
# ==========================================

moved_count = 0


for path in files_to_move:

    # Preserve original relative path

    relative_path = os.path.relpath(
        path,
        DATASET_DIR
    )

    destination = os.path.join(
        QUARANTINE_DIR,
        relative_path
    )

    destination_folder = os.path.dirname(
        destination
    )

    os.makedirs(
        destination_folder,
        exist_ok=True
    )


    # Move file

    shutil.move(
        path,
        destination
    )

    moved_count += 1


# ==========================================
# 9. FINAL RESULT
# ==========================================

print("\n==============================")
print("CLEANUP COMPLETED")
print("==============================")


print(
    "Files moved to quarantine:",
    moved_count
)

print(
    "\nQuarantine folder:"
)

print(
    QUARANTINE_DIR
)


print(
    "\nOriginal images were NOT permanently deleted."
)

print(
    "\nGLOW CAST dataset cleanup completed!"
)