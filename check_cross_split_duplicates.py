import os
import hashlib
from collections import defaultdict


# ==========================================
# GLOW CAST — CROSS SPLIT DUPLICATE CHECK
# ==========================================

DATASET_DIR = "skin_dataset"

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
# 1. FILE HASH FUNCTION
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
# 2. COLLECT ALL IMAGES
# ==========================================

hash_map = defaultdict(list)

total_images = 0


print("\n==============================")
print("GLOW CAST DUPLICATE CHECK")
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

            total_images += 1


# ==========================================
# 3. FIND DUPLICATE GROUPS
# ==========================================

duplicate_groups = {

    file_hash: files

    for file_hash, files
    in hash_map.items()

    if len(files) > 1
}


# ==========================================
# 4. FIND CROSS-SPLIT DUPLICATES
# ==========================================

cross_split_groups = {

    file_hash: files

    for file_hash, files
    in duplicate_groups.items()

    if len(
        set(
            split
            for split, _
            in files
        )
    ) > 1
}


# ==========================================
# 5. COUNT DUPLICATES
# ==========================================

same_split_groups = {

    file_hash: files

    for file_hash, files
    in duplicate_groups.items()

    if len(
        set(
            split
            for split, _
            in files
        )
    ) == 1
}


cross_split_files = sum(
    len(files)
    for files
    in cross_split_groups.values()
)


cross_split_extra = sum(
    len(files) - 1
    for files
    in cross_split_groups.values()
)


# ==========================================
# 6. DISPLAY SUMMARY
# ==========================================

print("\n==============================")
print("DATASET SUMMARY")
print("==============================")

print(
    "Total images:",
    total_images
)

print(
    "Unique images:",
    len(hash_map)
)

print(
    "Duplicate groups:",
    len(duplicate_groups)
)

print(
    "Same-split duplicate groups:",
    len(same_split_groups)
)

print(
    "Cross-split duplicate groups:",
    len(cross_split_groups)
)

print(
    "Files involved in cross-split duplicates:",
    cross_split_files
)

print(
    "Extra duplicate copies across splits:",
    cross_split_extra
)


# ==========================================
# 7. SHOW CROSS-SPLIT DUPLICATES
# ==========================================

print("\n==============================")
print("CROSS-SPLIT DUPLICATES")
print("==============================")


if not cross_split_groups:

    print(
        "\nNo cross-split duplicates found."
    )

else:

    print(
        f"\nFound {len(cross_split_groups)} "
        "cross-split duplicate groups."
    )

    print(
        "\nShowing first 20 groups:\n"
    )

    for number, (file_hash, files) in enumerate(
        cross_split_groups.items(),
        start=1
    ):

        if number > 20:
            break

        print(
            f"Duplicate Group {number}"
        )

        for split, file_path in files:

            print(
                f"  {split.upper()} : {file_path}"
            )

        print()


# ==========================================
# 8. FINAL RESULT
# ==========================================

print("==============================")
print("CHECK COMPLETED")
print("==============================")