import os


# ==========================================
# GLOW CAST — DATASET CLASS COUNT
# ==========================================

DATASET_DIR = "skin_dataset"

splits = [
    "train",
    "validation",
    "test"
]

classes = [
    "dry",
    "normal",
    "oily"
]


print("\n==============================")
print("GLOW CAST DATASET CHECK")
print("==============================")


for split in splits:

    print(f"\n{split.upper()} DATASET")
    print("------------------------------")

    total = 0

    for class_name in classes:

        folder = os.path.join(
            DATASET_DIR,
            split,
            class_name
        )

        count = len([
            file
            for file in os.listdir(folder)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        print(
            f"{class_name.upper():8} : {count}"
        )

        total += count

    print("------------------------------")
    print(f"TOTAL    : {total}")


print("\n==============================")
print("CHECK COMPLETED")
print("==============================")