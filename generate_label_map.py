# generate_classInd.py

import os

dataset_path = "data/UCF101"
output_path = "data/classInd.txt"

# Get class folder names and sort them
classes = sorted([
    d.strip().rstrip("\\") for d in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, d))
])

# Write class index and class name to the file
with open(output_path, "w") as f:
    for idx, cls in enumerate(classes, 1):
        f.write(f"{idx} {cls}\n")

print(f"classInd.txt generated with {len(classes)} classes at {output_path}")
