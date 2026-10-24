"""Convert HaGRID annotation json files into train.csv / val.csv / test.csv.
Run from the project root:  python src/convert.py
"""
import csv
import glob
import json
import os
import random

from common import DATA_DIR, CLASSES, mirror, normalize

LIMITS = {"train": 3000, "val": 500, "test": 500}  # max number of images per gesture
random.seed(42)


def find_json(split, name):
    pattern = os.path.join(DATA_DIR, "**", split, f"{name}.json")
    for p in glob.glob(pattern, recursive=True):
        if ".ipynb_checkpoints" not in p:
            return p
    raise FileNotFoundError(f"Could not find {split}/{name}.json inside {DATA_DIR}")


for split, limit in LIMITS.items():
    rows = []
    for name, label in CLASSES.items():
        try:
            path = find_json(split, name)
        except FileNotFoundError as e:
            print("Warning:", e)
            continue
        with open(path) as f:
            data = json.load(f)
        keys = list(data.keys())
        random.shuffle(keys)
        count = 0
        for k in keys:
            if count >= limit:
                break
            item = data[k]
            labels = item.get("labels", [])
            hands = item.get("hand_landmarks", [])
            for lab, pts in zip(labels, hands):
                if lab != name or not pts or len(pts) != 21:
                    continue
                row = normalize(pts)
                rows.append(row + [label])
                count += 1
                if split == "train":
                    rows.append(mirror(row) + [label])
                break                      # one hand per image
        print(split, name, count)
        if count == 0:
            print(f"  Warning: no samples for class {name} (if it's no_gesture, "
                  "the landmarks may be empty in this file; record it from the camera with record.py)")
    random.shuffle(rows)
    out = os.path.join(DATA_DIR, f"{split}.csv")
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"{out}: {len(rows)} rows\n")