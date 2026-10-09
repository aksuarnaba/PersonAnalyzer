"""
import_signs.py - build the sign-language (or custom gesture) dataset from images.

Folder layouts that work:

  A) one sub-folder per label        B) label in the file name
     signs_images/                      signs_images/
       A/ a1.jpg a2.jpg ...               A.jpg  B.png  C_1.jpg  C_2.jpg
       B/ b1.jpg ...

Usage:
  python import_signs.py signs_images
  python import_signs.py gesture_images --set custom
"""
import os
import re
import sys
import argparse
from collections import Counter

import cv2

from hands import HandTracker, GestureKNN

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def label_for(root, path):
    parts = os.path.relpath(path, root).split(os.sep)
    if len(parts) > 1:
        return parts[0]
    stem = os.path.splitext(parts[0])[0]
    return re.sub(r"[\s_\-]*\d+$", "", stem) or stem


def prepare(img):
    """Upscale small images and add a border so hands near the edge still detect."""
    h, w = img.shape[:2]
    if max(h, w) < 480:
        s = 480 / max(h, w)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_REPLICATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--set", choices=["custom", "signs"], default="signs")
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"Not a folder: {args.folder}")

    tracker = HandTracker(max_hands=1)
    knn = GestureKNN(args.set)
    added, skipped = Counter(), []

    for dirpath, _, files in os.walk(args.folder):
        for name in sorted(files):
            if not name.lower().endswith(EXT):
                continue
            path = os.path.join(dirpath, name)
            img = cv2.imread(path)
            if img is None:
                skipped.append(path)
                continue
            hands = tracker.detect(prepare(img))
            if not hands:
                skipped.append(path)
                continue
            label = label_for(args.folder, path)
            knn.add(label, hands[0].feat)
            added[label] += 1

    knn.save()
    print(f"\nImported into data/{args.set}.csv:")
    for label in sorted(added):
        print(f"  {label}: {added[label]}")
    print(f"\nNo hand found / unreadable: {len(skipped)}")
    for p in skipped[:20]:
        print("  ", p)


if __name__ == "__main__":
    main()