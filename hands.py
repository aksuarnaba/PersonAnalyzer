"""
hands.py - hand landmarks, finger counting, built-in gestures,
custom gestures and static sign language (all landmark-based).
"""
import os
import csv
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "hand", "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
DATA_DIR = os.path.join(BASE, "data")

FONT = cv2.FONT_HERSHEY_SIMPLEX
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4),
               (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16),
               (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

# finger -> (pip joint, tip)
FINGER_JOINTS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}


def _d(a, b):
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# Step 1: finger states (rotation independent - uses distances, not "y above")
# ---------------------------------------------------------------------------
def finger_states(p):
    f = {}
    # thumb: tip farther from the pinky base than the thumb's IP joint
    f["thumb"] = _d(p[4], p[17]) > _d(p[3], p[17]) * 1.05
    # other fingers: tip farther from the wrist than the PIP joint
    for name, (pip, tip) in FINGER_JOINTS.items():
        f[name] = _d(p[0], p[tip]) > _d(p[0], p[pip]) * 1.05
    return f


def open_closed(f):
    n = sum(f[k] for k in FINGER_JOINTS)
    return "Open" if n == 4 else "Closed" if n == 0 else "Partial"


# ---------------------------------------------------------------------------
# Step 2: built-in gestures (thumbs up, peace, ...)
# ---------------------------------------------------------------------------
def classify_gesture(p, f):
    four = [f[k] for k in FINGER_JOINTS]
    palm = _d(p[0], p[9])

    # Thumbs up / down: thumb out, other four curled, thumb mostly vertical
    if f["thumb"] and not any(four):
        dx, dy = p[4][0] - p[2][0], p[4][1] - p[2][1]
        if abs(dy) > abs(dx):
            return "Thumbs up" if dy < 0 else "Thumbs down"

    # OK sign: thumb tip touches index tip, other three fingers extended
    if _d(p[4], p[8]) < 0.3 * palm and f["middle"] and f["ring"] and f["pinky"]:
        return "OK"

    if all(four) and f["thumb"]:
        return "Open hand"
    if not any(four):
        return "Fist"
    if f["index"] and f["middle"] and not f["ring"] and not f["pinky"]:
        return "Peace"
    if f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
        return "Pointing"
    if f["index"] and f["pinky"] and not f["middle"] and not f["ring"]:
        return "Rock on"
    return ""


# ---------------------------------------------------------------------------
# Step 3: features for custom gestures / sign language
# ---------------------------------------------------------------------------
def features(pts, label):
    """Translation-, scale- and handedness-invariant 42-value vector."""
    p = pts[:, :2].copy()
    p -= p[0]                       # wrist at origin
    if label == "Left":             # mirror left hands onto right hands
        p[:, 0] *= -1
    s = np.linalg.norm(p, axis=1).max()
    return (p / (s + 1e-6)).flatten()


class GestureKNN:
    """Tiny k-nearest-neighbour classifier stored in data/<name>.csv"""

    def __init__(self, name):
        self.name = name
        self.path = os.path.join(DATA_DIR, f"{name}.csv")
        self.X, self.y, self.new = [], [], []
        if os.path.exists(self.path):
            with open(self.path, newline="") as fh:
                for row in csv.reader(fh):
                    self.y.append(row[0])
                    self.X.append([float(v) for v in row[1:]])

    def add(self, label, feat):
        self.y.append(label)
        self.X.append(list(feat))
        self.new.append((label, list(feat)))

    def save(self):
        if not self.new:
            return
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(self.path, "a", newline="") as fh:
            w = csv.writer(fh)
            for label, feat in self.new:
                w.writerow([label, *feat])
        self.new = []

    def count(self, label):
        return self.y.count(label)

    def predict(self, feat, k=5, max_dist=0.8):
        if not self.y:
            return None
        X = np.asarray(self.X)
        d = np.linalg.norm(X - feat, axis=1)
        idx = np.argsort(d)[:k]
        if d[idx[0]] > max_dist:            # too far from anything known
            return None
        votes = [self.y[i] for i in idx if d[i] <= max_dist]
        best = max(set(votes), key=votes.count)
        return best, votes.count(best) / len(votes)


# ---------------------------------------------------------------------------
# Hand object + tracker
# ---------------------------------------------------------------------------
class Hand:
    def __init__(self, pts, label):
        self.pts = pts                          # (21, 3) pixel coords
        self.label = label
        self.fingers = finger_states(pts)
        self.count = sum(self.fingers.values())
        self.state = open_closed(self.fingers)
        self.gesture = classify_gesture(pts, self.fingers)
        self.feat = features(pts, label)
        self.custom = None
        self.sign = None

    def lines(self):
        out = [f"Fingers: {self.count} | {self.state}"]
        if self.gesture:
            out.append(self.gesture)
        if self.custom:
            out.append(f"Custom: {self.custom[0]} {self.custom[1]:.0%}")
        if self.sign:
            out.append(f"Sign: {self.sign[0]} {self.sign[1]:.0%}")
        return out


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("Downloading hand_landmarker.task ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


class HandTracker:
    def __init__(self, max_hands=2):
        ensure_model()
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.lm = vision.HandLandmarker.create_from_options(opts)

    def detect(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        hands = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            pts = np.array([[l.x * w, l.y * h, l.z * w] for l in lms], dtype=np.float32)
            hands.append(Hand(pts, hd[0].category_name))
        return hands


def classify_hands(hands, knns):
    for h in hands:
        h.custom = knns["custom"].predict(h.feat)
        h.sign = knns["signs"].predict(h.feat)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def put_text(img, t, x, y, color=(255, 255, 255)):
    cv2.putText(img, t, (x, y), FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, t, (x, y), FONT, 0.55, color, 1, cv2.LINE_AA)


def draw_hands(frame, hands):
    for h in hands:
        pts = [(int(x), int(y)) for x, y, _ in h.pts]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 200, 0), 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 3, (0, 0, 255), -1)

        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x0, y0, x1, y1 = min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys) + 10
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 200, 0), 1)

        lines = h.lines()
        for i, t in enumerate(lines):
            ty = y0 - 8 - 18 * (len(lines) - 1 - i)
            if ty < 12:
                ty = y1 + 20 + 18 * i
            put_text(frame, t, x0, ty, (0, 255, 255))
    return frame