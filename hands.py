"""
hands.py - hand landmarks, finger counting, built-in gestures,
custom gestures and static sign language (all landmark-based).
"""
import os
import csv
import time
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

# Smoothing strength (One Euro filter). Lower min_cutoff = smoother when still,
# higher beta = less lag when the hand moves fast.
SMOOTH_MIN_CUTOFF = 1.2
SMOOTH_BETA = 0.02


class OneEuro:
    """One Euro filter: strong smoothing when still, low lag when moving."""

    def __init__(self, min_cutoff=SMOOTH_MIN_CUTOFF, beta=SMOOTH_BETA, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(dt, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = self._alpha(dt, cutoff)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


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
    def __init__(self, pts, label, id=0):
        self.pts = pts                          # (21, 3) pixel coords
        self.label = label
        self.id = id                            # stable id while the hand stays visible
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
    """
    video=False -> IMAGE mode (single photos, dataset import)
    video=True  -> VIDEO mode: MediaPipe tracks between frames (faster, steadier)
    smooth=True -> additionally smooth landmarks with a One Euro filter (video only)
    """

    def __init__(self, max_hands=2, video=False, smooth=False):
        ensure_model()
        self.video = video
        self.smooth = smooth and video
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mode,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.lm = vision.HandLandmarker.create_from_options(opts)
        self.tracks = {}        # id -> {"wrist", "filter", "miss"}
        self.next_id = 0
        self.last_ts = -1

    def _match(self, wrists, width):
        """Give each detected hand the id of the nearest hand from the last frame."""
        limit = 0.3 * width
        ids, used = [], set()
        for wr in wrists:
            best, bd = None, limit
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                d = float(np.linalg.norm(wr - tr["wrist"]))
                if d < bd:
                    best, bd = tid, d
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.tracks[best] = {"wrist": wr.copy(), "filter": OneEuro(), "miss": 0}
            used.add(best)
            self.tracks[best]["miss"] = 0
            ids.append(best)
        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid]["miss"] += 1
                if self.tracks[tid]["miss"] > 8:     # forget hands gone for ~8 frames
                    del self.tracks[tid]
        return ids

    def detect(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.video:
            ts = int(time.perf_counter() * 1000)
            if ts <= self.last_ts:                   # timestamps must increase
                ts = self.last_ts + 1
            self.last_ts = ts
            res = self.lm.detect_for_video(img, ts)
        else:
            res = self.lm.detect(img)

        raw = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            pts = np.array([[l.x * w, l.y * h, l.z * w] for l in lms], dtype=np.float32)
            raw.append((pts, hd[0].category_name))

        if not self.video:
            return [Hand(p, lab, i) for i, (p, lab) in enumerate(raw)]

        ids = self._match([p[0, :2] for p, _ in raw], w)
        now = time.perf_counter()
        hands = []
        for (pts, label), tid in zip(raw, ids):
            tr = self.tracks[tid]
            if self.smooth:
                pts = tr["filter"](pts, now)
            tr["wrist"] = pts[0, :2].copy()
            hands.append(Hand(pts, label, tid))
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


def draw_skeleton(frame, hands):
    for h in hands:
        pts = [(int(x), int(y)) for x, y, _ in h.pts]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 200, 0), 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 3, (0, 0, 255), -1)
    return frame


def draw_labels(frame, hands):
    for h in hands:
        pts = [(int(x), int(y)) for x, y, _ in h.pts]
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


def draw_hands(frame, hands):
    return draw_labels(draw_skeleton(frame, hands), hands)