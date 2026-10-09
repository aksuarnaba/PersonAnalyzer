import os
import sys
import time
import argparse

import cv2
import numpy as np

from hands import HandTracker, GestureKNN, classify_hands, put_text
from filters import FilterManager, MODES
from switch import SignSwitch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(BASE, "models")

YUNET        = os.path.join(M, "face_detector", "face_detection_yunet_2023mar.onnx")
AGE_PROTO    = os.path.join(M, "age", "age_deploy.prototxt")
AGE_MODEL    = os.path.join(M, "age", "age_net.caffemodel")
GENDER_PROTO = os.path.join(M, "gender", "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(M, "gender", "gender_net.caffemodel")
EMOTION      = os.path.join(M, "emotion", "emotion-ferplus-8.onnx")

AGES     = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+"]
GENDERS  = ["Male", "Female"]
EMOTIONS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]
MEAN     = (78.4263377603, 87.7689143744, 114.895847746)

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CONTROL_SIGNS = ("A", "B")      # reserved: A = features on, B = features off


def need(path):
    if not os.path.exists(path):
        sys.exit(f"Missing model file: {path}")
    return path


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
detector    = cv2.FaceDetectorYN.create(need(YUNET), "", (320, 320), 0.7, 0.3, 5000)
age_net     = cv2.dnn.readNet(need(AGE_MODEL), need(AGE_PROTO))
gender_net  = cv2.dnn.readNet(need(GENDER_MODEL), need(GENDER_PROTO))
emotion_net = cv2.dnn.readNetFromONNX(need(EMOTION))

hand_tracker = None          # created in main() (needs to know image vs video)
knns = {"custom": GestureKNN("custom"), "signs": GestureKNN("signs")}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def detect(frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))          # must match every frame
    _, faces = detector.detect(frame)
    return [] if faces is None else faces


def crop(frame, x, y, w, h, pad=0.2):
    H, W = frame.shape[:2]
    px, py = int(w * pad), int(h * pad)
    return frame[max(0, y - py):min(H, y + h + py),
                 max(0, x - px):min(W, x + w + px)]


def age_gender(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MEAN, swapRB=False)
    gender_net.setInput(blob)
    g = gender_net.forward()[0]
    age_net.setInput(blob)
    a = age_net.forward()[0]
    return GENDERS[g.argmax()], float(g.max()), AGES[a.argmax()]


def emotion(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    blob = cv2.dnn.blobFromImage(gray, 1.0, (64, 64))   # raw 0-255, no scaling
    emotion_net.setInput(blob)
    s = emotion_net.forward()[0]
    p = np.exp(s - s.max())
    p /= p.sum()                                        # softmax
    return EMOTIONS[p.argmax()], float(p.max())


def analyze(frame):
    out = []
    for f in detect(frame):
        x, y, w, h = map(int, f[:4])
        face = crop(frame, x, y, w, h)
        if face.size == 0:
            continue
        gender, gp, age = age_gender(face)
        emo, ep = emotion(face)
        out.append(((x, y, w, h),
                    f"{gender} {gp:.0%}",
                    f"Age {age}",
                    f"{emo} {ep:.0%}"))
    return out


def analyze_hands(frame):
    """Returns (hands, control). The control signs A / B are never shown as letters."""
    hands = hand_tracker.detect(frame)
    classify_hands(hands, knns)
    control = None
    for h in hands:
        if h.sign and h.sign[0] in CONTROL_SIGNS:
            control = h.sign[0]
            h.sign = None
    return hands, control


def draw_status(frame, sw):
    """Top-right: features ON/OFF and the hold progress of an A / B sign."""
    W = frame.shape[1]

    def right(text, y, color):
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        put_text(frame, text, W - tw - 10, y, color)

    right(f"Features: {'ON' if sw.on else 'OFF'}  (sign A = on, B = off)", 20,
          (0, 255, 0) if sw.on else (0, 0, 255))
    if sw.candidate and sw.progress > 0:
        right(f"Holding {sw.candidate}... {sw.progress:.0%}", 42, (0, 255, 255))


def draw(frame, results):
    for (x, y, w, h), *lines in results:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for i, t in enumerate(lines):
            ty = y - 8 - 18 * (len(lines) - 1 - i)
            if ty < 12:
                ty = y + h + 16 + 18 * i        # flip below the box near the top edge
            cv2.putText(frame, t, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, t, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Camera opening
# ---------------------------------------------------------------------------
def open_capture(source):
    """
    source = "auto"        -> find DroidCam (or any camera) automatically
           = "1400"/"700"  -> camera index exactly as listed by cv2_enumerate_cameras
           = "0", "1" ...  -> plain camera index
           = path or URL   -> video file / network stream
    """
    if source != "auto" and not source.isdigit():
        return cv2.VideoCapture(source)                 # file or URL

    if source.isdigit():
        idx = int(source)
        if idx >= 100:
            # backend-encoded index: 700 = DirectShow cam 0, 1400 = MSMF cam 0
            return cv2.VideoCapture(idx % 100, (idx // 100) * 100)
        return cv2.VideoCapture(idx)

    # Auto mode: needs cv2-enumerate-cameras (uv add cv2-enumerate-cameras)
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except ImportError:
        return cv2.VideoCapture(0)

    cams = list(enumerate_cameras())
    # DroidCam first, then everything else
    cams.sort(key=lambda c: "droidcam" not in c.name.lower())
    for c in cams:
        print(f"Trying camera {c.index}: {c.name}")
        cap = cv2.VideoCapture(c.index)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"Using camera {c.index}: {c.name}")
                return cap
        cap.release()
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="700",
                    help="auto, camera index (700 = DroidCam), video file, or image")
    ap.add_argument("--every", type=int, default=3,
                    help="run the age/gender/emotion models every N frames")
    ap.add_argument("--record", metavar="LABEL", default=None,
                    help="record training samples for this gesture / letter")
    ap.add_argument("--set", choices=["custom", "signs"], default="custom",
                    help="dataset to record into (custom gestures or sign language)")
    ap.add_argument("--start", choices=["on", "off"], default="on",
                    help="start with the A/B-controlled features on or off")
    ap.add_argument("--filter", choices=MODES, default="portal",
                    help="starting hand filter (press F to cycle while running)")
    ap.add_argument("--no-smooth", action="store_true",
                    help="turn off landmark smoothing")
    args = ap.parse_args()

    global hand_tracker
    is_image = args.source.lower().endswith(IMAGE_EXT)
    hand_tracker = HandTracker(max_hands=2, video=not is_image,
                               smooth=not args.no_smooth)
    filters = FilterManager(args.filter)

    # Single image
    if is_image:
        img = cv2.imread(args.source)
        if img is None:
            sys.exit(f"Cannot read image: {args.source}")
        hands, _ = analyze_hands(img)
        img = filters.apply(img, hands)
        draw(img, analyze(img))
        cv2.imshow("Person Analyzer", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Webcam / video file / stream
    cap = open_capture(args.source)
    if (cap is None or not cap.isOpened()) and args.source != "auto":
        print(f"Cannot open source {args.source}, trying auto-detect...")
        cap = open_capture("auto")
    if cap is None or not cap.isOpened():
        sys.exit(f"Cannot open source: {args.source}")

    rec_knn = knns[args.set] if args.record else None
    recording = False

    # A / B sign switch (disabled while recording samples, since you would hold A / B)
    use_switch = rec_knn is None
    switch = SignSwitch("A", "B", start_on=(args.start == "on"))
    missing = [c for c in CONTROL_SIGNS if knns["signs"].count(c) == 0]
    if use_switch and missing:
        print(f"Note: no samples for sign(s) {missing} in data/signs.csv - the A/B "
              f"switch needs them. Import or record them first.")

    n, results, force = 0, [], True
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            hands, ctrl = analyze_hands(frame)     # hands every frame (fast)
            if use_switch and switch.update(ctrl, time.perf_counter()):
                print(f"Features {'ON' if switch.on else 'OFF'} (sign {ctrl})")
                force = switch.on
            on = switch.on if use_switch else True

            if on:
                if force or n % args.every == 0:   # face models (slow) every N frames
                    results, force = analyze(frame), False
            else:
                results = []                       # face / gender / age / emotion off
                for h in hands:
                    h.sign = None                  # sign language off

            # Record training samples from the first detected hand
            if rec_knn and recording and hands and n % 2 == 0:
                rec_knn.add(args.record, hands[0].feat)
            n += 1

            frame = filters.apply(frame, hands, enabled=on,
                                  allow_style=ctrl is None)
            draw(frame, results)                   # face boxes on top
            if use_switch:
                draw_status(frame, switch)

            if rec_knn:
                state = "REC" if recording else "paused"
                put_text(frame, f"[{args.set}] '{args.record}': "
                                f"{rec_knn.count(args.record)} samples - {state} "
                                f"(SPACE = start/stop, Q = save & quit)",
                         10, frame.shape[0] - 12,
                         (0, 0, 255) if recording else (255, 255, 255))

            cv2.imshow("Person Analyzer", frame)
            key = cv2.waitKey(1) & 0xFF
            k = chr(key).lower() if key < 128 else ""
            if k == "q" or key == 27:
                break
            if key == 32:
                recording = not recording
            elif k == "f":
                filters.next()
            elif k == "c":
                filters.clear()
            elif k == "n":
                filters.next_portal()
    finally:
        if rec_knn:
            rec_knn.save()
            print(f"Saved. '{args.record}' now has {rec_knn.count(args.record)} samples "
                  f"in {rec_knn.path}")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()