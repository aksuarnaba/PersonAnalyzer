"""
filters.py - visual filters that follow your hands.
Press F in the app to cycle through them, C to clear the trails.
"""
import random
import time
from collections import deque

import cv2
import numpy as np

from hands import CONNECTIONS, draw_hands, draw_labels, put_text

MODES = ["portal", "skeleton", "neon", "trails", "pixelate", "spotlight"]

# fingertip landmark -> BGR colour
TIPS = {4: (0, 0, 255), 8: (0, 165, 255), 12: (0, 255, 255),
        16: (0, 255, 0), 20: (255, 128, 0)}
NEON = [(255, 0, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0)]

# colours for the style panel (BGR)
PALETTE = {"Pink": (180, 105, 255), "Yellow": (0, 255, 255),
           "Green": (0, 230, 0), "Blue": (255, 80, 0)}


# ---------------------------------------------------------------------------
# Portal effects: each takes the BGR region inside the portal and returns a
# restyled BGR image of the same size.
# ---------------------------------------------------------------------------
SEPIA = np.array([[0.131, 0.534, 0.272],
                  [0.168, 0.686, 0.349],
                  [0.189, 0.769, 0.393]], np.float32)


def fx_thermal(roi, mgr):
    return cv2.applyColorMap(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)


def fx_purple(roi, mgr):
    solid = np.full_like(roi, (160, 30, 110))
    return cv2.addWeighted(roi, 0.3, solid, 0.7, 0)


def fx_negative(roi, mgr):
    return cv2.bitwise_not(roi)


def fx_sepia(roi, mgr):
    return cv2.transform(roi, SEPIA)


def fx_pixelate(roi, mgr, block=16):
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def fx_grayscale(roi, mgr):
    return cv2.cvtColor(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def fx_sketch(roi, mgr):
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(255 - g, (21, 21), 0)
    return cv2.cvtColor(cv2.divide(g, 255 - blur, scale=256), cv2.COLOR_GRAY2BGR)


def fx_posterize(roi, mgr):
    gray = cv2.equalizeHist(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
    level = np.clip(gray // 64, 0, 3)                  # 4 brightness bands
    lut = np.array([PALETTE[n] for n in mgr.order], np.uint8)
    return lut[level]


PORTAL_EFFECTS = [("Thermal", fx_thermal), ("Purple", fx_purple),
                  ("Negative", fx_negative), ("Sepia", fx_sepia),
                  ("Pixelate", fx_pixelate), ("Grayscale", fx_grayscale),
                  ("Sketch", fx_sketch), ("Posterize", fx_posterize)]


def hand_mask(shape, hands):
    """Filled, slightly grown convex hull around every hand (uint8 0/255)."""
    mask = np.zeros(shape[:2], np.uint8)
    for h in hands:
        pts = h.pts[:, :2].astype(np.int32)
        hull = cv2.convexHull(pts)
        size = int(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
        cv2.fillConvexPoly(mask, hull, 255)
        cv2.polylines(mask, [hull], True, 255, max(2, int(size * 0.2)))
    return mask


class FilterManager:
    def __init__(self, start="portal"):
        self.i = MODES.index(start)
        self.trails = {}                     # (hand id, tip) -> deque of points
        self.fx = 0                          # index into PORTAL_EFFECTS
        self.order = ["Pink", "Yellow", "Green", "Blue"]   # Posterize colour order
        self.closed_prev = set()             # hand ids that were closed last frame
        self.last_change = 0.0

    @property
    def mode(self):
        return MODES[self.i]

    def next(self):
        self.i = (self.i + 1) % len(MODES)
        self.trails.clear()

    def clear(self):
        self.trails.clear()

    # ------------------------------------------------------------------
    def apply(self, frame, hands, enabled=True, allow_style=True):
        """
        enabled=False     -> only the plain skeleton (no image effects)
        allow_style=False -> closing a palm will not switch the portal filter
        """
        m = self.mode if enabled else "skeleton"
        if m == "portal":
            self._portal(frame, hands, allow_style)
            draw_hands(frame, hands)
        elif m == "skeleton":
            draw_hands(frame, hands)
        elif m == "neon":
            frame = self._neon(frame, hands)
            draw_labels(frame, hands)
        elif m == "trails":
            self._trails(frame, hands)
            draw_hands(frame, hands)
        elif m == "pixelate":
            frame = self._pixelate(frame, hands)
            draw_labels(frame, hands)
        elif m == "spotlight":
            frame = self._spotlight(frame, hands)
            draw_labels(frame, hands)
        label = self.mode if enabled else "off"
        put_text(frame, f"Filter: {label}   (F = next, C = clear)", 10, 20)
        return frame

    # ------------------------------------------------------------------
    def next_portal(self):
        self.fx = (self.fx + 1) % len(PORTAL_EFFECTS)
        name = PORTAL_EFFECTS[self.fx][0]
        if name == "Posterize":                  # new random colour order each time
            old = list(self.order)
            while self.order == old:
                random.shuffle(self.order)
        print(f"Portal filter: {name}")

    def _portal(self, frame, hands, allow_style):
        """Portal stretched between both hands (thumb + index tips = corners)."""
        H, W = frame.shape[:2]
        now = time.perf_counter()

        # closing a palm switches to the next filter (once per close)
        closed = {h.id for h in hands if h.state == "Closed"}
        newly = closed - self.closed_prev
        self.closed_prev = closed
        if newly and allow_style and now - self.last_change > 0.8:
            self.next_portal()
            self.last_change = now

        name, fx = PORTAL_EFFECTS[self.fx]
        title = f"Portal filter: {name} ({self.fx + 1}/{len(PORTAL_EFFECTS)})"
        if name == "Posterize":
            title += "  " + " > ".join(self.order)
        put_text(frame, title, 10, 48)
        put_text(frame, "Close palm for the next filter", 10, 70)
        put_text(frame, "Use both hands to hold the portal", 10, 92)

        if len(hands) < 2:
            return

        pts = np.array([h.pts[i][:2] for h in hands[:2] for i in (4, 8)], np.float32)
        c = pts.mean(0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        quad = pts[np.argsort(ang)].astype(np.int32)     # corners in drawing order

        x, y, w, h = cv2.boundingRect(quad)
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        roi = frame[y0:y1, x0:x1]                        # view into the frame
        styled = fx(roi, self)

        mask = np.zeros(roi.shape[:2], np.uint8)
        cv2.fillPoly(mask, [(quad - np.array([x0, y0])).astype(np.int32)], 255)
        roi[mask > 0] = styled[mask > 0]
        cv2.polylines(frame, [quad], True, (235, 235, 235), 1, cv2.LINE_AA)

    def _neon(self, frame, hands):
        if not hands:
            return frame
        layer = np.zeros_like(frame)
        for h in hands:
            pts = [(int(x), int(y)) for x, y, _ in h.pts]
            color = NEON[h.id % len(NEON)]
            for a, b in CONNECTIONS:
                cv2.line(layer, pts[a], pts[b], color, 5, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(layer, pt, 6, (255, 255, 255), -1, cv2.LINE_AA)
        glow = cv2.GaussianBlur(layer, (0, 0), 12)
        out = cv2.addWeighted(frame, 0.55, glow, 1.2, 0)   # dim scene + glow
        return cv2.add(out, layer)

    def _trails(self, frame, hands):
        seen = set()
        for h in hands:
            for tip in TIPS:
                key = (h.id, tip)
                dq = self.trails.setdefault(key, deque(maxlen=30))
                dq.append((int(h.pts[tip][0]), int(h.pts[tip][1])))
                seen.add(key)
        for key in list(self.trails):            # trails of lost hands fade out
            if key not in seen:
                self.trails[key].popleft()
                if not self.trails[key]:
                    del self.trails[key]
        for (_, tip), dq in self.trails.items():
            pts, n = list(dq), len(dq)
            for j in range(1, n):
                cv2.line(frame, pts[j - 1], pts[j], TIPS[tip],
                         1 + int(5 * j / n), cv2.LINE_AA)

    def _pixelate(self, frame, hands, block=14):
        if not hands:
            return frame
        H, W = frame.shape[:2]
        small = cv2.resize(frame, (max(1, W // block), max(1, H // block)),
                           interpolation=cv2.INTER_AREA)
        pix = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
        m = hand_mask(frame.shape, hands) > 0
        frame[m] = pix[m]
        return frame

    def _spotlight(self, frame, hands):
        if not hands:
            return frame
        m = cv2.GaussianBlur(hand_mask(frame.shape, hands), (0, 0), 15)
        m = (m.astype(np.float32) / 255.0)[..., None]
        out = frame * m + frame * 0.2 * (1 - m)
        return out.astype(np.uint8)