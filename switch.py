"""
switch.py - turn features ON / OFF by holding a hand sign (default: A = on, B = off).

The sign must be held for `hold` seconds, and fires once per hold, so a quick
flicker of the classifier or an accidental pose does not toggle anything.
"""


class SignSwitch:
    def __init__(self, on_label="A", off_label="B", hold=0.7, grace=0.25, start_on=True):
        self.on_label, self.off_label = on_label, off_label
        self.hold, self.grace = hold, grace      # seconds to hold / dropout tolerance
        self.on = start_on
        self.candidate = None                    # sign currently being held
        self.progress = 0.0                      # 0..1 hold progress
        self._t0 = 0.0
        self._last_seen = 0.0
        self._fired = False

    def update(self, label, now):
        """label = 'A', 'B' or anything else / None. Returns True if state changed."""
        if label in (self.on_label, self.off_label):
            if label != self.candidate:
                self.candidate, self._t0, self._fired = label, now, False
            self._last_seen = now
        elif self.candidate and now - self._last_seen > self.grace:
            self.candidate, self._fired = None, False   # released: re-arm

        self.progress = 0.0
        if self.candidate and not self._fired:
            self.progress = min(1.0, (now - self._t0) / self.hold)
            if self.progress >= 1.0:
                self._fired = True
                new_state = self.candidate == self.on_label
                if new_state != self.on:
                    self.on = new_state
                    return True
        return False