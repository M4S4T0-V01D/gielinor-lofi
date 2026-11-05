"""The 'record' and the arrangement timeline.

A Record is the original song pressed to audio: chosen tracks of the Jagex MIDI, meter-remapped to 4/4, rendered a
little fast and then varispeeded down, so pitch and tempo drop together the way a sampled record does. Chops are
taken from it by bar/beat and placed on a Timeline in song beats.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import notes as N
from .render import SR, fluid, varispeed

MELODIC_CH = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]


@dataclass
class Record:
    src: N.Source
    tracks: list[str]
    bpm: float                    # tempo after varispeed (the song tempo)
    semis: float = -2.0           # varispeed amount
    programs: dict[str, int] = field(default_factory=dict)   # program overrides per track
    vel: dict[str, float] = field(default_factory=dict)      # velocity scale per track
    bars: tuple[int, int] | None = None                      # source bar range to press (default: whole song)
    gain: float = 0.5
    swing: float = 0.5                                       # >0.5 swings the record's off-16ths into the groove
    dst_bar: int = 4                                         # song beats per source bar (2: a 3/4 bar as triplets)
    curve: object = N.WALTZ_TO_FOUR                          # meter curve for remap (None: proportional)
    audio: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        rbpm = self.bpm * 2 ** (-self.semis / 12)
        parts = []
        for t in self.tracks:
            ns = N.remap(self.src.tracks[t], self.src.beats_per_bar, self.dst_bar, self.curve)
            if self.bars:
                ns = N.window(ns, *self.bars, keep_tails=True)
            if t in self.vel:
                ns = N.scale_vel(ns, self.vel[t])
            if self.swing > 0.5:
                ns = N.swing(ns, self.swing)
            drum = t in self.src.drums
            parts.append((ns, 0 if drum else self.programs.get(t, self.src.programs.get(t, 0)),
                          128 if drum else 0, 9 if drum else MELODIC_CH[len(parts)]))
        raw = fluid(parts, rbpm, tail=6.0, gain=self.gain)
        self.audio = varispeed(raw, self.semis)
        self.beat_s = 60.0 / self.bpm * (len(self.audio) / len(raw)) / 2 ** (-self.semis / 12)

    def beats(self, b0: float, b1: float, tail_beats: float = 0.0) -> np.ndarray:
        """Audio from record beat b0 to b1 (4/4 beats from the start of the pressed range)."""
        i0 = int(round(b0 * self.beat_s * SR))
        i1 = int(round((b1 + tail_beats) * self.beat_s * SR))
        x = self.audio[max(0, i0):i1]
        want = i1 - i0
        if len(x) < want:
            x = np.pad(x, ((0, want - len(x)), (0, 0)))
        return x

    def bar(self, b0: float, b1: float | None = None, tail_beats: float = 0.0) -> np.ndarray:
        return self.beats(b0 * 4, (b1 if b1 is not None else b0 + 1) * 4, tail_beats)


def edge(x: np.ndarray, fin_ms=3.0, fout_ms=12.0) -> np.ndarray:
    """De-click a chop."""
    y = x.copy()
    a, b = int(fin_ms * SR / 1000), int(fout_ms * SR / 1000)
    if a and len(y) > a:
        y[:a] *= np.linspace(0, 1, a)[:, None]
    if b and len(y) > b:
        y[-b:] *= np.linspace(1, 0, b)[:, None]
    return y


def gate(x: np.ndarray, keep_s: float, release_ms=60.0) -> np.ndarray:
    """MPC-style note-off: keep the first keep_s seconds then fade."""
    k = int(keep_s * SR)
    r = int(release_ms * SR / 1000)
    y = x[:k + r].copy()
    if len(y) > k:
        y[k:] *= np.linspace(1, 0, len(y) - k)[:, None]
    return y


def reverse(x: np.ndarray) -> np.ndarray:
    return x[::-1].copy()


def repitch(x: np.ndarray, semis: float) -> np.ndarray:
    """Sampler pitch (changes length, like a pad played up or down)."""
    return varispeed(x, semis)


class Timeline:
    """Named buses of stereo audio addressed in song beats."""

    def __init__(self, bpm: float, beats: float, tail_s: float = 8.0):
        self.bpm = bpm
        self.n = int((beats * 60 / bpm + tail_s) * SR)
        self.bus: dict[str, np.ndarray] = {}
        self.kicks: list[float] = []      # seconds, for the sidechain

    def s(self, beat: float) -> float:
        return beat * 60.0 / self.bpm

    def i(self, beat: float) -> int:
        return int(round(self.s(beat) * SR))

    def get(self, name: str) -> np.ndarray:
        if name not in self.bus:
            self.bus[name] = np.zeros((self.n, 2), dtype=np.float32)
        return self.bus[name]

    def put(self, name: str, x: np.ndarray, beat: float, gain: float = 1.0, offset_ms: float = 0.0):
        b = self.get(name)
        i = self.i(beat) + int(offset_ms * SR / 1000)
        if i < 0:
            x, i = x[-i:], 0
        m = min(len(x), self.n - i)
        if m > 0:
            b[i:i + m] += x[:m] * gain

    def put_full(self, name: str, x: np.ndarray, gain: float = 1.0):
        """Add an audio file rendered from song beat 0 (a fluid() render of note parts)."""
        self.put(name, x, 0.0, gain)
