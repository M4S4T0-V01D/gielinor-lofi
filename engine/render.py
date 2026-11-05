"""Render note parts to audio with fluidsynth and the GeneralUser GS soundfont. Results are cached by content."""
from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import notes as N

ROOT = Path(__file__).resolve().parent.parent
SF2 = ROOT / "assets" / "sf2" / "GeneralUser-GS.sf2"
CACHE = ROOT / "build" / "cache"
SR = 44100


def fluid(parts, bpm: float, tail: float = 4.0, gain: float = 0.5) -> np.ndarray:
    """parts: [(notes, program, bank, channel)] -> stereo float32 (n, 2). fluidsynth's own reverb/chorus are off;
    space comes from engine.fx."""
    key = hashlib.sha1(json.dumps(
        [[[(round(n.start, 5), round(n.dur, 5), n.pitch, n.vel) for n in ns], p, b, c] for ns, p, b, c in parts]
        + [bpm, tail, gain, SR, SF2.name], sort_keys=True).encode()).hexdigest()[:20]
    CACHE.mkdir(parents=True, exist_ok=True)
    wav = CACHE / f"{key}.wav"
    if not wav.exists():
        mid = CACHE / f"{key}.mid"
        N.write_midi(mid, parts, bpm)
        tmp = wav.with_suffix(".tmp.wav")
        subprocess.run(["fluidsynth", "-ni", "-q", "-g", str(gain), "-r", str(SR), "-F", str(tmp),
                        "-o", "synth.reverb.active=0", "-o", "synth.chorus.active=0",
                        "-o", "synth.polyphony=512", str(SF2), str(mid)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tmp.rename(wav)
        mid.unlink(missing_ok=True)
    x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    assert sr == SR
    end = max((n.end for ns, *_ in parts for n in ns), default=0.0)
    want = int((end * 60.0 / bpm + tail) * SR)
    if len(x) < want:
        x = np.pad(x, ((0, want - len(x)), (0, 0)))
    return x[:want]


def varispeed(x: np.ndarray, semis: float) -> np.ndarray:
    """Turntable/sampler pitch change: shifts pitch and tempo together (negative = slower and lower)."""
    if abs(semis) < 1e-6:
        return x
    r = Fraction(2 ** (-semis / 12)).limit_denominator(2000)   # length factor
    return resample_poly(x, r.numerator, r.denominator, axis=0).astype(np.float32)


def load_wav(path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    if sr != SR:
        r = Fraction(SR, sr).limit_denominator(2000)
        x = resample_poly(x, r.numerator, r.denominator, axis=0).astype(np.float32)
    return x
