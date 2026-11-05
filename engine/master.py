"""Mixdown helpers, loudness mastering and analysis."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import resample_poly

from . import fx
from .render import SR


def true_peak_db(x: np.ndarray) -> float:
    up = resample_poly(x, 4, 1, axis=0)
    return float(20 * np.log10(np.max(np.abs(up)) + 1e-12))


def lufs(x: np.ndarray) -> float:
    return float(pyln.Meter(SR).integrated_loudness(x.astype(np.float64)))


def master(mix: np.ndarray, target_lufs=-14.0, ceiling_dbtp=-1.0, glue=True) -> np.ndarray:
    """Tape glue, a gentle bus compressor, then gain to target and a lookahead limiter with a true-peak check."""
    x = fx.highpass(mix, 28, 2)
    x = fx.mono_below(x, 140)
    if glue:
        x = fx.compress(x, threshold_db=-16, ratio=2.0, attack_ms=25, release_ms=250, knee_db=8)
    x = fx.tape(x * 0.9, drive=1.15, hf=15500) / 0.9
    for _ in range(4):
        g = target_lufs - lufs(x)
        x = x * fx.db(g)
        x = fx.limiter(x, ceiling_db=ceiling_dbtp - 0.6, release_ms=120, lookahead_ms=5)
        if abs(lufs(x) - target_lufs) < 0.15:
            break
    tp = true_peak_db(x)
    if tp > ceiling_dbtp:
        x = x * fx.db(ceiling_dbtp - tp - 0.05)
    return x.astype(np.float32)


def report(x: np.ndarray, bpm: float | None = None, sections=None) -> dict:
    r = {"seconds": round(len(x) / SR, 2), "lufs": round(lufs(x), 2), "true_peak_dbtp": round(true_peak_db(x), 2)}
    mono = x.mean(axis=1)
    spec = np.abs(np.fft.rfft(mono[: SR * 60]))
    f = np.fft.rfftfreq(min(len(mono), SR * 60), 1 / SR)
    tot = np.sum(spec ** 2)
    for lo, hi in ((20, 120), (120, 500), (500, 2000), (2000, 6000), (6000, 20000)):
        r[f"band_{lo}_{hi}"] = round(float(10 * np.log10(np.sum(spec[(f >= lo) & (f < hi)] ** 2) / tot)), 1)
    if sections and bpm:
        bs = 60 / bpm
        r["sections_rms_db"] = {name: round(float(20 * np.log10(np.sqrt(np.mean(
            x[int(a * 4 * bs * SR):int(b * 4 * bs * SR)] ** 2)) + 1e-9)), 1) for name, a, b in sections}
    return r


def write(path: Path, x: np.ndarray, mp3=True, tags=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, SR, subtype="PCM_24")
    if mp3:
        meta = sum((["-metadata", f"{k}={v}"] for k, v in (tags or {}).items()), [])
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-codec:a", "libmp3lame", "-b:a", "320k",
                        *meta, str(path.with_suffix(".mp3"))], check=True)
