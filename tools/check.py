"""Tuning/key check on rendered stems: chroma per stem, cents offset of spectral peaks, sample-vs-Rhodes agreement.

usage: check.py <stems_dir> <bpm> <bar0> <bar1> [tonic] [minor|dorian|major]   (default G minor)
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, stft

NAMES = "C C# D D# E F F# G G# A A# B".split()
MODES = {"minor": (0, 2, 3, 5, 7, 8, 10), "dorian": (0, 2, 3, 5, 7, 9, 10), "major": (0, 2, 4, 5, 7, 9, 11)}


def analyse(x, sr):
    f, _, Z = stft(x.mean(axis=1), sr, nperseg=16384, noverlap=8192)
    S = np.abs(Z).mean(axis=1)
    band = (f > 90) & (f < 2500)
    pk, _ = find_peaks(S * band, height=S[band].max() * 0.08, distance=3)
    cents, chroma = [], np.zeros(12)
    for i in pk:
        if 0 < i < len(S) - 1:              # parabolic interpolation for the true peak frequency
            a, b, c = np.log(S[i - 1:i + 2] + 1e-12)
            fi = f[i] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0])
            midi = 69 + 12 * np.log2(fi / 440)
            cents.append((midi - round(midi)) * 100)
            chroma[int(round(midi)) % 12] += S[i]
    return chroma / (chroma.sum() + 1e-12), np.array(cents), S[pk]


def main():
    d, bpm, b0, b1 = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    bs = 4 * 60 / bpm
    res = {}
    for name in ("sample", "rhodes", "bass", "oboe", "bell", "strings", "lead"):
        p = d / f"{name}.wav"
        if not p.exists():
            continue
        x, sr = sf.read(str(p))
        seg = x[int(b0 * bs * sr):int(b1 * bs * sr)]
        if np.sqrt(np.mean(seg ** 2)) < 1e-5:
            continue
        ch, cents, w = analyse(seg, sr)
        res[name] = ch
        top = ", ".join(f"{NAMES[i]} {ch[i]:.2f}" for i in np.argsort(ch)[::-1][:5])
        wc = np.average(cents, weights=w) if len(cents) else float("nan")
        print(f"{name:7s} top: {top:40s} weighted cents offset {wc:+.1f} (n={len(cents)})")
    if "sample" in res and "rhodes" in res:
        print(f"sample/rhodes chroma correlation {np.corrcoef(res['sample'], res['rhodes'])[0, 1]:+.2f}")
    tonic = sys.argv[5] if len(sys.argv) > 5 else "G"
    mode = sys.argv[6] if len(sys.argv) > 6 else "minor"
    scale = {(NAMES.index(tonic) + i) % 12 for i in MODES[mode]}
    for k, ch in res.items():
        print(f"{k:7s} energy in {tonic} {mode} scale {sum(ch[i] for i in scale):.2f}")


if __name__ == "__main__":
    main()
