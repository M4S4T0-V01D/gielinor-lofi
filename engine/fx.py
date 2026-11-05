"""Lo-fi processing on stereo float arrays (n, 2) at engine.render.SR."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, lfilter, oaconvolve, sosfilt, sosfiltfilt

from .render import SR


def db(x: float) -> float:
    return 10 ** (x / 20)


def _sos(kind, f, order=2, sr=SR):
    if kind == "band":
        return butter(order, [f[0] / (sr / 2), f[1] / (sr / 2)], "bandpass", output="sos")
    return butter(order, min(f / (sr / 2), 0.99), kind, output="sos")


def lowpass(x, f, order=2):
    return sosfilt(_sos("low", f, order), x, axis=0).astype(np.float32)


def highpass(x, f, order=2):
    return sosfilt(_sos("high", f, order), x, axis=0).astype(np.float32)


def bandpass(x, lo, hi, order=2):
    return sosfilt(_sos("band", (lo, hi), order), x, axis=0).astype(np.float32)


def sweep_lowpass(x, f_of_t, block=512):
    """Time-varying one-pole x2 lowpass; f_of_t(t_seconds array) -> cutoff Hz. Smooth enough for filter rides."""
    n = len(x)
    t = np.arange(0, n, block) / SR
    fc = np.clip(f_of_t(t), 20, SR / 2.2)
    out = np.empty_like(x)
    z1 = np.zeros(x.shape[1]); z2 = np.zeros(x.shape[1])
    for i, f in enumerate(fc):
        a = np.exp(-2 * np.pi * f / SR)
        seg = x[i * block:(i + 1) * block]
        y1, z1 = lfilter([1 - a], [1, -a], seg, axis=0, zi=z1[None, :] if z1.ndim == 1 else z1)
        y2, z2 = lfilter([1 - a], [1, -a], y1, axis=0, zi=z2[None, :] if z2.ndim == 1 else z2)
        out[i * block:(i + 1) * block] = y2
    return out.astype(np.float32)


def shelf_tilt(x, pivot=800.0, db_per_oct=-1.0):
    """Gentle tilt: blend of low and high bands around a pivot (negative = darker)."""
    lo = lowpass(x, pivot, 1)
    hi = x - lo
    g = db(db_per_oct * 3)
    return (lo / np.sqrt(g) + hi * np.sqrt(g)).astype(np.float32)


def saturate(x, drive=1.5, mix=1.0):
    y = np.tanh(x * drive) / np.tanh(drive)
    return (x * (1 - mix) + y * mix).astype(np.float32)


def tape(x, drive=1.3, hf=12000.0):
    """Asymmetric soft clip (even harmonics) and head-bump/HF roll-off."""
    y = np.tanh(drive * x + 0.08 * x * x) / np.tanh(drive)
    y = y - np.mean(y, axis=0)
    low = bandpass(y, 60, 140, 1) * 0.25          # head bump
    return lowpass(y + low, hf, 2)


def bitcrush(x, bits=12, rate=26040.0, mix=1.0):
    """Sampler-style: sample-and-hold to rate, quantize to bits (SP-1200 is 12-bit at 26.04 kHz)."""
    n = len(x)
    idx = np.floor(np.arange(n) * rate / SR) * (SR / rate)
    idx = np.minimum(idx.astype(np.int64), n - 1)
    y = x[idx]
    q = 2 ** (bits - 1)
    y = np.round(y * q) / q
    y = lowpass(y, rate * 0.45, 4)                 # the converter's reconstruction filter
    return (x * (1 - mix) + y * mix).astype(np.float32)


def wow_flutter(x, wow_ms=1.6, wow_hz=0.55, flutter_ms=0.09, flutter_hz=6.3, drift_ms=1.2, seed=1):
    """Tape/turntable speed wobble as a modulated delay (read position moves around the write position)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    t = np.arange(n) / SR
    k = max(2, int(t[-1] * 0.7) + 2)
    drift = np.interp(t, np.linspace(0, t[-1], k), rng.standard_normal(k))
    d = (wow_ms * np.sin(2 * np.pi * wow_hz * t + rng.uniform(0, 6.28))
         + flutter_ms * np.sin(2 * np.pi * flutter_hz * t)
         + drift_ms * drift) / 1000.0
    pos = np.clip((t - d - (wow_ms + flutter_ms + 3 * drift_ms) / 1000.0) * SR, 0, n - 1)
    i = np.floor(pos).astype(np.int64); f = (pos - i)[:, None]
    j = np.minimum(i + 1, n - 1)
    return (x[i] * (1 - f) + x[j] * f).astype(np.float32)


def detune_chorus(x, cents=6.0, rate=0.23, mix=0.35):
    """One slow chorus voice each side, a few cents off: the 'slightly out of tune' tape sheen."""
    n = len(x)
    t = np.arange(n) / SR
    out = x.copy()
    for ch, ph in ((0, 0.0), (1, 2.1)):
        depth = (2 ** (cents / 1200) - 1) / (2 * np.pi * rate)            # seconds of delay swing for +-cents
        d = 0.012 + depth * np.sin(2 * np.pi * rate * t + ph)
        pos = np.clip((t - d) * SR, 0, n - 1)
        i = pos.astype(np.int64); f = pos - i; j = np.minimum(i + 1, n - 1)
        out[:, ch] = x[:, ch] * (1 - mix) + (x[i, ch] * (1 - f) + x[j, ch] * f) * mix
    return out.astype(np.float32)


def sway(x, depth_ms, hz):
    """A slow pitch lean, like a deck rolling: a variable delay of up to depth_ms[i] (per-sample array, 0 = dry)
    following (1 - cos) at hz, so it starts and ends in time with the dry signal."""
    n = len(x)
    t = np.arange(n) / SR
    d = np.asarray(depth_ms) * 1e-3 * SR * (1 - np.cos(2 * np.pi * hz * t)) / 2
    pos = np.clip(np.arange(n) - d, 0, n - 1)
    i = pos.astype(np.int64)
    f = (pos - i)[:, None]
    return (x[i] * (1 - f) + x[np.minimum(i + 1, n - 1)] * f).astype(np.float32)


def phaser(x, rate=0.25, lo=300.0, hi=2400.0, stages=4, feedback=0.35, mix=0.5, block=256):
    """A swept allpass phaser: `stages` first-order allpasses whose corner glides between lo and hi (log) at
    `rate` Hz, the two channels a quarter cycle apart. Coefficients change per block; the filter state carries."""
    from scipy.signal import lfilter
    n = len(x)
    out = np.empty_like(x)
    for ch, ph in ((0, 0.0), (1, np.pi / 2)):
        zi = [np.zeros(1) for _ in range(stages)]
        fb = 0.0
        y = np.empty(n, dtype=np.float64)
        for i0 in range(0, n, block):
            seg = x[i0:i0 + block, ch].astype(np.float64)
            t = (i0 + block / 2) / SR
            f = lo * (hi / lo) ** (0.5 - 0.5 * np.cos(2 * np.pi * rate * t + ph))
            w = np.tan(np.pi * f / SR)
            a = (w - 1) / (w + 1)
            s = seg + fb * feedback
            for k in range(stages):
                s, zi[k] = lfilter([a, 1.0], [1.0, a], s, zi=zi[k])
            fb = s[-1]
            y[i0:i0 + len(seg)] = s
        out[:, ch] = x[:, ch] * (1 - mix * 0.5) + y * mix * 0.5
    return out.astype(np.float32)


def autopan(x, rate, depth=0.6, phase=0.0):
    """A slow constant-power pan LFO (rate in Hz; depth 1 = hard left to hard right)."""
    t = np.arange(len(x)) / SR
    p = depth * np.sin(2 * np.pi * rate * t + phase)
    a = (p + 1) * np.pi / 4
    m = x.mean(axis=1)
    return (np.stack([m * np.cos(a), m * np.sin(a)], axis=1) * np.sqrt(2)).astype(np.float32)


def small_speaker(x, lo=480.0, hi=3400.0, drive=1.8):
    """A tiny speaker in the room: mono, no lows or highs, a cone resonance near 1.4 kHz, a little breakup."""
    m = x.mean(axis=1, keepdims=True).repeat(2, axis=1)
    y = bandpass(m, lo, hi, 2)
    y = y + bandpass(y, 1100, 1800, 2) * 0.8
    return saturate(y, drive)


def widen(x, amount=1.2):
    m = (x[:, 0] + x[:, 1]) / 2
    s = (x[:, 0] - x[:, 1]) / 2 * amount
    return np.stack([m + s, m - s], axis=1).astype(np.float32)


def mono_below(x, f=150.0):
    lo = lowpass(x, f, 2)
    m = lo.mean(axis=1, keepdims=True)
    return (x - lo + m).astype(np.float32)


def pan(x, p):
    """p in [-1, 1], constant power."""
    a = (p + 1) * np.pi / 4
    m = x.mean(axis=1) if x.ndim == 2 else x
    return np.stack([m * np.cos(a), m * np.sin(a)], axis=1).astype(np.float32) * np.sqrt(2)


# --- space -------------------------------------------------------------------------------------------------------

_IR = {}


def _ir(seconds, damp_hz, predelay_ms, seed):
    key = (seconds, damp_hz, predelay_ms, seed)
    if key not in _IR:
        rng = np.random.default_rng(seed)
        n = int(seconds * 1.4 * SR)
        t = np.arange(n) / SR
        env = np.exp(-6.9 * t / seconds)                          # -60 dB at `seconds`
        noise = rng.standard_normal((n, 2))
        # darken the tail over time: split bright and dark noise, bright decays faster
        dark = lowpass(noise, damp_hz, 2)
        bright = noise - dark
        ir = dark * env[:, None] + bright * (env ** 2.2)[:, None]
        early = np.zeros((int(0.08 * SR), 2))
        for k in range(10):
            early[rng.integers(0, len(early)), k % 2] += rng.uniform(0.3, 0.8) * (1 - k / 12)
        ir[:len(early)] += early * 0.6
        fade = np.minimum(1, np.arange(n) / (0.004 * SR))[:, None]
        ir = ir * fade
        ir = np.pad(ir, ((int(predelay_ms * SR / 1000), 0), (0, 0)))
        ir /= np.sqrt(np.sum(ir ** 2) / 2)
        _IR[key] = ir.astype(np.float32)
    return _IR[key]


def reverb(x, seconds=2.2, damp_hz=3500.0, predelay_ms=25.0, hp=250.0, seed=7):
    """Wet signal only (mix it yourself). Stereo-decorrelated noise IR with a darkening tail."""
    ir = _ir(seconds, damp_hz, predelay_ms, seed)
    src = highpass(x, hp, 2)
    wet = np.stack([oaconvolve(src[:, c], ir[:, c], mode="full")[:len(x)] for c in range(2)], axis=1)
    return (wet * 0.35).astype(np.float32)


def tape_delay(x, beats, bpm, feedback=0.4, repeats=5, lp=2800.0, hp=300.0, pingpong=True):
    """Wet only. Each repeat is darker; alternates sides if pingpong."""
    d = int(round(beats * 60 / bpm * SR))
    wet = np.zeros_like(x)
    tap = highpass(lowpass(x, lp, 2), hp, 1)
    for k in range(1, repeats + 1):
        if k > 1:
            tap = lowpass(tap, lp * 0.85, 1)
        g = feedback ** (k - 1)
        sig = tap * g
        if pingpong:
            sig = sig[:, ::-1] if k % 2 else sig
        if k * d < len(x):
            wet[k * d:] += sig[:len(x) - k * d]
    return wet.astype(np.float32)


# --- dynamics ----------------------------------------------------------------------------------------------------

def envelope(x, attack_ms=10.0, release_ms=120.0):
    """Peak envelope (mono) with separate attack/release via a max-hold then smoothing."""
    a = np.max(np.abs(x), axis=1) if x.ndim == 2 else np.abs(x)
    hold = maximum_filter1d(a, size=max(1, int(attack_ms * SR / 1000)), origin=0)
    r = np.exp(-1.0 / (release_ms * SR / 1000))
    return lfilter([1 - r], [1, -r], hold).astype(np.float32)


def compress(x, threshold_db=-18.0, ratio=3.0, attack_ms=10.0, release_ms=150.0, makeup_db=0.0, knee_db=6.0):
    env = envelope(x, attack_ms, release_ms)
    lvl = 20 * np.log10(np.maximum(env, 1e-7))
    over = lvl - threshold_db
    gr = np.where(over <= -knee_db / 2, 0.0,
                  np.where(over >= knee_db / 2, over * (1 - 1 / ratio),
                           (1 - 1 / ratio) * (over + knee_db / 2) ** 2 / (2 * knee_db)))
    g = db(-gr + makeup_db)
    return (x * g[:, None]).astype(np.float32)


def duck(n, hits_s, depth_db=3.0, attack_ms=8.0, release_ms=220.0):
    """Sidechain gain curve from known kick times (no detection, so it never mis-triggers)."""
    g = np.zeros(n, dtype=np.float32)
    t = np.arange(int((attack_ms + release_ms * 5) * SR / 1000)) / SR
    shape = np.where(t < attack_ms / 1000, t / (attack_ms / 1000), np.exp(-(t - attack_ms / 1000) / (release_ms / 1000)))
    for h in hits_s:
        i = int(h * SR)
        if 0 <= i < n:
            m = min(len(shape), n - i)
            g[i:i + m] = np.maximum(g[i:i + m], shape[:m])
    return (1 - (1 - db(-depth_db)) * g).astype(np.float32)


def limiter(x, ceiling_db=-1.0, release_ms=80.0, lookahead_ms=5.0):
    c = db(ceiling_db)
    la = int(lookahead_ms * SR / 1000)
    a = np.max(np.abs(x), axis=1)
    pk = maximum_filter1d(a, size=2 * la + 1)
    need = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    r = np.exp(-1.0 / (release_ms * SR / 1000))
    # instant attack (the max filter already looked ahead), smoothed release
    red = lfilter([1 - r], [1, -r], 1 - need)
    g = 1 - np.maximum(red, 1 - need)
    y = x * g[:, None]
    return np.clip(y, -c, c).astype(np.float32)


def gated_rms_db(x, block_s=0.1, rel_gate_db=-30.0):
    """RMS over the blocks where the part is actually playing (so sparse parts aren't judged by their silence)."""
    k = int(block_s * SR)
    m = len(x) // k
    if m == 0:
        return -120.0
    e = np.mean(x[:m * k].reshape(m, k, -1) ** 2, axis=(1, 2))
    lv = 10 * np.log10(e + 1e-12)
    on = lv > max(-110.0, lv.max() + rel_gate_db)
    return float(10 * np.log10(np.mean(e[on]) + 1e-12)) if on.any() else -120.0


def level(x, target_db):
    """Scale a stem so its gated RMS sits at target_db (dBFS)."""
    cur = gated_rms_db(x)
    return x if cur < -110 else (x * db(target_db - cur)).astype(np.float32)


def tape_stop(x, seconds, curve=2.0):
    """The last `seconds` of x slow to a halt (pitch and speed fall together); returns the same length."""
    n = len(x)
    k = min(n, int(seconds * SR))
    speed = (1 - np.linspace(0, 1, k)) ** curve
    pos = (n - k) + np.cumsum(speed)
    y = x.copy()
    i = np.minimum(pos.astype(np.int64), n - 1)
    f = (pos - np.floor(pos))[:, None]
    y[n - k:] = x[i] * (1 - f) + x[np.minimum(i + 1, n - 1)] * f
    y[n - k:] *= np.linspace(1, 0, k)[:, None] ** 0.5
    return y.astype(np.float32)


def fade(x, fin_s=0.0, fout_s=0.0):
    y = x.copy()
    if fin_s > 0:
        k = int(fin_s * SR); y[:k] *= (np.linspace(0, 1, k) ** 2)[:, None]
    if fout_s > 0:
        k = int(fout_s * SR); y[-k:] *= (np.linspace(1, 0, k) ** 2)[:, None]
    return y
