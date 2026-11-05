"""Synthesized textures: vinyl surface, tape hiss, rain, harbour waves, seagulls. All generated from seeds, so nothing here is sampled."""
from __future__ import annotations

import numpy as np

from . import fx
from .render import SR

RPM_HZ = 33.333 / 60.0


def _pink(n, rng):
    """Voss-ish pink noise via FFT shaping; returns (n, 2)."""
    w = rng.standard_normal((n, 2))
    W = np.fft.rfft(w, axis=0)
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    W /= np.sqrt(f)[:, None]
    p = np.fft.irfft(W, n=n, axis=0)
    return (p / np.std(p)).astype(np.float32)


def _clicks(n, rate, amp, rng, width=(1, 6)):
    out = np.zeros((n, 2), dtype=np.float32)
    k = rng.poisson(rate * n / SR)
    pos = rng.integers(0, n - 64, k)
    for p in pos:
        w = rng.integers(*width)
        a = amp * rng.uniform(0.2, 1.0) ** 2 * rng.choice((-1, 1))
        shape = np.exp(-np.arange(w * 4) / w) * a
        side = rng.uniform(0.3, 1.0)
        ch = rng.integers(0, 2)
        out[p:p + len(shape), ch] += shape
        out[p:p + len(shape), 1 - ch] += shape * side
    return out


def vinyl(seconds, level_db=-38.0, seed=11, crackle_rate=14.0):
    """Crackle, the once-per-revolution tick of a scratch, surface noise and rumble.
    Dense crackle (the default 14/s) reads as rain; use a few per second where there must be no rain."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    crackle = _clicks(n, crackle_rate, 1.0, rng) + _clicks(n, 0.6, 3.0, rng, (3, 12))
    crackle = fx.highpass(crackle, 900, 2)
    tick = np.zeros((n, 2), dtype=np.float32)
    period = SR / RPM_HZ
    for p in np.arange(rng.uniform(0, period), n - 200, period):
        i = int(p + rng.normal(0, 8))
        tick[i:i + 40] += (np.exp(-np.arange(40) / 6) * 1.6)[:, None] * np.array([1.0, 0.8])
    tick = fx.bandpass(tick, 700, 5000, 1)
    surface = fx.bandpass(_pink(n, rng), 300, 9000, 1) * 0.12
    rumble = fx.lowpass(rng.standard_normal((n, 2)).astype(np.float32), 35, 2) * 0.9
    rumble = rumble.mean(axis=1, keepdims=True).repeat(2, axis=1)
    x = crackle * 0.5 + tick * 0.25 + surface + rumble
    return (x / max(1e-9, np.sqrt(np.mean(x ** 2))) * fx.db(level_db)).astype(np.float32)


def hiss(seconds, level_db=-54.0, seed=12):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = fx.highpass(rng.standard_normal((n, 2)).astype(np.float32), 2500, 1)
    x = fx.lowpass(x, 14000, 1)
    t = np.arange(n) / SR
    x *= (1 + 0.08 * np.sin(2 * np.pi * 0.13 * t))[:, None]
    return (x / np.sqrt(np.mean(x ** 2)) * fx.db(level_db)).astype(np.float32)


def rain(seconds, level_db=-30.0, intensity=1.0, seed=13, window=True):
    """Rain heard from indoors: a soft noise bed, dense small drops, drops on the window, and a gutter drip."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    swell = 1 + 0.25 * np.interp(t, np.linspace(0, seconds, int(seconds / 6) + 2),
                                 rng.uniform(-1, 1, int(seconds / 6) + 2))
    bed = fx.bandpass(_pink(n, rng), 350, 6500, 2) * 0.5 * swell[:, None]
    # small drops: tiny noise bursts through a resonant-ish band
    drops = _clicks(n, 260 * intensity, 1.0, rng, (2, 10))
    drops = fx.bandpass(drops, 1800, 7000, 1) * 0.6
    # drops on glass/sill: lower, sparser, a little ring
    hits = np.zeros((n, 2), dtype=np.float32)
    if window:
        k = rng.poisson(5 * intensity * seconds)
        for p in rng.integers(0, n - 4000, k):
            f = rng.uniform(900, 2600)
            L = int(rng.uniform(0.01, 0.035) * SR)
            s = np.sin(2 * np.pi * f * np.arange(L) / SR) * np.exp(-np.arange(L) / (L / 5))
            s += rng.standard_normal(L) * np.exp(-np.arange(L) / (L / 12)) * 0.6
            pnorm = rng.uniform(-0.8, 0.8)
            a = rng.uniform(0.2, 1.0) ** 2 * 0.5
            hits[p:p + L, 0] += s * a * (1 - pnorm) / 2
            hits[p:p + L, 1] += s * a * (1 + pnorm) / 2
    # a gutter drip: slow, irregular, pitched plinks off to one side
    drip = np.zeros((n, 2), dtype=np.float32)
    p = rng.uniform(0.5, 2) * SR
    while p < n - SR:
        L = int(0.09 * SR)
        f0 = rng.uniform(1100, 1500)
        tt = np.arange(L) / SR
        s = np.sin(2 * np.pi * (f0 * tt + 2500 * tt * tt)) * np.exp(-tt / 0.018)
        i = int(p)
        drip[i:i + L, 0] += s * 0.10
        drip[i:i + L, 1] += s * 0.04
        p += rng.uniform(1.1, 2.6) * SR
    x = bed + drops + hits + drip
    x = fx.lowpass(x, 9000, 1)     # through a window
    return (x / np.sqrt(np.mean(x ** 2)) * fx.db(level_db)).astype(np.float32)


def _brown(n, rng):
    w = np.cumsum(rng.standard_normal((n, 2)), axis=0)
    w = fx.highpass(w.astype(np.float32), 15, 1)
    return (w / np.std(w)).astype(np.float32)


def _surf(rng, rise, fall, size):
    """One wave on the shore. It comes in as a darkening rumble that brightens as it steepens, breaks as a
    broadband burst, then drags back out as fizzing foam. Returns (L, 2) audio and the crest index."""
    L = int((rise + fall + 0.5) * SR)
    u = np.arange(L) / SR
    k = int(rise * SR)
    # the swell: slow, then steepening; the break: ~0.1 s to its peak; the wash: a long decay
    amp = np.where(u < rise, 0.04 + 0.3 * (u / rise) ** 3,
                   0.34 + 0.66 * np.exp(-(u - rise) / 0.4) * np.minimum(1, (u - rise) / 0.08))
    amp = np.where(u > rise + 0.08, amp * np.exp(-np.clip(u - rise - 0.08, 0, None) / (fall * 0.3)), amp)

    def cutoff(tt):
        up = 220 * (2600 / 220) ** np.clip(tt / rise, 0, 1) ** 2
        down = 6500 * (700 / 6500) ** np.clip((tt - rise) / (fall * 0.8), 0, 1) ** 0.6
        return np.where(tt < rise, up, down)
    body = fx.sweep_lowpass(_pink(L, rng), cutoff) * amp[:, None]
    # the crash itself: a short burst of bright noise right on the break
    b = np.exp(-np.clip(u - rise, 0, None) / 0.18) * np.minimum(1, np.clip(u - rise, 0, None) / 0.03)
    burst = fx.bandpass(rng.standard_normal((L, 2)).astype(np.float32), 600, 8000, 1) * (b * 0.7)[:, None]
    # foam: thousands of popping bubbles, densest after the break, thinning as the water drains
    foam = np.zeros((L, 2), dtype=np.float32)
    dens = np.where(u < rise - 0.2, 0, np.exp(-np.clip(u - rise, 0, None) / (fall * 0.18)))
    nb = int(900 * size * fall)
    pos = rng.choice(L, nb, p=dens / dens.sum())
    for p in pos:
        f = rng.uniform(1500, 7000)
        m = int(rng.uniform(0.002, 0.006) * SR)
        if p + m >= L:
            continue
        tt = np.arange(m) / SR
        s = np.sin(2 * np.pi * f * (1 + 40 * tt) * tt) * np.exp(-tt * 900) * rng.uniform(0.1, 1.0) ** 2
        c = rng.integers(0, 2)
        foam[p:p + m, c] += s
        foam[p:p + m, 1 - c] += s * rng.uniform(0.2, 0.8)
    foam = fx.lowpass(foam, 5000, 1) * 0.3
    x = (body + burst + foam) * size
    # the wave runs along the shore: it travels a little across the stereo field while it breaks
    p0 = rng.uniform(-0.6, 0.6)
    pan = np.clip(p0 + np.linspace(0, rng.uniform(-0.4, 0.4), L), -0.9, 0.9)
    x[:, 0] *= np.sqrt((1 - pan) / 2) * 1.41
    x[:, 1] *= np.sqrt((1 + pan) / 2) * 1.41
    return x.astype(np.float32), k


def waves(seconds, level_db=-30.0, seed=21, period=(6.0, 10.0), crashes=()):
    """Surf on the shore below the docks: a constant low roar of the open water, and wave after wave coming in,
    breaking and draining back out through the foam. `crashes` adds a big wave that breaks at each of those seconds."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    # the open sea: brown-noise roar that breathes slowly, plus faraway surf as a dull wash
    breathe = 1 + 0.3 * np.interp(t, np.linspace(0, seconds, int(seconds / 4) + 2),
                                  rng.uniform(-1, 1, int(seconds / 4) + 2))
    roar = fx.lowpass(_brown(n, rng), 180, 2) * 0.16
    far = fx.lowpass(fx.highpass(_pink(n, rng), 250, 1), 900, 2) * 0.05
    x = (roar + far) * breathe[:, None].astype(np.float32)
    waves_at = []
    t0 = rng.uniform(1.0, 3.0)
    while t0 < seconds:
        waves_at.append((t0, rng.uniform(1.8, 3.2), rng.uniform(3.5, 6.0), rng.uniform(0.55, 1.0)))
        t0 += rng.uniform(*period)
    waves_at += [(c, 2.6, 7.0, 1.7) for c in crashes]
    for crest, rise, fall, size in waves_at:
        w, k = _surf(rng, rise, fall, size)
        i0 = int(crest * SR) - k
        a, b = max(0, i0), min(n, i0 + len(w))
        if b > a:
            x[a:b] += w[a - i0:b - i0]
    return (x / np.sqrt(np.mean(x ** 2)) * fx.db(level_db)).astype(np.float32)


def wind(seconds, level_db=-36.0, seed=31, gusts=()):
    """Wind over open country: a low, dark body that never stops moving, and gusts that swell in, brighten and
    whistle through the trees, then fall away. Kept under ~1.5 kHz and always moving, so it can't read as rain or
    hiss. `gusts` forces a big gust that peaks at each of those seconds."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR

    def drift(step, lo, hi):          # a smooth random curve, one new point every `step` seconds
        k = int(seconds / step) + 3
        pts = rng.uniform(lo, hi, k)
        return np.interp(t, np.arange(k) * step, np.convolve(pts, np.ones(2) / 2, mode="same"))
    # gust envelope: slow breathing plus random gusts (rise 2-4 s, fall 3-6 s) and the forced ones
    env = drift(3.0, 0.25, 0.6)
    peaks = []
    t0 = rng.uniform(2, 6)
    while t0 < seconds:
        peaks.append((t0, rng.uniform(2, 4), rng.uniform(3, 6), rng.uniform(0.4, 0.8)))
        t0 += rng.uniform(7, 14)
    peaks += [(g, 4.0, 6.0, 1.4) for g in gusts]
    for c, rise, fall, h in peaks:
        u = t - c
        env += h * np.where(u < 0, np.exp(-(u / rise) ** 2 * 3), np.exp(-(u / fall) ** 2 * 3))
    body = fx.lowpass(_brown(n, rng), 420, 2) * 0.7 + fx.bandpass(_pink(n, rng), 150, 700, 1) * 0.35
    body = fx.sweep_lowpass(body, lambda tt: np.interp(tt, t, 300 + 900 * np.clip(env, 0, 1.6) / 1.6))
    # the whistle through branches: a narrow resonance that wanders in pitch, only heard in the gusts
    hiss = _pink(n, rng)
    howl = np.zeros_like(hiss)
    fc = drift(1.7, 520, 1100)
    blk = 2048
    for i in range(0, n, blk):
        f0 = float(fc[min(n - 1, i + blk // 2)])
        seg = hiss[max(0, i - blk):i + blk]
        y = fx.bandpass(seg, f0 * 0.94, f0 * 1.06, 2)
        howl[i:i + blk] = y[min(i, blk):min(i, blk) + blk][:len(howl[i:i + blk])]
    howl *= np.clip(env - 0.55, 0, None)[:, None] * 0.9
    x = body * env[:, None] + howl
    # the air moves across the field
    pan = 0.5 * np.sin(2 * np.pi * t / 23 + rng.uniform(0, 6))
    x[:, 0] *= np.sqrt((1 - pan) / 2) * 1.41
    x[:, 1] *= np.sqrt((1 + pan) / 2) * 1.41
    x = fx.lowpass(x, 1800, 2)
    return (x / np.sqrt(np.mean(x ** 2)) * fx.db(level_db)).astype(np.float32)


def _gull_call(rng, f_hi, dur):
    """One 'kee-ow': a nasal harmonic tone that snaps up then sags, with a rough throat flutter."""
    L = int(dur * SR)
    u = np.arange(L) / L
    f = np.where(u < 0.12, f_hi * (0.75 + 0.25 * u / 0.12), f_hi * 0.62 ** (np.clip((u - 0.12) / 0.88, 0, 1) ** 0.7))
    ph = np.cumsum(2 * np.pi * f / SR)
    s = sum(np.sin(k * ph) / k ** 1.1 for k in range(1, 7))
    s *= 1 + 0.45 * np.sin(2 * np.pi * rng.uniform(70, 110) * np.arange(L) / SR)
    env = np.minimum(1, u / 0.06) * np.minimum(1, (1 - u) / 0.25)
    return (s * env).astype(np.float32)


def gulls(seconds, level_db=-34.0, seed=22, rate=1 / 14, times=()):
    """Seagulls overhead and across the harbour: bursts of 2-5 calls at random spots in the stereo field.
    `times` forces a burst at those seconds (e.g. the opening bar); `rate` adds random ones (bursts per second)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = np.zeros((n, 2), dtype=np.float32)
    starts = list(times) + list(np.sort(rng.uniform(0, seconds - 3, rng.poisson(rate * seconds))))
    for t0 in starts:
        p, near = rng.uniform(-0.8, 0.8), rng.uniform(0.3, 1.0)
        f_hi, t = rng.uniform(900, 1250), t0
        for _ in range(rng.integers(2, 6)):
            c = _gull_call(rng, f_hi, rng.uniform(0.22, 0.4)) * near
            i = int(t * SR)
            m = min(len(c), n - i)
            if m <= 0:
                break
            x[i:i + m, 0] += c[:m] * (1 - p) / 2
            x[i:i + m, 1] += c[:m] * (1 + p) / 2
            t += rng.uniform(0.32, 0.55)
            f_hi *= rng.uniform(0.94, 1.0)
    x = fx.lowpass(fx.highpass(x, 600, 2), 4500, 2)      # distance takes the edge off
    x = x + fx.reverb(x, seconds=1.4, damp_hz=3000, predelay_ms=40) * 0.5
    return fx.level(x, level_db)


def _tone(f, amp):
    """A sine following a per-sample frequency curve under an amplitude curve."""
    return (np.sin(np.cumsum(2 * np.pi * f / SR)) * amp).astype(np.float32)


def _song_bird(rng, kind):
    """One phrase from a small garden bird: 'tweet' (falling chirps), 'trill' (a fast pulsed note), or 'whistle'
    (a clear two-note fee-bee)."""
    if kind == "tweet":
        out = []
        f0 = rng.uniform(3800, 5200)
        for _ in range(rng.integers(3, 8)):
            L = int(rng.uniform(0.04, 0.08) * SR)
            u = np.arange(L) / L
            f = f0 * (1.25 - 0.45 * u) * (1 + 0.04 * np.sin(2 * np.pi * 90 * np.arange(L) / SR))
            out += [_tone(f, np.sin(np.pi * u) ** 1.5), np.zeros(int(rng.uniform(0.05, 0.11) * SR), np.float32)]
            f0 *= rng.uniform(0.95, 1.04)
        return np.concatenate(out)
    if kind == "trill":
        L = int(rng.uniform(0.5, 1.1) * SR)
        u = np.arange(L) / L
        t = np.arange(L) / SR
        f = rng.uniform(3200, 4400) * (1 + 0.06 * u)
        pulse = np.clip(np.sin(2 * np.pi * rng.uniform(18, 28) * t), 0, None) ** 2
        return _tone(f, pulse * np.minimum(1, u / 0.15) * np.minimum(1, (1 - u) / 0.2))
    f1 = rng.uniform(3300, 4100)
    parts = []
    for f, d in ((f1, 0.28), (f1 * rng.uniform(0.78, 0.86), 0.34)):
        L = int(d * SR)
        u = np.arange(L) / L
        parts += [_tone(np.full(L, f) * (1 - 0.02 * u), np.minimum(1, u / 0.1) * np.minimum(1, (1 - u) / 0.3)),
                  np.zeros(int(0.06 * SR), np.float32)]
    return np.concatenate(parts)


def birds(seconds, level_db=-40.0, seed=41, rate=0.22, quiet=()):
    """Morning birdsong from the hedges: a few birds, each keeping to one kind of call and one spot in the stereo
    field, singing phrases at random. `quiet` is a list of (t0, t1) seconds where they hold off."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = np.zeros((n, 2), dtype=np.float32)
    flock = [(k, rng.uniform(-0.8, 0.8), rng.uniform(0.35, 1.0)) for k in ("tweet", "trill", "whistle", "tweet")]
    t0 = rng.uniform(0.5, 2.0)
    while t0 < seconds - 2:
        kind, p, near = flock[rng.integers(len(flock))]
        if not any(a <= t0 < b for a, b in quiet):
            c = _song_bird(rng, kind) * near
            i = int(t0 * SR)
            m = min(len(c), n - i)
            x[i:i + m, 0] += c[:m] * np.sqrt((1 - p) / 2)
            x[i:i + m, 1] += c[:m] * np.sqrt((1 + p) / 2)
        t0 += rng.exponential(1 / rate) + 0.6
    x = fx.lowpass(fx.highpass(x, 1800, 2), 7500, 2)       # across the field, not at the window
    x = x + fx.reverb(x, seconds=1.2, damp_hz=5000, predelay_ms=30, hp=1500) * 0.4
    return fx.level(x, level_db)


def crickets(seconds, level_db=-42.0, seed=51, count=4):
    """A warm night: a few crickets, each chirping its own pulse train (3-4 pulses at ~30 Hz, ~2 chirps a second)
    on its own carrier near 4.5 kHz, spread across the field. Steady and tonal, so it never reads as rain."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = np.zeros((n, 2), dtype=np.float32)
    for _ in range(count):
        fc, pr, cr = rng.uniform(4000, 5200), rng.uniform(26, 34), rng.uniform(1.6, 2.6)
        pulses = rng.integers(3, 5)
        ph = rng.uniform(0, 1)
        cyc = (t * cr + ph) % 1.0                         # position in the chirp cycle
        on = (cyc * pr / cr < pulses).astype(np.float32)
        pul = np.sin(np.pi * ((cyc * pr / cr) % 1.0)) ** 2 * on
        # crickets rest and restart
        knots = (rng.uniform(0, 1, int(seconds / 4) + 3) > 0.25).astype(float)
        sing = np.interp(t, np.arange(len(knots)) * 4, np.convolve(knots, [0.5, 0.5], "same"))
        s = np.sin(2 * np.pi * fc * t) * pul * sing * rng.uniform(0.4, 1.0)
        p = rng.uniform(-0.9, 0.9)
        x[:, 0] += s * np.sqrt((1 - p) / 2)
        x[:, 1] += s * np.sqrt((1 + p) / 2)
    x = fx.lowpass(x, 6500, 2)
    x = x + fx.reverb(x, seconds=0.9, damp_hz=5000, predelay_ms=20, hp=2000) * 0.3
    return fx.level(x, level_db)
