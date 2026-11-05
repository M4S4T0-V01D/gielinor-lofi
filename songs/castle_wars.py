"""Gielinor Lo-Fi - Castle Wars (after the Castle Wars theme, Old School RuneScape), a 1990s jungle detour.

The original (4/4, E minor, 180 bpm) is pressed to a 'record' a semitone down and laid bar for bar at 174 bpm, so
the horn-and-pizzicato ostinato keeps its march. The drums are the source's own kinds (tambourine on 1 and 3 with
the pickup, hats, toms, crash, the taiko war march in 3+3+2) plus a kick and a snare from the same GM kit. They are
programmed as a two-bar funk break at about 130 bpm, pitched up five semitones to 174 like a sampled break on a 90s
sampler, then chopped into jungle edits. A reese and a sub follow the source's pizzicato bass.
This track's move: in the breakdown the break comes back at its programmed speed and is wound up to tempo
(a pitch ramp) into the second drop. No tape stops, reversed pickups, echo throws, timpani rolls, underwater dips
or wind gusts: those belong to the other tracks.
The 90s jungle effects, all synthesized or cut from the break: a dub siren wailing into the drop, the breakdown and
the outro; 8-bit orchestra-hit stabs on the bass root through drop 2; a falling 808 boom under each drop; a pitched
snare rush (16ths into 32nds, rising a fifth) closing each drop; a resonant filter peak riding the break in through
the build; and the break pushed into a crunchy parallel 'desk' bus.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from engine import ambience as AMB, fx, master as M, notes as N, parts as PT  # noqa: E402
from engine.render import SR, fluid, varispeed  # noqa: E402
from engine.sampler import Record, Timeline, edge  # noqa: E402

TITLE = "Castle Wars"
FILE = "Castle Wars"
BPM = 174.0
KEY = -1                      # E minor -> D# minor; the record renders at ~184 bpm, near the source's 180
PUSH = 5                      # semitones the break is pitched up (programmed at BPM * 2**(-PUSH/12) ~ 130 bpm)
SLOW = BPM * 2 ** (-PUSH / 12)
KIT = 0                       # GeneralUser GS standard kit, as the source's drum track
SRC = N.load(ROOT / "source" / "castle_wars.mid")

# source tracks (GM programs): taiko drum 116, french horns 60, pizzicato 45, syn brass1 62, choir ahhs 52,
# trombone 57, church organ 19, harp 46, tremolo strings 44, Track 10 drums, slow strings 49, timpani 47,
# xylophone 13. Record bar k = source bar k; the ostinato cycle (E C F G) starts on even bars.

FORM = [("intro", 0, 8), ("build", 8, 8), ("drop1", 16, 32), ("bridge", 48, 16), ("breakdown", 64, 8),
        ("ramp", 72, 8), ("drop2", 80, 32), ("outro", 112, 16)]
BARS = 128
LEVELS = {"sample": -21, "osti": -25, "strings": -28, "break": -19, "crash": -33, "reese": -27, "sub": -28,
          "siren": -31, "stab": -26, "boom": -22}
SIRENS = [(14, 2, 520, 1180), (62, 2, 440, 980), (110, 2, 560, 1250)]   # (song bar, bars, from Hz, to Hz)
BOOMS = (16, 80)
RUSHES = (47, 111)

# --- the break: two bars at the slow tempo, 16 steps a bar -------------------------------------------------------
# kick and snare are new; tambourine (1, the pickup, 3), 8th hats, toms, open hat and the taiko march are the
# source's own drum kinds. The taiko is pitched G3 so the push lands it near C4, where the source plays it.
TAIKO = 55
BREAK = [  # (step 0-31, piece, velocity); piece None = taiko
    (0, PT.KICK, 112), (3, PT.KICK, 78), (10, PT.KICK, 104),
    (16, PT.KICK, 108), (18, PT.KICK, 86), (26, PT.KICK, 100),
    (4, PT.SNARE, 112), (12, PT.SNARE, 110), (20, PT.SNARE, 112), (28, PT.SNARE, 108),
    (7, PT.SNARE, 40), (9, PT.SNARE, 46), (15, PT.SNARE, 38), (22, PT.SNARE, 42), (25, PT.SNARE, 48),
    (29, PT.SNARE, 34),
    (0, PT.TAMB, 78), (6, PT.TAMB, 62), (8, PT.TAMB, 72), (16, PT.TAMB, 74), (22, PT.TAMB, 60), (24, PT.TAMB, 70),
    (14, PT.OHH, 66), (30, 48, 84), (31, 45, 90),
    (0, None, 74), (6, None, 62), (12, None, 68), (18, None, 64), (22, None, 60), (24, None, 70),
] + [(s, PT.CHH, 92 if s % 4 == 2 else 72) for s in range(0, 32, 2) if s not in (14, 30)]
KICKS = {0, 3, 10, 16, 18, 26}

S = list(range(32))
R2, R4 = 2, 4
# chop patterns, one 2-bar unit each: a slice index plays that 16th of the break; (idx, k) rolls its head k times
STRAIGHT = S
EDIT = S[:16] + [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 12, 13, (4, R2), (4, R2)]
ROLL = S[:16] + [16, 17, 18, 19, 4, 5, 6, 7, (12, R2), (12, R2), (4, R2), (4, R2), (12, R4), (12, R4), (4, R4),
                 (28, R4)]
CHOP = S[:20] + [4, 5, 22, 23, 10, 11, 26, 27, 12, (12, R2), 28, (28, R2)]
CHOP2 = [0, 1, 2, 3, 4, 5, 4, 5, 26, 27, 10, 11, 12, 13, 14, 15, 16, 17, 2, 3, 20, 21, 22, 23, 24, 25, 26, 27, 4,
         (4, R2), 12, (12, R4)]
HALF = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 26, 27, 16, 17, 18, 19, 24, 25, 26, 27, 20, 21, 22, 23, 13,
        14, (12, R2), 15]


def bar_s(b):
    return b * 4 * 60 / BPM


def place_bars(tl, bus, rec, rec_bar, song_bar, nbars, gain=1.0):
    for k in range(nbars):
        tl.put(bus, edge(rec.bar(rec_bar + k, tail_beats=0.2), 2, 25), (song_bar + k) * 4, gain)


def press_break(reps):
    """Render the break at the slow tempo; return (raw, pressed) - pressed is pitched up to the song tempo."""
    rng = np.random.default_rng(5)
    kit, taiko = [], []
    for r in range(reps):
        for step, piece, vel in BREAK:
            at = r * 8 + step / 4 + (0.02 if step % 2 else 0.0)          # a breath of swing on the off-16ths
            vv = int(np.clip(vel + rng.integers(-5, 6), 1, 127))
            if piece is None:
                taiko.append(N.Note(at, 0.5, TAIKO, vv))
            else:
                kit.append(N.Note(at, 0.25 if piece != PT.OHH else 0.4, piece, vv))
    raw = fluid([(kit, KIT, 128, 9), (taiko, 116, 0, 0)], SLOW, tail=1.0, gain=0.6)
    return raw, varispeed(raw, PUSH)


def chop(tl, pressed, units, song_bar, gain=1.0):
    """Lay chop patterns (2 bars each) from song_bar on the 'break' bus; consecutive slices play as one run."""
    p16 = 60 / SLOW / 4 * SR * 2 ** (-PUSH / 12)                       # one 16th of the pressed break, in samples
    beat0 = song_bar * 4
    for u, pat in enumerate(units):
        steps, k = [], 0
        for e in pat:
            steps.append((k, e))
            k += 1
        i = 0
        while i < len(steps):
            at, e = steps[i]
            t = beat0 + u * 8 + at / 4
            if isinstance(e, tuple):                                    # a roll: the slice's head, k times
                idx, rep = e
                head = pressed[int(idx * p16):int(idx * p16 + p16 / rep)]
                for j in range(rep):
                    tl.put("break", edge(head, 1, 4), t + j / (4 * rep), gain * (0.75 + 0.25 * j / rep))
                if idx in KICKS:
                    tl.kicks.append(tl.s(t))
                i += 1
                continue
            j = i
            while j + 1 < len(steps) and not isinstance(steps[j + 1][1], tuple) and steps[j + 1][1] == steps[j][1] + 1:
                j += 1
            run = pressed[int(e * p16):int((steps[j][1] + 1) * p16)]
            tl.put("break", edge(run, 1.5, 5), t, gain)
            for jj in range(i, j + 1):
                if steps[jj][1] in KICKS:
                    tl.kicks.append(tl.s(beat0 + u * 8 + steps[jj][0] / 4))
            i = j + 1


def wind_up(raw, song_bars):
    """The move: the break at its programmed speed, wound up to tempo over song_bars, ending on a loop start."""
    L = int(bar_s(song_bars) * SR)
    R = 2 ** (PUSH / 12)
    rate = R ** (np.arange(L) / L)                                     # source samples per output sample
    pos = np.concatenate([[0.0], np.cumsum(rate[:-1])])
    period = 8 * 60 / SLOW * SR                                        # one 2-bar loop of the raw render
    end = pos[-1] + rate[-1]
    pos += np.ceil(end / period) * period - end
    assert pos[-1] < len(raw) - 2
    return np.stack([np.interp(pos, np.arange(len(raw)), raw[:, c]) for c in range(2)], axis=1).astype(np.float32)


def siren(seconds, f0, f1, lfo_hz=5.5, depth=0.22, seed=7):
    """A ragga dub siren: a square-ish oscillator wailing on a triangle LFO while its centre climbs f0 -> f1."""
    t = np.arange(int(seconds * SR)) / SR
    lfo = 2 * np.abs((t * lfo_hz + np.random.default_rng(seed).random()) % 1 * 2 - 1) - 1
    f = f0 * (f1 / f0) ** (t / seconds) * (1 + depth * lfo)
    ph = 2 * np.pi * np.cumsum(f) / SR
    osc = np.tanh(2.5 * np.sin(ph)) * 0.7 + np.sin(2 * ph) * 0.15
    env = np.minimum(1, t / 0.25) * np.clip((seconds - t) / 0.6, 0, 1)
    x = (osc * env).astype(np.float32)[:, None].repeat(2, axis=1)
    return fx.lowpass(fx.highpass(x, 300, 2), 3800, 2)


def boom(seconds=1.6, f_end=38.9):
    """An 808 drop: a sine falling from 130 Hz to the key's low D# (38.9 Hz), with a click on the front."""
    t = np.arange(int(seconds * SR)) / SR
    f = f_end + (130 - f_end) * np.exp(-t / 0.09)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.65) * np.minimum(1, t / 0.0015)
    return np.tanh(1.6 * x).astype(np.float32)[:, None].repeat(2, axis=1)


def root_at(bass, beat):
    """The bass note sounding at a song beat (or the last one before it)."""
    cur = [x for x in bass if x.start <= beat + 1e-6]
    return max(cur, key=lambda x: x.start).pitch if cur else None


def snare_rush(pressed, steps=24, semis=7.0):
    """One bar: the break's snare slice in 16ths then 32nds, pitched up a fifth on the way, getting louder."""
    p16 = 60 / SLOW / 4 * SR * 2 ** (-PUSH / 12)
    head = pressed[int(4 * p16):int(5 * p16)]
    bar = int(bar_s(1) * SR)
    out = np.zeros((bar + len(head), 2), dtype=np.float32)
    times = [k / 16 for k in range(8)] + [0.5 + k / 32 for k in range(16)]
    for k, at in enumerate(times[:steps]):
        slot = int((1 / 16 if at < 0.5 else 1 / 32) * bar)
        hit = edge(varispeed(head, semis * k / (steps - 1))[:slot], 0.5, 3)
        i0 = int(at * bar)
        out[i0:i0 + len(hit)] += hit * (0.55 + 0.45 * k / (steps - 1))
    return out


def sweep_peak(x, f_of_t, q=6.0, block=256):
    """A resonant bandpass following a cutoff curve: the squelchy peak of a 90s filter sweep."""
    from scipy.signal import butter, sosfilt, sosfilt_zi
    n = len(x)
    out = np.empty_like(x)
    zi = None
    for i0 in range(0, n, block):
        f = float(np.clip(f_of_t(np.array([i0 / SR]))[0], 60, SR / 2.5))
        lo, hi = f * (1 - 0.5 / q), f * (1 + 0.5 / q)
        sos = butter(1, [lo / (SR / 2), hi / (SR / 2)], "bandpass", output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2, x.shape[1]))
        out[i0:i0 + block], zi = sosfilt(sos, x[i0:i0 + block], axis=0, zi=zi)
    return out.astype(np.float32)


def bass_from_pizz(rec_bar, song_bar, nbars):
    """The pizzicato's low line, held legato, registered between C#2 and C3 (source key)."""
    ns = [x for x in N.window(SRC.tracks["pizzicato"], rec_bar, rec_bar + nbars) if x.pitch < 52]
    by = {}
    for x in ns:
        by[round(x.start, 3)] = min(by.get(round(x.start, 3), 999), x.pitch)
    hits = sorted(by.items())
    out = []
    for i, (st, p) in enumerate(hits):
        if out and (p - out[-1].pitch) % 12 == 0 and st - out[-1].start < 3:
            continue                                                     # a repeated note: keep holding
        nxt = next((s2 for s2, p2 in hits[i + 1:] if (p2 - p) % 12), nbars * 4)
        pc = 37 + (p - 37) % 12
        out.append(N.Note(song_bar * 4 + st, max(0.4, min(nxt - st, 3.0) - 0.08), pc, 100))
    return out


def reese(notes, n, seed=3):
    """Two detuned saws a little apart in the stereo field: the 90s reese."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 2), dtype=np.float32)
    bs = 60 / BPM
    for x in notes:
        f = 440 * 2 ** ((x.pitch - 69) / 12)
        i0 = int(x.start * bs * SR)
        L = min(int((x.dur * bs + 0.04) * SR), n - i0)
        if L <= 0:
            continue
        t = np.arange(L) / SR
        a = (f * 2 ** (0.13 / 12) * t + rng.random()) % 1 * 2 - 1
        b = (f * 2 ** (-0.13 / 12) * t + rng.random()) % 1 * 2 - 1
        env = np.minimum(1, t / 0.006) * np.clip((L / SR - t) / 0.04, 0, 1) * (x.vel / 127)
        out[i0:i0 + L, 0] += (a * 0.62 + b * 0.38) * env
        out[i0:i0 + L, 1] += (a * 0.38 + b * 0.62) * env
    return out * 0.3


def build(out_dir: Path, stems: bool = False) -> dict:
    tl = Timeline(BPM, BARS * 4, tail_s=6.0)
    n = tl.n

    # --- the record -----------------------------------------------------------------------------------------
    rec = dict(semis=KEY, dst_bar=4, curve=None)
    intro = Record(SRC, ["taiko drum", "french horns", "pizzicato", "timpani"], BPM, **rec)
    osti = Record(SRC, ["french horns", "pizzicato", "harp", "timpani"], BPM, **rec)
    theme = Record(SRC, ["trombone", "syn brass1", "xylophone"], BPM, **rec)
    pads = Record(SRC, ["choir ahhs", "slow strings", "tremolo strings", "church organ"], BPM, **rec)
    brk = Record(SRC, ["trombone", "harp", "pizzicato"], BPM, **rec)

    place_bars(tl, "osti", intro, 0, 0, 8)                               # intro: the war drum, then the ostinato
    place_bars(tl, "strings", pads, 10, 8, 8)                            # build: choir over the ostinato
    place_bars(tl, "osti", osti, 10, 8, 8)
    place_bars(tl, "osti", osti, 2, 16, 8)                               # drop 1: the ostinato is the hook
    place_bars(tl, "sample", theme, 17, 23, 1)                           # the trombone's pickup
    place_bars(tl, "sample", theme, 18, 24, 16)                          # the theme
    place_bars(tl, "osti", osti, 18, 24, 16, 0.75)
    place_bars(tl, "strings", pads, 18, 24, 16, 0.8)
    place_bars(tl, "sample", theme, 42, 40, 8)                           # the xylophone answer
    place_bars(tl, "osti", osti, 42, 40, 8, 0.75)
    place_bars(tl, "strings", pads, 42, 40, 8)
    place_bars(tl, "strings", pads, 74, 48, 16, 1.2)                     # bridge: choir and organ
    place_bars(tl, "osti", osti, 74, 48, 16, 0.8)
    place_bars(tl, "sample", brk, 138, 64, 8)                            # breakdown: D, F, then E minor
    place_bars(tl, "strings", pads, 138, 64, 16)
    place_bars(tl, "sample", brk, 146, 72, 4, 0.5)
    place_bars(tl, "sample", theme, 57, 79, 1)                           # drop 2
    place_bars(tl, "sample", theme, 58, 80, 16)
    place_bars(tl, "osti", osti, 58, 80, 16, 0.8)
    place_bars(tl, "strings", pads, 58, 80, 16, 0.7)
    place_bars(tl, "sample", theme, 105, 95, 1)
    place_bars(tl, "sample", theme, 106, 96, 16)
    place_bars(tl, "osti", osti, 106, 96, 16, 0.8)
    place_bars(tl, "strings", pads, 106, 96, 16, 0.8)
    place_bars(tl, "osti", osti, 154, 112, 16)                           # outro: the source's own ending
    place_bars(tl, "strings", pads, 154, 112, 16)

    # --- the break ------------------------------------------------------------------------------------------
    raw, pressed = press_break(8)
    chop(tl, pressed, [STRAIGHT, STRAIGHT, STRAIGHT, EDIT], 8)
    drop1 = [STRAIGHT, CHOP, STRAIGHT, EDIT, STRAIGHT, CHOP, STRAIGHT, ROLL] * 2
    chop(tl, pressed, drop1, 16)
    chop(tl, pressed, [HALF] * 7 + [ROLL], 48)
    up = wind_up(raw, 8)
    tl.put("break", edge(up, 400, 2) * np.linspace(0.45, 1.0, len(up))[:, None], 72 * 4)
    drop2 = [CHOP, CHOP2, CHOP, EDIT, CHOP2, CHOP, CHOP2, ROLL] * 2
    chop(tl, pressed, drop2, 80)
    chop(tl, pressed, [STRAIGHT, CHOP, STRAIGHT, EDIT], 112)
    rush = snare_rush(pressed)
    for b0 in RUSHES:                                                  # the drop's last bar becomes a snare rush
        i0, i1 = int(bar_s(b0) * SR), int(bar_s(b0 + 1) * SR)
        tl.bus["break"][i0:i1] *= np.linspace(1, 0, i1 - i0)[:, None] ** 8
        tl.kicks = [k for k in tl.kicks if not bar_s(b0) + 0.05 <= k < bar_s(b0 + 1)]
        tl.put("break", rush, b0 * 4, 0.7)
    tl.kicks.sort()
    for b0, nb, f0, f1 in SIRENS:
        tl.put("siren", siren(bar_s(nb) + 0.6, f0, f1, seed=b0), b0 * 4)
    for b0 in BOOMS:
        tl.put("boom", boom(), b0 * 4)

    crash = varispeed(fluid([([N.Note(0, 2, 57, 104)], KIT, 128, 9)], SLOW, tail=2.5, gain=0.6), PUSH)
    for b0 in (16, 24, 40, 48, 80, 96, 112, 120):
        tl.put("crash", crash, b0 * 4)

    # --- bass: a reese and a sub on the pizzicato's line ------------------------------------------------------
    bass = []
    for song_bar, rec_bar, nb in [(16, 2, 8), (24, 18, 16), (40, 42, 8), (48, 74, 16), (80, 58, 16), (96, 106, 16),
                                  (112, 154, 8)]:
        bass += bass_from_pizz(rec_bar, song_bar, nb)
    bass = N.shift(bass, 0, KEY)
    tl.put_full("reese", reese(bass, n))
    tl.put_full("sub", PT.sub(bass, BPM, n, level=0.5))
    stabs = []                                                         # orchestra hits on the root through drop 2
    for b0 in range(80, 111):
        for at in ([0.0] if b0 % 4 == 0 else []) + ([2.5, 3.25] if b0 % 8 == 6 else []):
            r = root_at(bass, b0 * 4 + at)
            if r is not None:
                stabs += [N.Note(b0 * 4 + at, 0.4, r + d, 112) for d in (12, 19, 24)]
    tl.put_full("stab", fluid([(stabs, 55, 0, 0)], BPM, tail=1.5))

    # --- processing -----------------------------------------------------------------------------------------
    b = tl.bus

    def ride(t):        # opens through the intro, dims for the breakdown, closes through the outro
        f = np.full_like(t, 6000.0)
        f = np.where(t < bar_s(16), 800 * (6000 / 800) ** np.clip(t / bar_s(16), 0, 1) ** 1.3, f)
        f = np.where((t > bar_s(64)) & (t < bar_s(80)), 3200.0, f)
        o0, o1 = bar_s(120), bar_s(127.5)
        f = np.where(t > o0, 6000 * (400 / 6000) ** np.clip((t - o0) / (o1 - o0), 0, 1), f)
        return f

    def break_ride(t):  # the build fades the break in from under a lowpass, the wind-up opens it again
        f = np.full_like(t, 11000.0)
        a0, a1 = bar_s(8), bar_s(16)
        f = np.where((t >= a0) & (t < a1), 350 * (11000 / 350) ** np.clip((t - a0) / (a1 - a0), 0, 1) ** 2, f)
        r0, r1 = bar_s(72), bar_s(80)
        f = np.where((t >= r0) & (t < r1), 700 * (11000 / 700) ** np.clip((t - r0) / (r1 - r0), 0, 1) ** 1.5, f)
        return f

    s = fx.highpass(b["sample"], 160, 2)
    s = fx.bitcrush(s, 12, 26040, mix=0.85)
    s = fx.saturate(s, 1.5)
    s = fx.sweep_lowpass(s, ride)
    b["sample"] = fx.wow_flutter(s, wow_ms=0.8, seed=31) * 1.1
    o = fx.bitcrush(fx.highpass(b["osti"], 180, 2), 12, 26040, mix=0.85)
    b["osti"] = fx.wow_flutter(fx.sweep_lowpass(fx.saturate(o, 1.4), ride), wow_ms=0.8, seed=32)
    st = fx.highpass(b["strings"], 200, 2)
    st = fx.sweep_lowpass(fx.bitcrush(st, 12, 26040, mix=0.5), lambda t: np.minimum(ride(t), 4200.0))
    b["strings"] = fx.widen(fx.wow_flutter(st, seed=33), 1.5) * 0.5

    br = fx.bitcrush(fx.highpass(b["break"], 40, 2), 12, 26040, mix=0.9)            # the sampler
    br = fx.compress(fx.saturate(fx.shelf_tilt(br, 900, 2.0), 1.7), -18, 4, 2, 90, makeup_db=2)
    b["break"] = fx.sweep_lowpass(br, break_ride)
    b["crash"] = fx.lowpass(fx.bitcrush(fx.highpass(b["crash"], 400, 2), 12, 26040), 9000, 2)
    peak = sweep_peak(b["break"], lambda t: np.where((t >= bar_s(8)) & (t < bar_s(16)), break_ride(t), 20.0))
    b["break"] = b["break"] + peak * 1.4 * (np.arange(n) < int(bar_s(16) * SR))[:, None]
    desk = fx.saturate(fx.bandpass(b["break"], 180, 7000, 2), 5.0)        # the break pushed into the desk
    b["break"] = b["break"] + fx.compress(desk, -20, 6, 1, 60) * 0.3
    sr_ = fx.reverb(b["siren"], seconds=2.6, damp_hz=3000, predelay_ms=30, seed=47)
    b["siren"] = fx.autopan(b["siren"] + sr_ * 0.6, rate=BPM / 60 / 4, depth=0.5)
    sb = fx.bitcrush(fx.highpass(b["stab"], 250, 2), 8, 11025)             # an 8-bit sampler hit
    b["stab"] = fx.widen(fx.saturate(sb, 1.8), 1.3)
    b["boom"] = fx.lowpass(b["boom"], 220, 2)

    rz = fx.highpass(b["reese"], 70, 2)
    rz = fx.sweep_lowpass(rz, lambda t: 650 * 2 ** (0.9 * np.sin(2 * np.pi * t / bar_s(4))))
    b["reese"] = fx.mono_below(fx.compress(fx.saturate(rz, 2.2), -18, 3, 5, 80), 160)
    b["sub"] = fx.lowpass(b["sub"], 100, 2)

    for k, target in LEVELS.items():
        b[k] = fx.level(b[k], target)

    send = (b["sample"] * 0.35 + b["osti"] * 0.3 + b["strings"] * 0.6 + b["break"] * 0.08 + b["crash"] * 0.3
            + b["stab"] * 0.4)
    verb = fx.reverb(send, seconds=2.2, damp_hz=3600, predelay_ms=18)

    music = (b["sample"] + b["osti"] + b["strings"]) * fx.duck(n, tl.kicks, depth_db=2.0, release_ms=140)[:, None]
    low = (b["reese"] + b["sub"]) * fx.duck(n, tl.kicks, depth_db=3.5, release_ms=110)[:, None]
    drum_bus = fx.tape(fx.compress(b["break"] + b["crash"], -15, 3, 4, 120, makeup_db=1), 1.4, 12000)
    hiss = AMB.hiss(n / SR, -62, seed=46)

    mix = music + b["stab"] + b["siren"] + low + b["boom"] + drum_bus + verb * 0.5 + hiss
    end = int((bar_s(BARS) + 4.5) * SR)
    mix = fx.fade(mix[:end], fin_s=0.5, fout_s=3.5)
    out = M.master(mix, -14.0, -1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{FILE}.wav"
    M.write(path, out, tags={"title": TITLE, "artist": "Gielinor Lo-Fi", "album": "Gielinor Lo-Fi",
                             "comment": "Jungle reinterpretation of the Castle Wars theme (Jagex) from Old School RuneScape"})
    if stems:
        sd = out_dir / "stems"
        sd.mkdir(exist_ok=True)
        import soundfile as sf
        for k, x in {**b, "verb": verb}.items():
            sf.write(str(sd / f"{k}.wav"), x[:end], SR, subtype="PCM_16")
    rep = M.report(out, BPM, [(nm, b0, b0 + nb) for nm, b0, nb in FORM])
    rep["file"] = str(path)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    print(json.dumps(build(ROOT / "build" / "castle_wars", stems="--stems" in sys.argv), indent=1))
