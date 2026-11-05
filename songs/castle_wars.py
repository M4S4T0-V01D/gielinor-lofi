"""Gielinor Lo-Fi - Castle Wars (after the Castle Wars theme, Old School RuneScape), a 1990s jungle detour.

The original (4/4, E minor, 180 bpm) is pressed to a 'record' a semitone down and laid bar for bar at 174 bpm, so
the horn-and-pizzicato ostinato keeps its march. The drums are the source's own kinds (tambourine on 1 and 3 with
the pickup, hats, toms, crash, the taiko war march in 3+3+2) plus a kick and a snare from the same GM kit. They are
programmed as a two-bar funk break at about 130 bpm, pitched up five semitones to 174 like a sampled break on a 90s
sampler, then chopped into jungle edits. A reese and a sub follow the source's pizzicato bass.
This track's move: after the breakdown the break comes back frozen in an 8x timestretch and thaws to tempo, the
sampler steps it up in pitch bar by bar, and its snare is held as a rising stretched buzz that cuts dead before
the second drop. No tape stops, reversed pickups, echo throws, timpani rolls, underwater dips
or wind gusts: those belong to the other tracks.
The 90s jungle effects, all synthesized or cut from the break: a dub siren wailing into the drop, the breakdown and
the outro; brass-and-timpani stabs (horns, trombone, pizzicato) on the bass root through drop 2; a falling 808
boom under each drop; a pitched snare rush (16ths into 32nds, rising a fifth) closing each drop; a resonant filter peak riding the break in through
the build; and the break pushed into a crunchy parallel 'desk' bus.
Timestretch, the way a 90s sampler did it (akai(): fixed grains read slower than written, so it buzzes): the
source's choir, sung as Voice Oohs, stretched to half speed as the intro and outro pads, with a robot octave
pitched up the same way in the breakdown and gated 'ooh' chops in the drops; a beat of the break dragged over two
at phrase ends; the bridge's last two bars at half speed; and the break stretched out to a crawl at the end.
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
          "siren": -31, "stab": -26, "boom": -22, "vox": -24, "chops": -27,
          "organ": -23}
ORGAN_LIFT = (48, 64, 3.0)    # the bridge: the organ steps out front (bars, dB)
VOX_HITS = [(0, 3), (6, 2), (10, 3), (14, 2), (16, 3), (22, 2), (26, 2), (28, 4)]   # (16th of 2 bars, 16ths)
RAMP_STEPS = [(75, 2, 40), (76, 4, 32), (77, 7, 24), (78, 12, 18)]  # (bar, semitones up, grain ms)
STRETCH_FILLS = {23: (4, 2.0), 39: (12, "rubber"), 87: (4, 2.0), 103: (12, "rubber")}  # bar: (slice, ratio)
SIRENS = [(14, 2, 520, 1180), (62, 2, 440, 980), (110, 2, 560, 1250)]   # (song bar, bars, from Hz, to Hz)
BOOMS = (16, 72, 80)
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


def akai(x, ratio, grain_ms=45.0, xf_ms=4.0, n_out=None, start=0.0):
    """A 90s sampler's cyclic timestretch: fixed grains, read slower than they are written, butted together with
    tiny crossfades, so the stretched sound warbles and buzzes at the grain rate. ratio > 1 is longer; it can be
    a function of output progress 0..1 (then n_out sets the length). start is the read position in samples."""
    g, xf = int(grain_ms * SR / 1000), max(1, int(xf_ms * SR / 1000))
    L = n_out if n_out is not None else int(len(x) * ratio)
    win = np.ones(g + xf, dtype=np.float32)
    win[:xf], win[-xf:] = np.linspace(0, 1, xf), np.linspace(1, 0, xf)
    out = np.zeros((L + g + xf, x.shape[1]), dtype=np.float32)
    pos, o = float(start), 0
    while o < L:
        r = ratio(o / L) if callable(ratio) else ratio
        i = int(pos)
        seg = x[i:i + g + xf]
        if len(seg) < g + xf:
            seg = np.pad(seg, ((0, g + xf - len(seg)), (0, 0)))
        out[o:o + g + xf] += seg * win[:, None]
        o += g
        pos += g / r
    return out[:L]


def thaw(pressed, bars, from_ratio=8.0, grain_ms=60.0):
    """The move: the break comes back frozen on the sampler, stretched 8x into a buzzing smear, and un-stretches
    to tempo over `bars`, timed so it lands on a loop start and the edits after it carry on in the groove."""
    L = int(bar_s(bars) * SR)
    g = int(grain_ms * SR / 1000)
    ratio = lambda p: from_ratio ** ((1 - p) ** 2)                      # stays smeared, then snaps in
    read = sum(g / ratio(o / L) for o in range(0, L, g))
    period = 32 * 60 / SLOW / 4 * SR * 2 ** (-PUSH / 12)                 # one 2-bar loop of the pressed break
    start = np.ceil(read / period) * period - read
    return akai(pressed, ratio, grain_ms, xf_ms=6, n_out=L, start=start)


def akai_pitch(x, semis, grain_ms=35.0):
    """Pitch without tempo, the sampler way: varispeed, then timestretched back to the original length."""
    y = varispeed(x, semis)
    return akai(y, len(x) / len(y), grain_ms, n_out=len(x))


def stretch_in(tl, x, bar, beats, gain=1.0):
    """Replace a stretch of the break bus with x (the break's own slices, stretched), with short edges."""
    b = tl.bus["break"]
    i0 = tl.i(bar * 4 + (4 - beats))
    i1 = min(i0 + len(x), tl.n)
    f = int(0.004 * SR)
    b[i0 - f:i0] *= np.linspace(1, 0, f)[:, None]
    b[i0:i1] = 0
    tl.put("break", edge(x, 2, 6), bar * 4 + (4 - beats), gain)
    tl.kicks = [k for k in tl.kicks if not i0 / SR <= k < i1 / SR]


def vox_chops(tl, vox, rec_bar, song_bar, units, seed=11):
    """Gated 'ooh' chops off the choir, on an off-beat ragga pattern, every third one an octave up."""
    rng = np.random.default_rng(seed)
    loop = vox.bar(rec_bar, rec_bar + 2, tail_beats=0.5)
    s16 = 60 / BPM / 4 * SR
    for u in range(units):
        for k, (st, ln) in enumerate(VOX_HITS):
            x = loop[int(st * s16):int((st + ln) * s16 + 0.03 * SR)]
            if k % 3 == 2:
                x = akai_pitch(x, 12.0)
            tl.put("chops", edge(x, 3, 30), song_bar * 4 + u * 8 + st / 4, 0.8 + 0.2 * rng.random())


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
    intro = Record(SRC, ["taiko drum", "timpani"], BPM, **rec)
    osti = Record(SRC, ["french horns", "pizzicato", "harp", "timpani"], BPM, **rec)
    theme = Record(SRC, ["trombone", "syn brass1", "xylophone"], BPM, **rec)
    pads = Record(SRC, ["slow strings", "tremolo strings"], BPM, **rec)
    organ = Record(SRC, ["church organ"], BPM, **rec)
    vox = Record(SRC, ["choir ahhs"], BPM, programs={"choir ahhs": 53}, **rec)   # the choir sung as 'oohs'
    brk = Record(SRC, ["trombone", "harp", "pizzicato"], BPM, **rec)

    def place_pads(rec_bar, song_bar, nbars, gain=1.0):
        place_bars(tl, "strings", pads, rec_bar, song_bar, nbars, gain)
        place_bars(tl, "vox", vox, rec_bar, song_bar, nbars, gain)
        place_bars(tl, "organ", organ, rec_bar, song_bar, nbars, gain)

    place_bars(tl, "osti", intro, 0, 0, 8)                               # intro: the war drum under the oohs,
    pad = vox.bar(10, 14, tail_beats=2)                                  # stretched to half speed on the sampler
    tl.put("vox", edge(akai(pad, 2.0, grain_ms=70, xf_ms=8), 900, 400), 0)
    tl.put("vox", edge(akai_pitch(akai(pad, 2.0, grain_ms=70, xf_ms=8), 12.0, 50), 900, 400), 0, 0.3)
    place_pads(10, 8, 8)                                                 # build: the oohs over the ostinato
    place_bars(tl, "osti", osti, 10, 8, 8)
    place_bars(tl, "osti", osti, 2, 16, 8)                               # drop 1: the ostinato is the hook
    place_bars(tl, "sample", theme, 17, 23, 1)                           # the trombone's pickup
    place_bars(tl, "sample", theme, 18, 24, 16)                          # the theme
    place_bars(tl, "osti", osti, 18, 24, 16, 0.75)
    place_pads(18, 24, 16, 0.8)
    place_bars(tl, "sample", theme, 42, 40, 8)                           # the xylophone answer
    place_bars(tl, "osti", osti, 42, 40, 8, 0.75)
    place_pads(42, 40, 8)
    place_pads(74, 48, 16, 1.2)                                          # bridge: the ooh melody and organ
    place_bars(tl, "osti", osti, 74, 48, 16, 0.8)
    place_bars(tl, "sample", brk, 138, 64, 8)                            # breakdown: D, F, then E minor
    place_pads(138, 64, 16)
    place_bars(tl, "sample", brk, 146, 72, 4, 0.5)
    place_bars(tl, "sample", theme, 57, 79, 1)                           # drop 2
    place_bars(tl, "sample", theme, 58, 80, 16)
    place_bars(tl, "osti", osti, 58, 80, 16, 0.8)
    place_pads(58, 80, 16, 0.7)
    place_bars(tl, "sample", theme, 105, 95, 1)
    place_bars(tl, "sample", theme, 106, 96, 16)
    place_bars(tl, "osti", osti, 106, 96, 16, 0.8)
    place_pads(106, 96, 16, 0.8)
    place_bars(tl, "osti", osti, 154, 112, 16)                           # outro: the source's own ending
    place_pads(154, 112, 16)
    place_bars(tl, "vox", vox, 122, 112, 8)                              # outro: the oohs come back, then
    tl.put("vox", edge(akai(pad, 2.0, grain_ms=70, xf_ms=8), 300, 2500), 120 * 4)   # stretch out like the intro
    hi = akai_pitch(vox.bar(138, 154, tail_beats=2), 12.0, 40)          # breakdown: a robot octave over the oohs
    tl.put("vox", edge(hi, 600, 600), 64 * 4, 0.35)
    vox_chops(tl, vox, 10, 32, 4)                                        # ragga 'ooh' chops in the drops
    vox_chops(tl, vox, 10, 88, 4, seed=12)
    vox_chops(tl, vox, 10, 104, 3, seed=13)

    # --- the break ------------------------------------------------------------------------------------------
    _, pressed = press_break(8)
    chop(tl, pressed, [STRAIGHT, STRAIGHT, STRAIGHT, EDIT], 8)
    drop1 = [STRAIGHT, CHOP, STRAIGHT, EDIT, STRAIGHT, CHOP, STRAIGHT, ROLL] * 2
    chop(tl, pressed, drop1, 16)
    chop(tl, pressed, [HALF] * 7 + [ROLL], 48)
    tl.put("break", edge(thaw(pressed, 2), 1, 4), 72 * 4)            # the way back: see RAMP below
    chop(tl, pressed, [STRAIGHT, EDIT, CHOP2], 74)
    drop2 = [CHOP, CHOP2, CHOP, EDIT, CHOP2, CHOP, CHOP2, ROLL] * 2
    chop(tl, pressed, drop2, 80)
    chop(tl, pressed, [STRAIGHT, CHOP, STRAIGHT, EDIT, STRAIGHT, CHOP], 112)

    # timestretches on the sampler: a beat dragged over two, the bridge's last two bars at half speed (same
    # pitch), and the break stretched out to a crawl over the last four bars
    p16 = 60 / SLOW / 4 * SR * 2 ** (-PUSH / 12)
    beat = int(4 * p16)
    for b0, (idx, ratio) in STRETCH_FILLS.items():
        sl = pressed[int(idx * p16):int(idx * p16) + beat + int(0.05 * SR)]
        if ratio == "rubber":                                          # stretches harder as it goes
            x = akai(sl, lambda p: 1 + 3 * p, grain_ms=28, n_out=2 * beat)
        else:
            x = akai(sl, ratio, grain_ms=30)[:2 * beat]
        stretch_in(tl, x, b0, 2)
    half = akai(pressed[:int(16 * p16) + int(0.05 * SR)], 2.0, grain_ms=50)[:int(bar_s(2) * SR)]
    stretch_in(tl, half, 63, 8, 0.9)
    crawl = akai(pressed[int(32 * p16):], lambda p: 1 + 4 * p ** 1.5, grain_ms=55, xf_ms=6,
                 n_out=int(bar_s(4) * SR))
    stretch_in(tl, crawl, 127, 16)

    # RAMP, the way back into drop 2: the break thaws out of an 8x stretch (bars 72-73), then runs at tempo while
    # the sampler steps its pitch up bar by bar with the grains getting shorter and buzzier, and in the last bar
    # the snare is held, stretched into a rising robot buzz, cut dead on beat 4 so drop 2 lands out of silence
    bb = tl.bus["break"]
    for b0, semis, grain in RAMP_STEPS:
        i0, i1 = int(bar_s(b0) * SR), int(bar_s(b0 + 1) * SR)
        bb[i0:i1] = akai_pitch(bb[i0:i1].copy(), semis, grain)
    snare = pressed[int(4 * p16):int(6 * p16)]
    hold = akai(akai_pitch(snare, 12.0, 12), lambda p: 1 + 60 * p, grain_ms=14, xf_ms=2, n_out=3 * beat)
    hold *= np.linspace(0.8, 1.15, len(hold))[:, None]
    stretch_in(tl, hold, 79, 4)
    i0, i1 = tl.i(79 * 4 + 3), tl.i(80 * 4)
    bb[i0 - int(0.004 * SR):i0] *= np.linspace(1, 0, int(0.004 * SR))[:, None]
    bb[i0:i1] = 0
    tl.kicks = [k for k in tl.kicks if not i0 / SR <= k < i1 / SR]
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
    horns, bones, pizz, timp = [], [], [], []                          # a brass-and-timpani hit on the root through
    for b0 in range(80, 111):                                          # drop 2, from the tune's own band
        for at in ([0.0] if b0 % 4 == 0 else []) + ([2.5, 3.25] if b0 % 8 == 6 else []):
            st = b0 * 4 + at
            r = root_at(bass, st)
            if r is None:
                continue
            pc = r % 12
            horns += [N.Note(st, 0.3, 48 + pc + d, 110) for d in (12, 19, 24)]
            bones += [N.Note(st, 0.3, 36 + pc + d, 105) for d in (12, 19)]
            pizz += [N.Note(st, 0.25, 60 + pc + d, 100) for d in (0, 7, 12)]
            timp.append(N.Note(st, 0.5, 36 + pc + (12 if pc < 5 else 0), 120))
    tl.put_full("stab", fluid([(horns, 60, 0, 0), (bones, 57, 0, 1), (pizz, 45, 0, 2), (timp, 47, 0, 3)],
                              BPM, tail=1.5))

    # --- processing -----------------------------------------------------------------------------------------
    b = tl.bus

    def ride(t):        # opens through the intro, dims for the breakdown, closes through the outro
        f = np.full_like(t, 6000.0)
        f = np.where(t < bar_s(16), 800 * (6000 / 800) ** np.clip(t / bar_s(16), 0, 1) ** 1.3, f)
        f = np.where((t > bar_s(64)) & (t < bar_s(80)), 3200.0, f)
        o0, o1 = bar_s(120), bar_s(127.5)
        f = np.where(t > o0, 6000 * (400 / 6000) ** np.clip((t - o0) / (o1 - o0), 0, 1), f)
        return f

    def break_ride(t):  # the build fades the break in from under a lowpass
        f = np.full_like(t, 11000.0)
        a0, a1 = bar_s(8), bar_s(16)
        f = np.where((t >= a0) & (t < a1), 350 * (11000 / 350) ** np.clip((t - a0) / (a1 - a0), 0, 1) ** 2, f)
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
    v = fx.lowpass(fx.highpass(b["vox"], 170, 2), 5200, 2)                 # the oohs: a sampled choir pad
    v = fx.bitcrush(fx.saturate(v, 1.3), 12, 26040, mix=0.6)
    b["vox"] = fx.widen(fx.wow_flutter(v, wow_ms=1.4, seed=34), 1.6)
    og = fx.saturate(fx.highpass(b["organ"], 220, 2), 1.6)               # the organ: its own, brighter bus
    og = fx.lowpass(fx.bitcrush(og, 12, 26040, mix=0.5), 7500, 2)
    b["organ"] = fx.widen(fx.detune_chorus(fx.wow_flutter(og, wow_ms=0.8, seed=35), cents=5, mix=0.3), 1.3)
    c = fx.bitcrush(fx.highpass(b["chops"], 280, 2), 12, 22050)
    b["chops"] = fx.autopan(fx.saturate(c, 1.6), rate=BPM / 60 / 8, depth=0.35, phase=0.25)

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
    sb = fx.bitcrush(fx.highpass(b["stab"], 90, 2), 12, 22050, mix=0.4)    # a sampled brass hit, lightly crunched
    b["stab"] = fx.widen(fx.saturate(sb, 1.3), 1.2)
    b["boom"] = fx.lowpass(b["boom"], 220, 2)

    rz = fx.highpass(b["reese"], 70, 2)
    rz = fx.sweep_lowpass(rz, lambda t: 650 * 2 ** (0.9 * np.sin(2 * np.pi * t / bar_s(4))))
    b["reese"] = fx.mono_below(fx.compress(fx.saturate(rz, 2.2), -18, 3, 5, 80), 160)
    b["sub"] = fx.lowpass(b["sub"], 100, 2)

    for k, target in LEVELS.items():
        b[k] = fx.level(b[k], target)
    l0, l1, lift = ORGAN_LIFT
    tt = np.arange(n) / SR
    up = np.interp(tt, [bar_s(l0) - 0.3, bar_s(l0), bar_s(l1), bar_s(l1) + 0.3], [1, 10 ** (lift / 20)] * 2)
    b["organ"] = b["organ"] * np.where((tt > bar_s(l0) - 0.3) & (tt < bar_s(l1) + 0.3), up, 1.0)[:, None]

    send = (b["sample"] * 0.35 + b["osti"] * 0.3 + b["strings"] * 0.6 + b["break"] * 0.08 + b["crash"] * 0.3
            + b["stab"] * 0.4 + b["chops"] * 0.5 + b["organ"] * 0.4)
    verb = fx.reverb(send, seconds=2.2, damp_hz=3600, predelay_ms=18)
    verb = verb + fx.reverb(b["vox"], seconds=4.5, damp_hz=4200, predelay_ms=40, seed=48) * 0.9   # the cathedral

    music = (b["sample"] + b["osti"] + b["strings"] + b["vox"] + b["chops"] + b["organ"]) * fx.duck(n, tl.kicks, depth_db=2.0, release_ms=140)[:, None]
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
