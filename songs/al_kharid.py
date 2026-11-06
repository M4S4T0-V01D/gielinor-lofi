"""Gielinor Lo-Fi - Al Kharid (after Ian Taylor's 'Al Kharid', Old School RuneScape).

The toll gate at noon: desert lo-fi hip-hop at 74 bpm in Eb hijaz, turning to C hijaz halfway. The 140 bpm MIDI
is pressed half-time (each source bar is two song beats) and varispeeded down a semitone. The pizzicato ostinato and
timpani stay as the record; the rest is re-voiced: the ocarina tune as a bowed rebab (fiddle), the dulcimer as a saz
(sitar), the koto counterline and the orchestra hits as a harp, the marimba motif as a kalimba. New parts: a
chorused DX piano, a synth bass and sub, a rebab doubling the saz in A2, and a kit on the CM-64 bank with a darbuka
layer that changes rhythm each section (saidi in A, baladi in B, ayoub in C, a bare doum in D) under a tambourine
on the 16ths. The move: the band stops dead at the end of B1, ten coins are counted onto the toll-booth counter and
the wooden gate creaks open onto the key change; at the end it swings shut with a thud. The desert: a scimitar is
drawn into A2, pickaxes ping from the Al Kharid mine across the breakdown, and a hot wind blows throughout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scipy.signal import lfilter  # noqa: E402

from engine import ambience as AMB, fx, master as M, notes as N, parts as PT  # noqa: E402
from engine.render import SR, fluid  # noqa: E402
from engine.sampler import Record, Timeline, edge  # noqa: E402

TITLE = "Al Kharid (The Toll Gate)"
FILE = "Al Kharid (The Toll Gate)"
BPM = 74.0
KEY = -1                      # E hijaz -> Eb hijaz (C# -> C in the second half)
KIT = 127                     # GeneralUser GS CM-64/32L kit
SWING = 0.56
SRC = N.load(ROOT / "source" / "al_kharid.mid")
DOUM, TEK, KA, CRASH = 64, 60, 61, 49     # low conga as the doum, hi / low bongo as tek and ka

PT.VOICINGS["A7"] = PT.v("A2", "G3", "C#4", "E4")
PT.VOICINGS["C#7b9"] = PT.v("C#3", "F3", "B3", "D4")
PT.VOICINGS["C#5"] = PT.v("C#3", "G#3", "C#4", "G#4")

E, AA, CS, FS = [("E7b9", 4)], [("A7", 4)], [("C#7b9", 4)], [("F#7", 4)]
HIJ_E = [E] * 4 + [AA] * 2 + [E] * 2
HIJ_C = [CS] * 4 + [FS] * 2 + [CS] * 2

# name, first song bar, bars, source bar (two source bars per song bar)
FORM = [("intro", 0, 4, 8), ("A1", 4, 8, 16), ("B1", 12, 8, 32), ("C", 20, 8, 48), ("D", 28, 8, 64),
        ("A2", 36, 8, 80), ("outro", 44, 4, 96)]
BARS = 48
CHART = [E] * 4 + HIJ_E + HIJ_E + HIJ_C + [[("C#5", 4)]] * 8 + HIJ_C + [CS] * 4
LEVELS = {"sample": -21, "pad": -31, "rebab": -25, "saz": -25, "harp": -27, "kal": -28, "keys": -29,
          "perc": -28, "bass": -27, "sub": -33, "kick": -22, "snare": -26, "hats": -36, "darb": -27, "tamb": -34}
GATE = (19 * 4 + 1, 20 * 4)      # the band stops on beat 2 of bar 19; the gate opens onto bar 20
DOWN = (28 * 4, 36 * 4)          # the breakdown

# Al Kharid's groove: a darbuka saidi (doum on 1, the a of 2 and 3, tek on the e of 1 and on 4) with the kick
# following the doums, a snare on 2 and 4 and a tambourine on the 16ths, leaning on each beat. B is a baladi (two
# doums to open), C an ayoub (doum on every beat, the tek a 16th ahead of the next), D just the doum.
TAMB16 = dict(bus="tamb", piece=PT.TAMB, at=list(range(16)), vel=(20, 32), accent={0: 56, 4: 54, 8: 56, 12: 54},
              p=0.75)
SAIDI = [
    dict(bus="kick", piece=PT.KICK, at=[0, 6, 8], odd=[0, 6, 8, 11], fill=[0, 6, 8, 10, 11], vel=(86, 98),
         accent={0: 110}),
    dict(bus="snare", piece=PT.SNARE, at=[4, 12], vel=94),
    dict(bus="darb", piece=DOUM, at=[0, 6, 8], vel=(80, 92)),
    dict(bus="darb", piece=TEK, at=[2, 12], fill=[2, 12, 13, 14, 15], vel=(70, 86)),
    dict(bus="darb", piece=KA, at=[4, 10, 14], vel=(32, 48), p=0.6),
    TAMB16,
]
BALADI = [
    dict(bus="kick", piece=PT.KICK, at=[0, 2, 8], odd=[0, 2, 8, 11], vel=(84, 96), accent={0: 106, 2: 76}),
    dict(bus="snare", piece=PT.CLAP, at=[4, 12], vel=(72, 84)),
    dict(bus="darb", piece=DOUM, at=[0, 2, 8], vel=(80, 92)),
    dict(bus="darb", piece=TEK, at=[6, 12], fill=[6, 12, 14, 15], vel=(72, 86)),
    dict(bus="darb", piece=KA, at=[4, 10, 14], vel=(30, 46), p=0.5),
    dict(bus="tamb", piece=PT.TAMB, at=[4, 12], vel=(60, 72)),
    dict(bus="tamb", piece=PT.TAMB, at=[2, 6, 10, 14], vel=(24, 32), p=0.6),
]
AYOUB = [
    dict(bus="kick", piece=PT.KICK, at=[0, 8], odd=[0, 8, 11], vel=(86, 98), accent={0: 108}),
    dict(bus="snare", piece=PT.SNARE, at=[4, 12], vel=92),
    dict(bus="darb", piece=DOUM, at=[0, 4, 8, 12], vel=(76, 90)),
    dict(bus="darb", piece=TEK, at=[3, 6, 11, 14], fill=[3, 6, 11, 13, 14, 15], vel=(68, 84)),
    dict(bus="darb", piece=KA, at=[1, 9], vel=(28, 40), p=0.4),
    TAMB16,
]
BARE = [
    dict(bus="darb", piece=DOUM, at=[0], odd=[0, 8], vel=(70, 80)),
    dict(bus="darb", piece=KA, at=[6, 12], vel=(28, 40), p=0.5),
    dict(bus="tamb", piece=PT.TAMB, at=[4, 12], vel=(40, 52)),
]
# A2: everything, plus a clap on the 4 and an open hat into each odd bar
BIG = SAIDI + [dict(bus="snare", piece=PT.CLAP, at=[12], vel=58),
               dict(bus="hats", piece=PT.OHH, at=[], odd=[14], vel=(44, 52), dur=0.4)]
GROOVE = {"A1": SAIDI, "B1": BALADI, "C": AYOUB, "D": BARE, "A2": BIG}


def bar_s(b):
    return b * 4 * 60 / BPM


def place_bars(tl, bus, rec, src_bar, song_bar, nbars, gain=1.0):
    """Lay half-time record bars: each song bar is two source bars (four record beats)."""
    for k in range(nbars):
        x = rec.beats(src_bar * 2 + k * 4, src_bar * 2 + k * 4 + 4, tail_beats=0.2)
        tl.put(bus, edge(x, 2, 25), (song_bar + k) * 4, gain)


def from_src(notes, src_bar, song_beat):
    """Source notes from src_bar on, laid half-time from song_beat."""
    return [N.Note(song_beat + (x.start - src_bar * 4) / 2, x.dur / 2, x.pitch, x.vel) for x in notes]


def resonate(x, bands):
    """A bank of two-pole resonators: [(hz, bandwidth hz, gain)] summed (the body of wood or steel)."""
    y = np.zeros_like(x)
    for f, bw, g in bands:
        r = np.exp(-np.pi * bw / SR)
        y += lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / SR), r * r], x) * g
    return y


def stereo(y, p):
    return np.stack([y * np.sqrt((1 - p) / 2), y * np.sqrt((1 + p) / 2)], axis=1).astype(np.float32)


def ring(rng, f0, ratios, decays, seconds, attack_ms=0.5):
    """An inharmonic metal ring: partials at f0 * ratios, each dying away on its own decay."""
    t = np.arange(int(seconds * SR)) / SR
    y = sum(np.sin(2 * np.pi * f0 * r * t + rng.uniform(0, 6)) * np.exp(-t / d) / (1 + k)
            for k, (r, d) in enumerate(zip(ratios, decays)))
    return y * (1 - np.exp(-t / (attack_ms / 1000)))


def add(buf, y, t0):
    i = int(t0 * SR)
    if i < 0:
        y, i = y[-i:], 0
    m = min(len(y), len(buf) - i)
    if m > 0:
        buf[i:i + m] += y[:m]


def coins(seconds, level_db, t0, t1, count=10, seed=80):
    """Ten coins counted onto a wooden counter between t0 and t1: each lands with a bright ring and a tick of wood,
    bounces once or twice, and the last one spins down, rattling faster and faster until it lies flat."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = np.zeros((n, 2), dtype=np.float32)
    times = np.sort(t0 + (t1 - t0) * (np.arange(count) + rng.uniform(-0.3, 0.3, count)) / count)
    tick = fx.bandpass(rng.standard_normal((int(0.015 * SR), 1)), 600, 2400, 2)[:, 0]
    tick *= np.exp(-np.arange(len(tick)) / (0.003 * SR))
    for k, tc in enumerate(times):
        f0, p = rng.uniform(2600, 3900), rng.uniform(-0.35, 0.35)
        hits = [(0.0, 1.0)] + [(rng.uniform(0.04, 0.07) * j, 0.45 / j) for j in (1, 2) if rng.random() < 0.8]
        if k == count - 1:          # the spin: a rattle whose period shrinks from 70 ms to 6 ms
            dt, at, a = 0.07, 0.12, 0.5
            while dt > 0.006:
                hits.append((at, a))
                at, dt, a = at + dt, dt * 0.88, a * 0.95
        for off, a in hits:
            y = ring(rng, f0, [1.0, 1.47, 2.09, 2.56, 3.2], [0.22, 0.14, 0.1, 0.07, 0.05], 0.5) * a
            y[:len(tick)] += tick * a * 2.5
            add(x, stereo(y, p), tc + off)
    x = fx.lowpass(fx.highpass(x, 500, 2), 9000, 2)
    x = x * 0.7 + fx.reverb(x, seconds=0.9, damp_hz=5000, predelay_ms=8, hp=500, seed=seed) * 0.5
    return fx.level(x, level_db)


def creak(dur, f_from, f_to, rng):
    """A dry wooden hinge: stick-slip, a train of tiny snaps whose rate glides f_from -> f_to (with the jitter of
    wood catching), rung through the resonances of a heavy plank door."""
    m = int(dur * SR)
    imp = np.zeros(m)
    t, prog = 0.0, 0.0
    while t < dur:
        prog = t / dur
        f = f_from * (f_to / f_from) ** prog * rng.uniform(0.85, 1.15)
        i = int(t * SR)
        imp[i] += rng.uniform(0.5, 1.0) * (0.3 if rng.random() < 0.06 else 1.0)
        t += 1 / f
    env = np.minimum(1, np.arange(m) / (0.08 * SR)) * np.minimum(1, (m - np.arange(m)) / (0.15 * SR))
    body = resonate(imp, [(310, 60, 1.0), (720, 90, 0.8), (1280, 140, 0.6), (2350, 220, 0.35)])
    return body * env * (0.8 + 0.2 * np.sin(2 * np.pi * 3.1 * np.arange(m) / SR))


def thud(rng, seconds=0.8):
    """The gate's weight landing in its frame: a dull low knock and a short rattle of the planks."""
    t = np.arange(int(seconds * SR)) / SR
    knock = np.sin(2 * np.pi * (48 * t + 30 * (1 - np.exp(-t / 0.03)) * 0.03)) * np.exp(-t / 0.22)
    noise = np.zeros(len(t))
    noise[:int(0.03 * SR)] = rng.standard_normal(int(0.03 * SR))
    planks = resonate(noise, [(180, 40, 1.0), (420, 70, 0.7), (950, 120, 0.4)])
    return knock * 0.9 + planks * 0.6 * np.exp(-t / 0.12)


def toll_gate(seconds, level_db, opens_at, closes_at, seed=81):
    """The gate swinging open (rising creak, then a short settle) at opens_at and shut (falling creak, a thud) at
    closes_at, both seconds; the creak ends exactly on the time given."""
    rng = np.random.default_rng(seed)
    x = np.zeros((int(seconds * SR), 2), dtype=np.float32)
    y = np.concatenate([creak(1.1, 38, 150, rng), creak(0.35, 140, 95, rng) * 0.6])
    add(x, stereo(y, 0.25), opens_at - 1.45)
    y = np.concatenate([creak(1.3, 120, 45, rng), thud(rng)])
    add(x, stereo(y, 0.25), closes_at - 1.3)
    x = fx.lowpass(fx.highpass(x, 70, 2), 6000, 2)
    x = x * 0.75 + fx.reverb(x, seconds=1.2, damp_hz=3000, predelay_ms=12, hp=120, seed=seed) * 0.45
    return fx.level(x, level_db)


def scimitar(seconds, level_db, at, seed=82):
    """A scimitar drawn from its scabbard, hitting at `at` (seconds): a steel scrape that rises along the blade, then
    the tip clears the throat with a shimmering ring (two detuned copies beating), a whoosh behind it."""
    rng = np.random.default_rng(seed)
    x = np.zeros((int(seconds * SR), 2), dtype=np.float32)
    m = int(0.55 * SR)
    tt = np.arange(m) / SR
    scrape = fx.bandpass(rng.standard_normal((m, 1)), 2200, 7000, 2)[:, 0]
    scrape *= (tt / tt[-1]) ** 1.8 * (0.7 + 0.3 * np.sin(2 * np.pi * 37 * tt))
    scrape = scrape + resonate(scrape, [(3150, 80, 0.6), (4700, 120, 0.4)])
    add(x, stereo(scrape, -0.1), at - 0.55)
    shing = sum(ring(rng, 2150 * dt, [1.0, 1.56, 2.41, 3.2], [1.4, 0.9, 0.6, 0.4], 2.5, attack_ms=3)
                for dt in (1.0, 1.004))
    add(x, stereo(shing * 0.5, 0.1), at)
    w = int(0.4 * SR)
    whoosh = rng.standard_normal(w) * np.sin(np.pi * np.arange(w) / w) ** 2
    whoosh = fx.sweep_lowpass(np.repeat(whoosh[:, None], 2, axis=1).astype(np.float32),
                              lambda s: 600 + 3400 * np.clip(s / 0.4, 0, 1))[:, 0]
    add(x, stereo(whoosh * 0.35, 0.4), at + 0.05)
    x = fx.lowpass(fx.highpass(x, 300, 2), 10000, 2)
    x = x * 0.7 + fx.reverb(x, seconds=1.6, damp_hz=6000, predelay_ms=20, hp=600, seed=seed) * 0.5
    return fx.level(x, level_db)


def mine(seconds, level_db, t0, t1, seed=83):
    """The Al Kharid mine across the sand: pickaxes on rock, a steel ping and a crack, in twos and threes from two
    miners, far off and smeared by distance."""
    rng = np.random.default_rng(seed)
    x = np.zeros((int(seconds * SR), 2), dtype=np.float32)
    for f0, p, t in ((1850.0, 0.6, t0 + 0.4), (2240.0, -0.45, t0 + 2.1)):
        while t < t1:
            for j in range(int(rng.integers(2, 4))):
                y = ring(rng, f0 * rng.uniform(0.98, 1.02), [1.0, 1.38, 2.27], [0.18, 0.1, 0.06], 0.4)
                crack = np.zeros(len(y))
                crack[:int(0.02 * SR)] = fx.bandpass(rng.standard_normal((int(0.02 * SR), 1)), 900, 3500, 2)[:, 0]
                add(x, stereo((y + crack * 1.5) * rng.uniform(0.6, 1.0), p), t + j * rng.uniform(0.7, 0.95))
            t += rng.uniform(4.0, 7.0)
    x = fx.lowpass(fx.highpass(x, 600, 2), 4000, 2)
    x = x * 0.35 + fx.reverb(x, seconds=2.8, damp_hz=2500, predelay_ms=60, hp=600, seed=seed) * 0.9
    return fx.level(x, level_db)


def build(out_dir: Path, stems: bool = False) -> dict:
    tl = Timeline(BPM, BARS * 4, tail_s=7.0)
    n = tl.n

    # --- the record (half-time) -----------------------------------------------------------------------------
    rec = dict(bpm=BPM, semis=KEY, dst_bar=2, curve=None, swing=SWING)
    loop = Record(SRC, ["pizz", "timp"], vel={"timp": 0.75}, **rec)
    pad = Record(SRC, ["slow string"], **rec)
    rebab = Record(SRC, ["ocarina"], programs={"ocarina": 110}, **rec)              # fiddle as a rebab
    saz = Record(SRC, ["dulcimer"], programs={"dulcimer": 104}, **rec)               # sitar as a saz
    harp = Record(SRC, ["koto", "hit"], programs={"koto": 46, "hit": 46}, vel={"hit": 0.85}, **rec)
    kal = Record(SRC, ["Track 9"], programs={"Track 9": 108}, semis=KEY - 12,      # the marimba as a kalimba
                 **{k: v for k, v in rec.items() if k != "semis"})
    perc = Record(SRC, ["perc"], **rec)                                              # the source's own drums

    for name, b0, nb, sb in FORM:
        place_bars(tl, "sample", loop, sb, b0, nb, {"intro": 0.9, "D": 0.8}.get(name, 1.0))
        place_bars(tl, "pad", pad, sb, b0, nb)
        if name in ("A1", "B1", "D"):
            place_bars(tl, "rebab", rebab, sb, b0, nb)
        if name in ("A1", "B1", "A2"):
            place_bars(tl, "saz", saz, sb, b0, nb)
        if name in ("A1", "B1", "C", "A2"):
            place_bars(tl, "harp", harp, sb, b0, nb)
        if name == "C":
            place_bars(tl, "kal", kal, sb, b0, nb)
        if name in ("C", "A2"):
            place_bars(tl, "perc", perc, sb, b0, nb, 0.6)

    # --- new parts ------------------------------------------------------------------------------------------
    def open_gate(ns):
        return [x for x in ns if not (GATE[0] <= x.start < GATE[1])]

    keys = PT.comp([E] * 2, start_bar=2, vel=36, style="pad", bpm=BPM)
    keys += PT.comp(CHART[4:28], start_bar=4, vel=46, seed=6, bpm=BPM)
    keys += PT.comp(CHART[28:36], start_bar=28, vel=34, style="pad", bpm=BPM)
    keys += PT.comp(CHART[36:44], start_bar=36, vel=48, seed=7, bpm=BPM)
    keys += [N.Note(x.start, x.dur + 6, x.pitch, x.vel) for x in PT.comp([CS], 44, 38, "pad", bpm=BPM)]
    keys = open_gate(keys)
    tl.put_full("keys", fluid([(N.shift(N.swing(keys, SWING), 0, KEY), 5, 0, 0)], BPM, tail=7))       # DX piano

    theme = [x for x in SRC.tracks["dulcimer"] if 80 * 4 <= x.start < 96 * 4]           # the rebab joins the saz
    dbl = N.humanize(PT.melody_double(from_src(theme, 80, 36 * 4), octave=-1, vel=64), BPM, 5, 4, seed=10)
    tl.put_full("rebab", fluid([(N.swing(N.shift(dbl, 0, KEY), SWING), 110, 0, 1)], BPM, tail=4))

    bass = PT.bass(CHART[4:28], start_bar=4, vel=84, seed=8) + PT.bass(CHART[36:44], start_bar=36, vel=86, seed=9)
    bass += [N.Note(b * 4, 7.5, PT.P["C#2"], 72) for b in range(28, 36, 2)]           # the breakdown's drone
    bass += [N.Note(44 * 4, 7.5, PT.P["C#2"], 76), N.Note(46 * 4, 6, PT.P["C#2"], 68)]
    bass = N.shift(N.humanize(N.swing(open_gate(bass), SWING), BPM, 6, 4, seed=11, push_ms=4), 0, KEY)
    tl.put_full("bass", fluid([(bass, 38, 0, 2)], BPM, tail=4))                         # synth bass 1
    tl.put_full("sub", PT.sub(bass, BPM, n))

    drums = {"kick": [], "snare": [], "hats": [], "darb": [], "tamb": []}
    for name, b0, nb, _ in FORM:
        g = GROOVE.get(name)
        if name == "outro":
            g, nb = [x for x in SAIDI if x["bus"] in ("darb", "tamb")], 2
        if not g:
            continue
        for piece, ns in PT.groove(nb, b0, g, seed=50 + b0, swing_amt=SWING, bpm=BPM).items():
            drums[piece] += ns
    drums = {k: open_gate(ns) for k, ns in drums.items()}
    drums["hats"] += [N.Note(GATE[1], 2.0, CRASH, 96), N.Note(DOWN[1], 2.0, CRASH, 90)]
    tl.kicks = [x.start * 60 / BPM for x in drums["kick"]]
    for piece, ns in drums.items():
        tl.put_full(piece, fluid([(ns, KIT, 128, 9)], BPM, tail=2.5))

    # --- processing -----------------------------------------------------------------------------------------
    b = tl.bus

    def ride(t):        # record filter: opens through the intro, closes through the outro
        i1s, o0, o1 = bar_s(4), bar_s(44), bar_s(47.5)
        f = np.where(t < i1s, 420 * (5200 / 420) ** np.clip(t / i1s, 0, 1) ** 1.3, 5200.0)
        return np.where(t > o0, 5200 * (340 / 5200) ** np.clip((t - o0) / (o1 - o0), 0, 1), f)

    def crush(x, mix=0.6):
        return fx.bitcrush(x, 12, 26040, mix=mix)

    s = crush(fx.highpass(b["sample"], 140, 2), 0.75)
    b["sample"] = fx.wow_flutter(fx.sweep_lowpass(fx.saturate(s, 1.5), ride), wow_ms=1.6, wow_hz=0.5, seed=41)
    b["pad"] = fx.widen(fx.sweep_lowpass(fx.highpass(b["pad"], 200, 2), ride), 1.5)
    rb = fx.saturate(fx.lowpass(fx.highpass(b["rebab"], 260, 2), 3400, 2), 1.3)            # a rough bowed gut string
    rb = rb - 0.5 * fx.bandpass(rb, 1800, 5000, 2)                                    # take the rasp off the bow
    b["rebab"] = fx.pan(fx.wow_flutter(crush(rb, 0.5), wow_ms=1.6, wow_hz=0.5, seed=42), -0.15)
    sz = crush(fx.lowpass(fx.highpass(b["saz"], 220, 2), 4200, 2), 0.6)
    sz = sz - 0.4 * fx.bandpass(sz, 2000, 5000, 2)
    b["saz"] = fx.pan(fx.detune_chorus(sz, cents=7, rate=0.4, mix=0.3), 0.25)
    hp = crush(fx.lowpass(fx.highpass(b["harp"], 200, 2), 5000, 2), 0.5)
    b["harp"] = fx.pan(fx.sweep_lowpass(hp, ride), -0.35)
    b["kal"] = fx.pan(crush(fx.lowpass(fx.highpass(b["kal"], 300, 2), 5000, 2), 0.6), 0.4)
    b["perc"] = fx.saturate(crush(fx.lowpass(fx.highpass(b["perc"], 150, 2), 5000, 2), 1.0), 1.5)

    k = fx.detune_chorus(b["keys"], cents=9, rate=0.6, mix=0.45)
    b["keys"] = fx.wow_flutter(fx.saturate(fx.lowpass(fx.highpass(k, 180, 2), 4600, 2), 1.3), seed=43)

    bb = fx.saturate(fx.lowpass(b["bass"], 800, 2), 1.7)
    b["bass"] = fx.mono_below(fx.compress(bb, -20, 3, 12, 140), 300)
    b["sub"] = fx.lowpass(b["sub"], 100, 2)

    kk = fx.saturate(fx.lowpass(b["kick"], 4500, 2), 2.2)
    b["kick"] = fx.compress(kk, -14, 3, 3, 100)
    sn = crush(fx.highpass(b["snare"], 170, 2), 1.0)
    b["snare"] = fx.lowpass(fx.saturate(sn, 1.6), 8000, 2)
    b["hats"] = fx.pan(crush(fx.lowpass(fx.highpass(b["hats"], 700, 2), 9000, 2), 1.0), -0.2)
    b["darb"] = fx.pan(fx.saturate(crush(fx.highpass(b["darb"], 110, 2), 1.0), 1.4), 0.15)
    b["tamb"] = fx.pan(fx.lowpass(fx.highpass(b["tamb"], 1800, 2), 9500, 2), 0.35)

    for key, target in LEVELS.items():
        b[key] = fx.level(b[key], target)

    # the toll: everything stops dead for the coins and the gate (the reverbs ring on)
    t = np.arange(n) / SR
    g0, g1 = bar_s(GATE[0] / 4), bar_s(GATE[1] / 4)
    stop = np.interp(t, [g0 - 0.02, g0 + 0.03, g1 - 0.01, g1], [1, 0, 0, 1])[:, None]
    for key in b:
        b[key] = b[key] * stop

    # space: a plate on the rebab, saz and harp, a room for the rest; an 8th echo on the saz, a dotted-quarter
    # echo on the rebab
    plate = fx.reverb(b["rebab"] * 0.7 + b["saz"] * 0.5 + b["harp"] * 0.6 + b["kal"] * 0.6, seconds=3.0,
                      damp_hz=3400, predelay_ms=30, seed=44)
    room = fx.reverb(b["sample"] * 0.3 + b["keys"] * 0.45 + b["pad"] * 0.5 + b["snare"] * 0.2 + b["darb"] * 0.25
                     + b["perc"] * 0.3 + b["tamb"] * 0.2, seconds=1.5, damp_hz=3600, predelay_ms=12, seed=45)
    echo = fx.tape_delay(b["saz"], 0.5, BPM, feedback=0.3, repeats=3, lp=3000, pingpong=False)
    long = fx.tape_delay(b["rebab"], 1.5, BPM, feedback=0.35, repeats=4, lp=2400)

    duck = fx.duck(n, tl.kicks, depth_db=2.2, release_ms=230)[:, None]
    music = (b["sample"] + b["pad"] + b["rebab"] + b["saz"] + b["harp"] + b["kal"] + b["keys"]) * duck
    low = (b["bass"] + b["sub"]) * fx.duck(n, tl.kicks, depth_db=4.0, release_ms=160)[:, None]
    drum_bus = fx.compress(b["kick"] + b["snare"] + b["hats"] + b["darb"] + b["tamb"], -16, 2.5, 10, 160,
                           makeup_db=1)
    drum_bus = fx.tape(drum_bus, 1.3, 10500) + b["perc"]

    # the desert at noon: a hot wind (up in the breakdown), hiss, and the props
    secs = n / SR
    amt = np.interp(t, [0, bar_s(3.5), bar_s(4.5), bar_s(27.5), bar_s(28.5), bar_s(35.5), bar_s(36.2),
                        bar_s(44), secs], [1.0, 1.0, 0.6, 0.6, 1.1, 1.1, 0.6, 0.6, 1.2])[:, None]
    def fit(x):
        return np.pad(x[:n], ((0, n - len(x[:n])), (0, 0)))

    air = fit(AMB.wind(secs, -43, seed=46)) * amt
    hiss = fit(AMB.hiss(secs, -52, seed=47))

    toll = fit(coins(secs, -27, g0 + 0.1, g1 - 1.25)) + fit(toll_gate(secs, -23, g1, bar_s(47.5)))
    blade = fit(scimitar(secs, -27, bar_s(DOWN[1] / 4)))
    picks = fit(mine(secs, -38, bar_s(28.25), bar_s(35.5)))
    props = toll + blade + picks

    mix = (music + low + drum_bus * 0.85 + plate * 0.4 + room * 0.45 + echo * 0.16 + long * 0.14 + air + hiss
           + props)
    mix = mix + 0.12 * fx.bandpass(mix, 1600, 4800)                  # presence: the bowed and plucked lines forward
    mix = fx.wow_flutter(mix, wow_ms=0.5, wow_hz=0.35, flutter_ms=0.04, drift_ms=0.4, seed=40)
    end = int((bar_s(BARS) + 6.5) * SR)
    mix = fx.fade(mix[:end], fin_s=1.0, fout_s=4.5)
    out = M.master(mix, -14.0, -1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{FILE}.wav"
    M.write(path, out, tags={"title": TITLE, "artist": "Gielinor Lo-Fi", "album": "Gielinor Lo-Fi",
                             "comment": "Lo-fi reinterpretation of 'Al Kharid' (Ian Taylor, Jagex) from Old School RuneScape"})
    if stems:
        sd = out_dir / "stems"
        sd.mkdir(exist_ok=True)
        import soundfile as sf
        for key, x in {**b, "plate": plate, "room": room, "echo": echo, "long": long, "air": air, "toll": toll,
                       "blade": blade, "picks": picks}.items():
            sf.write(str(sd / f"{key}.wav"), x[:end], SR, subtype="PCM_16")
    rep = M.report(out, BPM, [(nm, b0, b0 + nb) for nm, b0, nb, _ in FORM])
    rep["file"] = str(path)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    print(json.dumps(build(ROOT / "build" / "al_kharid", stems="--stems" in sys.argv), indent=1))
