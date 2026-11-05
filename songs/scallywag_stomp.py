"""Gielinor Lo-Fi - The Scallywag Stomp (after Ian Taylor's 'Sea Shanty' and 'Sea Shanty 2', Old School RuneScape).

The whole night at Port Sarim. Sea Shanty (Port Sarim) plays as it is, up to the end of its swaying A4; then the
band lands on D minor, the surf swells, and the tavern door slams on the sea. Inside, the floor stomps a count-in
that speeds up from 76 to 100, and Sea Shanty 2 (4/4, A major, 100 bpm) comes in as a 12/8 jig in D major: the
same record treatment (pressed with its eighths swung into triplets, varispeeded a semitone), Rhodes and bass.
Its own moves: the accelerating stomp count-in, the door shutting out the sea (and opening on it at the end), a
bodhran rolling under the jig, tankards drummed on the table into the bridge, stop-time in the bridge, and two
tankards clashing for the last tune.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from engine import ambience as AMB, fx, master as M, notes as N, parts as PT  # noqa: E402
from engine.render import SR, fluid  # noqa: E402
from engine.sampler import Record, Timeline, edge  # noqa: E402
from songs import sea_shanty as S1  # noqa: E402

TITLE = "The Scallywag Stomp"
FILE = "The Scallywag Stomp"
BPM = 100.0                   # the second half; the first is Sea Shanty at 76
KIT = 32                      # the same jazz kit the band played on the docks
SRC = N.load(ROOT / "source" / "sea_shanty2.mid")


def jt(t):
    """Swing a straight 4/4 position into the jig: each beat's eighths become triplets (2:1), its 16ths fall in."""
    b = math.floor(t)
    f = t - b
    return b + (f * 4 / 3 if f < 0.5 else 2 / 3 + (f - 0.5) * 2 / 3)


def jig(notes, semis=0):
    out = []
    for x in notes:
        s, e = jt(x.start), jt(x.start + x.dur)
        out.append(N.Note(s, max(0.05, e - s), x.pitch + semis, x.vel))
    return out


# pressed a tritone-ish up: +6 in the MIDI and the record varispeeded down a semitone, so A major lands on D major
# with a little tape drag, like the first half's -2.
JIGGED = replace(SRC, tracks={k: jig(v, 0 if k in SRC.drums else 6) for k, v in SRC.tracks.items()})

# source bars (4/4, one song bar each): 0-3 the horn alone, 4-19 the tune with the band (nylon from 11), 20-31 the
# oom-pah without it (piccolo 24-31), 32-39 the accordion's bridge over A7 (bandoneon 36-39), 40-47 the tune with
# marimba, 48-51 the last line over A / E. Chords per source bar, named here in the song's D.
ROOTS = "AAAAAAAAEEAAAAAAEEAA" + "A" * 12 + "C" * 8 + "AAAAEEAA" + "AGAG"
PT.VOICINGS.update({"A9": PT.v("A2", "G3", "B3", "C#4", "E4"), "A13": PT.v("A2", "G3", "C#4", "F#4"),
                    "D13": PT.v("D3", "C4", "F#4", "B4"), "A/C#": PT.v("C#3", "G3", "A3", "E4")})


def chord(s):
    return {"A": "D69" if s % 2 == 0 else "Dadd9", "E": "A9" if s % 2 == 0 else "A13",
            "C": "D13" if s % 2 == 0 else "Dadd9", "G": "A/C#"}[ROOTS[s]]


def chart(s0, nb):
    return [[(chord(s), 4)] for s in range(s0, s0 + nb)]


FORM = [("strike-up", 0, 4), ("jig A", 4, 16), ("oom-pah", 20, 12), ("stop-time", 32, 8), ("last tune", 40, 8),
        ("last orders", 48, 4)]
BARS = 52
LEVELS = {"sample": -21, "horn": -23, "whistle": -27, "pic": -27, "rhodes": -28, "lead": -29, "bass": -26,
          "sub": -32, "kick": -23, "snare": -27, "hats": -38, "bodhran": -31, "stomp": -24, "clap": -29}

# the tavern's own groove, on the 12/8 grid: a bodhran rolling triplets under kick on 1 and 3, a rim backbeat and
# spoons on the last triplet of each beat. Not the docks' stomp-and-tambourine.
BODHRAN = dict(bus="bodhran", piece=PT.LO_FLOOR, at=list(range(12)), odd=[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11],
               accent={0: 92, 3: 74, 6: 86, 9: 74}, vel=(28, 46), p=0.85)
SPOONS = dict(bus="hats", piece=PT.STICK, at=[2, 5, 8, 11], vel=(38, 54), p=0.8)
JIG = [dict(bus="kick", piece=PT.KICK, at=[0, 6], odd=[0, 6, 10], fill=[0, 4, 6, 10], vel=(86, 96)),
       dict(bus="snare", piece=PT.SNARE, at=[3, 9], fill=[3, 9, 10, 11], vel=(84, 98)), BODHRAN, SPOONS]
# the strike-up: feet and hands, stomp-stomp-clap
STOMP = [dict(bus="stomp", piece=PT.KICK, at=[0, 2, 6, 8], accent={0: 104, 6: 98}, vel=(80, 92)),
         dict(bus="stomp", piece=PT.LO_FLOOR, at=[0, 2, 6, 8], vel=(70, 84)),
         dict(bus="clap", piece=PT.CLAP, at=[3, 9], vel=(84, 96))]
OOMPAH = [dict(bus="kick", piece=PT.KICK, at=[0, 6], vel=(84, 94)), dict(BODHRAN, vel=(36, 54), p=0.95),
          dict(bus="clap", piece=PT.CLAP, at=[3, 9], vel=(70, 84)), SPOONS]
STOP = [dict(bus="kick", piece=PT.KICK, at=[0], vel=104), dict(bus="snare", piece=PT.SNARE, at=[0], vel=100),
        dict(bus="stomp", piece=PT.KICK, at=[0], vel=100)]
RETURN = [dict(bus="kick", piece=PT.KICK, at=[0, 6], vel=(84, 94)), dict(BODHRAN, p=0.9), SPOONS]
GROOVE = [("strike-up", 0, 4, STOMP), ("jig A", 4, 16, JIG), ("oom-pah", 20, 12, OOMPAH), ("stop", 32, 4, STOP),
          ("return", 36, 4, RETURN), ("last tune", 40, 8, JIG), ("last orders", 48, 3, JIG)]

# the count-in: 8 beats of stomping that speed up from the first half's 76 to 100, stomp-stomp-clap in the second bar
RAMP = [60 / 76 + (60 / BPM - 60 / 76) * (k + 0.5) / 8 for k in range(8)]
R0 = sum(RAMP)                # seconds from the first stomp to the jig's bar 0
P = R0 * BPM / 60             # ... in song beats


def bar_s(b):
    return R0 + b * 4 * 60 / BPM


def ramp_notes():
    out, t = [], 0.0
    for k, d in enumerate(RAMP):
        v = 76 + 4 * k
        hits = [(0, PT.KICK, v + 10), (0, PT.LO_FLOOR, v)]
        if k >= 4 and k % 2 == 0:
            hits += [(2 / 3, PT.KICK, v), (2 / 3, PT.LO_FLOOR, v - 8)]
        elif k >= 4:
            hits = [(0, PT.CLAP, v + 8)]
        for off, piece, vel in hits:
            out.append(N.Note((t + off * d) * BPM / 60, 0.3, piece, min(120, vel)))
        t += d
    return out


# --- the tavern: synthesized, sparse (a crowd or dense rattle would read as rain) -----------------------------
def _pan(s, p):
    return np.stack([s * np.sqrt((1 - p) / 2), s * np.sqrt((1 + p) / 2)], axis=1)


def clinks(n, times, seed, cheers=()):
    """Pewter and glass: a mug set down or knocked here and there; at each of cheers, two tankards clash."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 2), dtype=np.float32)
    ratios, decays = [1.0, 2.76, 5.4, 8.93], [0.16, 0.08, 0.045, 0.025]
    for t0, big in [(t, False) for t in times] + [(t, True) for t in cheers]:
        for k in range(2 if big else 1):
            f0 = rng.uniform(2300, 3100) if big else rng.uniform(1700, 2600)
            i = int((t0 + k * rng.uniform(0.012, 0.025)) * SR)
            m = min(int(0.9 * SR), n - i)
            if m <= 0:
                continue
            tt = np.arange(m) / SR
            stretch = 2.2 if big else 1.0
            s = sum(np.sin(2 * np.pi * f0 * r * tt + rng.uniform(0, 6)) * np.exp(-tt / (d * stretch)) * w
                    for r, d, w in zip(ratios, decays, [1, 0.5, 0.3, 0.15]))
            s = s * (1 - np.exp(-tt / 0.0008)) * (1.0 if big else rng.uniform(0.3, 0.7))
            x[i:i + m] += _pan(s, (-0.25 + 0.5 * k) if big else rng.uniform(-0.7, 0.7))
    x = fx.highpass(x, 900, 2)
    return x * 0.6 + fx.reverb(x, seconds=1.0, damp_hz=4000, predelay_ms=12, hp=600, seed=seed) * 0.5


def thumps(n, hits, seed):
    """Tankards banged on the table: a wooden knock, and the mugs jumping. hits: (seconds, 0..1)."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 2), dtype=np.float32)
    for t0, a in hits:
        i = int(t0 * SR)
        m = min(int(0.4 * SR), n - i)
        if m <= 0:
            continue
        tt = np.arange(m) / SR
        f = 70 + 80 * np.exp(-tt / 0.03)
        body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.08)
        knock = rng.standard_normal(m) * np.exp(-tt / 0.012)
        s = body + 0.5 * fx.lowpass(knock[:, None].repeat(2, 1), 1100, 2)[:, 0]
        j = int(rng.uniform(0.008, 0.02) * SR)
        f0 = rng.uniform(1300, 1800)
        rattle = np.sin(2 * np.pi * f0 * tt) * np.exp(-tt / 0.05) + 0.4 * np.sin(2 * np.pi * f0 * 2.76 * tt) * np.exp(-tt / 0.03)
        s[j:] += 0.18 * rattle[:m - j]
        x[i:i + m] += _pan(s * a * rng.uniform(0.85, 1.0), rng.uniform(-0.3, 0.3))
    return x + fx.reverb(x, seconds=0.9, damp_hz=2500, predelay_ms=10, hp=150, seed=seed) * 0.4


def door(n, t0, shut, seed):
    """The tavern door: a slam with the latch dropping, or the latch lifting and a creak as it swings open."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=np.float32)

    def add(s, at):
        i = int(at * SR)
        m = min(len(s), n - i)
        if m > 0:
            x[i:i + m] += s[:m]

    def click(m_s, hp):
        m = int(m_s * SR)
        c = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.0015 * SR))
        return fx.highpass(c[:, None].repeat(2, 1), hp, 2)[:, 0]
    if shut:
        m = int(0.8 * SR)
        tt = np.arange(m) / SR
        f = 48 + 40 * np.exp(-tt / 0.04)
        slam = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.16)
        slam += 0.6 * fx.lowpass((rng.standard_normal(m) * np.exp(-tt / 0.05))[:, None].repeat(2, 1), 500, 2)[:, 0]
        add(slam, t0)
        add(click(0.02, 1800) * 0.35, t0 + 0.004)
        add(click(0.02, 2200) * 0.2, t0 + 0.16)            # the latch drops
    else:
        add(click(0.02, 2000) * 0.25, t0)
        m = int(1.1 * SR)
        rate = 55 + 35 * np.sin(np.linspace(0, 2.6, m)) + 12 * rng.standard_normal(m).cumsum() / np.sqrt(m)
        ph = np.cumsum(np.maximum(rate, 20)) / SR
        pulses = (np.diff(np.floor(ph), prepend=0) > 0).astype(np.float32) * rng.uniform(0.6, 1.0, m)
        env = np.sin(np.linspace(0, np.pi, m)) ** 0.7
        creak = fx.bandpass(pulses[:, None].repeat(2, 1), 650, 1700, order=2)[:, 0] * env * 3.0
        add(creak, t0 + 0.12)
    s = _pan(x, 0.15)
    return s + fx.reverb(s, seconds=1.2, damp_hz=2200, predelay_ms=8, hp=80, seed=seed) * 0.5


def place(tl, bus, rec, b0, nb, gain=1.0):
    for k in range(nb):
        tl.put(bus, edge(rec.bar(b0 + k, tail_beats=0.2), 2, 25), P + (b0 + k) * 4, gain)


def jig_half() -> dict:
    """Sea Shanty 2, from the first stomp (t=0) to the end."""
    tl = Timeline(BPM, P + (BARS + 3) * 4, tail_s=6.0)
    n = tl.n

    # --- the record -----------------------------------------------------------------------------------------
    rec = dict(semis=-1, dst_bar=4, curve=None)
    horn = Record(JIGGED, ["horn"], BPM, **rec)
    whistle = Record(JIGGED, ["horn"], BPM, programs={"horn": 78}, vel={"horn": 0.85}, **rec)
    guitar = Record(JIGGED, ["steel guitar", "nylon"], BPM, **rec)
    squeeze = Record(JIGGED, ["accordion", "bandeon"], BPM, **rec)
    pic = Record(JIGGED, ["piccolo", "marimba"], BPM, **rec)

    place(tl, "horn", horn, 0, 4, 0.9)                                  # strike-up: the horn alone over the floor
    for b0, nb in ((4, 16), (20, 12), (40, 8), (48, 3)):
        place(tl, "sample", guitar, b0, nb)
        place(tl, "sample", squeeze, b0, nb)
    place(tl, "horn", horn, 4, 16)                                      # jig A, the whistle joins on its repeat
    place(tl, "whistle", whistle, 12, 8)
    place(tl, "pic", pic, 24, 8)                                        # oom-pah: the piccolo's tune
    place(tl, "horn", horn, 31, 1, 0.8)
    place(tl, "sample", squeeze, 32, 8)                                 # stop-time: the accordion runs on alone
    for k in range(4):                                                  # ... the guitar only hits the one
        tl.put("sample", edge(guitar.beats((32 + k) * 4, (32 + k) * 4 + 0.55), 2, 60), P + (32 + k) * 4)
    place(tl, "sample", guitar, 36, 4)
    place(tl, "horn", horn, 40, 8)                                      # the last tune, marimba under it
    place(tl, "pic", pic, 40, 8)
    place(tl, "horn", horn, 48, 4)                                      # last orders
    for r in (guitar, squeeze):
        tl.put("sample", edge(r.beats(51 * 4, 51 * 4 + 3), 2, 40), P + 51 * 4)

    # --- new parts ------------------------------------------------------------------------------------------
    end = 51 * 4 + 3                                                    # the final hit: bar 51, beat 4
    rhodes = PT.comp(chart(4, 16), start_bar=4, vel=54, seed=31, bpm=BPM)
    rhodes += PT.comp(chart(20, 12), start_bar=20, vel=44, style="pad", bpm=BPM)     # room for the piccolo
    rhodes += [x for x in PT.comp(chart(32, 4), start_bar=32, vel=60, bpm=BPM) if x.start % 4 < 0.3]
    rhodes += PT.comp(chart(36, 15), start_bar=36, vel=54, seed=32, bpm=BPM)
    rhodes = jig(rhodes)
    rhodes = [replace(x, dur=0.5) if 32 * 4 <= x.start < 36 * 4 else x for x in rhodes]   # stabs on the one
    rhodes += [N.Note(end + 0.01 * j, 5, p, 58) for j, p in enumerate(PT.VOICINGS["D69"])]
    tune = N.shift(N.top_line(N.window(jig(SRC.tracks["horn"], 5), 40, 48)), 40 * 4)
    lead = PT.melody_double(tune, vel=46)                                           # the last tune, doubled
    tl.put_full("rhodes", fluid([(N.shift(rhodes, P), 4, 0, 0)], BPM, tail=8))
    tl.put_full("lead", fluid([(N.shift(lead, P), 4, 0, 1)], BPM, tail=4))

    # oom-pah walking bass: root on 1, the fifth below on 3, a triplet pickup into each change
    low = {"A": PT.P["D2"], "E": PT.P["A1"], "C": PT.P["D2"], "G": PT.P["C#2"]}
    bass, rng = [], np.random.default_rng(33)
    for s in range(4, 51):
        r, base = low[ROOTS[s]], s * 4
        if 32 <= s < 36:
            bass.append(N.Note(base, 1.2, r, 92))
            continue
        bass += [N.Note(base, 0.9, r, int(rng.integers(84, 94))), N.Note(base + 2, 0.9, r - 5, int(rng.integers(76, 86)))]
        nxt = low[ROOTS[s + 1]]
        if nxt != r:
            bass.append(N.Note(base + 3 + 2 / 3, 0.3, nxt - 1, 74))
        elif s % 2:
            bass.append(N.Note(base + 3 + 2 / 3, 0.3, r + 12 if r < PT.P["D2"] else r - 2, 70))
    bass += [N.Note(51 * 4, 0.9, PT.P["D2"], 88), N.Note(51 * 4 + 2, 0.9, PT.P["C#2"], 80),
             N.Note(end, 4, PT.P["D2"], 96)]
    bass = N.shift(N.humanize(bass, BPM, 6, 4, seed=34, push_ms=4), P)
    tl.put_full("bass", fluid([(bass, 33, 0, 2)], BPM, tail=4))
    tl.put_full("sub", PT.sub(bass, BPM, n))

    drums = {"kick": [], "snare": [], "hats": [], "bodhran": [], "stomp": [], "clap": []}
    for _, b0, nb, feel in GROOVE:
        for piece, ns in PT.groove(nb, b0, feel, seed=60 + b0, steps=12, bpm=BPM).items():
            drums[piece] += ns
    drums["kick"] += [N.Note(end, 0.5, PT.KICK, 110)]
    drums["snare"] += [N.Note(end, 0.5, PT.SNARE, 104)]
    drums["stomp"] += [N.Note(end, 0.5, PT.KICK, 104)]
    drums = {k: N.shift(v, P) for k, v in drums.items()}
    drums["stomp"] += [x for x in ramp_notes() if x.pitch != PT.CLAP]
    drums["clap"] += [x for x in ramp_notes() if x.pitch == PT.CLAP]
    tl.kicks = [x.start * 60 / BPM for x in drums["kick"]]
    for piece, ns in drums.items():
        tl.put_full(piece, fluid([(ns, KIT, 128, 9)], BPM, tail=2))

    # --- processing (the first half's chains, in a smaller, drier room) -----------------------------------
    b = tl.bus

    def corner(t):      # the horn starts it from the corner of the room and steps up when the band comes in
        return np.where(t < bar_s(4), 2200 * (6000 / 2200) ** np.clip((t - bar_s(3)) / (bar_s(4) - bar_s(3)), 0, 1),
                        6000.0)

    s = fx.highpass(b["sample"], 140, 2)
    s = fx.bitcrush(s, 12, 26040, mix=0.85)
    s = fx.saturate(s, 1.6)
    b["sample"] = fx.wow_flutter(fx.lowpass(s, 5000, 2), seed=42) * 1.1
    h = fx.bitcrush(fx.highpass(b["horn"], 220, 2), 12, 26040, mix=0.7)
    b["horn"] = fx.wow_flutter(fx.sweep_lowpass(fx.saturate(h, 1.4), corner), seed=43)
    for k, sd in (("whistle", 44), ("pic", 45)):
        w = fx.bitcrush(fx.highpass(b[k], 300, 2), 12, 26040, mix=0.6)
        b[k] = fx.wow_flutter(fx.lowpass(fx.saturate(w, 1.3), 5200, 2), seed=sd) * 0.75
    b["whistle"], b["pic"] = fx.pan(b["whistle"], 0.3), fx.pan(b["pic"], -0.3)

    r = fx.detune_chorus(b["rhodes"], cents=7, mix=0.3)
    r = fx.saturate(fx.lowpass(fx.highpass(r, 190, 2), 5500, 2), 1.4)
    b["rhodes"] = fx.wow_flutter(r, wow_ms=1.2, seed=46) * 0.75
    b["lead"] = fx.wow_flutter(fx.saturate(fx.lowpass(b["lead"], 4200, 2), 1.3), seed=47) * 0.6

    bs = fx.saturate(fx.lowpass(b["bass"], 650, 2), 1.8)
    b["bass"] = fx.mono_below(fx.compress(bs, -20, 3, 15, 120), 300) * 0.9
    b["sub"] = fx.lowpass(b["sub"], 110, 2) * 0.8

    kk = fx.saturate(fx.lowpass(b["kick"], 4500, 2), 2.0)
    b["kick"] = fx.compress(kk, -14, 3, 3, 90) * 1.1
    sn = fx.bitcrush(fx.highpass(b["snare"], 170, 2), 12, 26040)
    b["snare"] = fx.lowpass(fx.saturate(sn, 1.7), 8000, 2) * 0.9
    b["hats"] = fx.pan(fx.lowpass(fx.highpass(b["hats"], 400, 2), 6000, 2), 0.25)
    bd = fx.saturate(fx.lowpass(fx.highpass(b["bodhran"], 70, 2), 2200, 2), 1.5)     # skin, not tom
    b["bodhran"] = fx.pan(bd, -0.2)
    st = fx.lowpass(fx.highpass(b["stomp"], 45, 2), 900, 2)                           # boots on boards
    b["stomp"] = fx.saturate(st, 1.8)
    b["clap"] = fx.lowpass(fx.highpass(b["clap"], 300, 2), 6000, 2)

    for k, target in LEVELS.items():
        b[k] = fx.level(b[k], target)

    send = (b["sample"] * 0.3 + b["horn"] * 0.4 + b["whistle"] * 0.5 + b["pic"] * 0.5 + b["rhodes"] * 0.45
            + b["snare"] * 0.3 + b["stomp"] * 0.5 + b["clap"] * 0.6 + b["lead"] * 0.4)
    verb = fx.reverb(send, seconds=1.6, damp_hz=3000, predelay_ms=14, seed=48)
    slap = fx.tape_delay(b["horn"] + b["whistle"], 1 / 3, BPM, feedback=0.2, repeats=2, lp=3000, pingpong=False)

    duck = fx.duck(n, tl.kicks, depth_db=2.5, release_ms=180)[:, None]
    music = (b["sample"] + b["horn"] + b["whistle"] + b["pic"] + b["rhodes"] + b["lead"]) * duck
    lows = (b["bass"] + b["sub"]) * fx.duck(n, tl.kicks, depth_db=4, release_ms=120)[:, None]
    kit = b["kick"] + b["snare"] + b["hats"] + b["bodhran"] + b["stomp"] + b["clap"]
    drum_bus = fx.tape(fx.compress(kit, -16, 2.5, 12, 160, makeup_db=1), 1.3, 11000)
    band = music + lows + drum_bus * 0.85 + verb * 0.55 + slap * 0.3

    # the tavern: tankards drummed on the table into the bridge, a mug here and there, the clash for the last tune
    beat = 60 / BPM
    tb = [(bar_s(28) + k * beat, 0.5) for k in (0, 2)] + [(bar_s(29) + k * beat, 0.6) for k in range(4)]
    tb += [(bar_s(30) + k * beat / 3, 0.65) for k in range(12) if k % 3 != 1]
    tb += [(bar_s(31) + k * beat / 3, 0.7 + 0.3 * k / 11) for k in range(12)]
    table = fx.level(thumps(n, tb, seed=49), -30) * (n > 0)
    rng = np.random.default_rng(50)
    mugs = sorted(rng.uniform(bar_s(4), bar_s(50), 11))
    tavern = clinks(n, mugs, seed=51, cheers=(bar_s(40) - 0.06,))
    tavern = fx.level(tavern, -41)
    t_end = bar_s(end / 4)
    return dict(band=band, table=table, tavern=tavern, b=b, verb=verb, n=n, t_end=t_end)


def build(out_dir: Path, stems: bool = False) -> dict:
    first = S1.render(stomp=True)
    second = jig_half()
    t2 = S1.bar_s(49)                         # the first stomp, a bar after the landing
    door_shut = S1.bar_s(48.5)
    i2 = int(round(t2 * SR))
    t_end = t2 + second["t_end"]
    n = int((t_end + 11.0) * SR)
    t = np.arange(n) / SR

    # outside: the band's last hit and the swelling surf, until the door slams and leaves them muffled behind it
    outside = np.zeros((n, 2), dtype=np.float32)
    o = first["band"] + first["sea"] + first["gulls"]
    outside[:min(n, len(o))] += o[:n]
    d0 = door_shut + 0.01
    outside = fx.sweep_lowpass(outside, lambda tt: np.where(tt < d0, 18000.0, 18000 * (300 / 18000) ** np.clip(
        (tt - d0) / 0.06, 0, 1)))
    outside *= np.interp(t, [0, d0, d0 + 0.06, d0 + 3.0], [1, 1, 0.4, 0])[:, None]

    # inside: the jig, matched to the first half's level and a touch hotter
    band2 = second["band"]
    rms1 = fx.gated_rms_db(first["band"][: int(S1.bar_s(48) * SR)])
    rms2 = fx.gated_rms_db(band2[int(round(P * 60 / BPM * SR)):int(second["t_end"] * SR)])
    g = 10 ** ((rms1 - rms2 + 0.5) / 20)
    inside = np.zeros((n, 2), dtype=np.float32)
    seg = band2 * g + second["table"] + second["tavern"]
    m = min(len(seg), n - i2)
    inside[i2:i2 + m] = seg[:m]

    # the end: a mug set down, the door opens, and the sea and a gull come back in
    door_open = t_end + 3.0
    fx_doors = door(n, door_shut, True, seed=55) * 0.5 + door(n, door_open, False, seed=56) * 0.35
    fx_doors = fx_doors + clinks(n, (t_end + 2.0,), seed=57) * 0.4
    secs = (n + 0.5) / SR
    sea2 = AMB.waves(secs, -29, seed=58)[:n]
    sea2 = fx.sweep_lowpass(sea2, lambda tt: 300 * (16000 / 300) ** np.clip((tt - door_open - 0.15) / 0.8, 0, 1))
    sea2 = sea2 * np.interp(t, [0, door_open, door_open + 0.8, secs], [0, 0, 1, 1])[:, None]
    gull = AMB.gulls(secs, -33, seed=59, rate=0, times=(door_open + 1.5,))[:n]

    vinyl = AMB.vinyl(secs, -47, seed=53, crackle_rate=3.0)[:n]       # dense crackle would read as rain
    hiss = AMB.hiss(secs, -56, seed=54)[:n]
    mix = outside + inside + fx_doors + sea2 + gull + vinyl + hiss
    mix = fx.wow_flutter(mix, wow_ms=0.4, flutter_ms=0.03, drift_ms=0.3, seed=29)
    mix = fx.fade(mix, fin_s=1.0, fout_s=5.0)
    out = M.master(mix, -14.0, -1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{FILE}.wav"
    M.write(path, out, tags={"title": TITLE, "artist": "Gielinor Lo-Fi", "album": "Gielinor Lo-Fi",
                             "comment": "Lo-fi reinterpretation of 'Sea Shanty' and 'Sea Shanty 2' (Ian Taylor, Jagex) "
                                        "from Old School RuneScape"})
    bar1 = S1.bar_s(1)
    j = lambda b: (t2 + bar_s(b)) / bar1  # noqa: E731  jig bars in first-half bars, for the report
    secs_form = [(nm, b0, b0 + nb) for nm, b0, nb in S1.FORM if nm != "outro"] + [("landing", 48, 49),
                                                                                    ("count-in", 49, j(0))]
    secs_form += [(nm, j(b0), j(b0 + nb)) for nm, b0, nb in FORM]
    if stems:
        sd = out_dir / "stems"
        sd.mkdir(exist_ok=True)
        import soundfile as sf
        i0 = int(round(R0 * SR))                           # jig stems from its bar 0
        for k, x in {**second["b"], "verb": second["verb"]}.items():
            sf.write(str(sd / f"{k}.wav"), x[i0:], SR, subtype="PCM_16")
    rep = M.report(out, S1.BPM, secs_form)
    rep["file"] = str(path)
    rep["jig_starts_s"] = round(t2 + R0, 2)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    print(json.dumps(build(ROOT / "build" / "scallywag_stomp", stems="--stems" in sys.argv), indent=1))
