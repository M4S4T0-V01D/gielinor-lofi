"""Gielinor Lo-Fi - Gnomeball (after Ian Taylor's 'Gnomeball', Old School RuneScape).

Match day at the Gnome Stronghold, as 90s piano house: 124 bpm in Bb major. The source is a 130 bpm C major romp
whose trumpets already stab the off-beats and whose bass already bounces in octaves, so it is pressed 1:1 (song
bar b = source bar b + 1) and varispeeded down two semitones. The record keeps the brass, marimba and woodblock
band; the whistle tune stays whistled in A1, is sung by a choir ("diva") in B, and moves to an M1-style bright piano
in the breakdown and A2. New parts: syncopated bright-piano chord stabs, an off-beat synth bass with a sub, and a
909-ish Electronic kit with a ride on the off-beats and the source's own cabasa. The move is the game itself: a
referee's pea whistle blows for kick-off into A1, and in the breakdown the source's woodblock becomes a dropped
gnomeball, bouncing quicker and quieter while the band thins out under a rising highpass, until it settles, the
whistle restarts play and A2 slams in. The source's own chromatic ending climbs into a last chord and the
full-time whistle.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from engine import ambience as AMB, fx, master as M, notes as N, parts as PT  # noqa: E402
from engine.render import SR, fluid  # noqa: E402
from engine.sampler import Record, Timeline, edge  # noqa: E402

TITLE = "Gnomeball (Match Day)"
FILE = "Gnomeball (Match Day)"
BPM = 124.0
KEY = -2                      # C major -> Bb major
KIT = 24                      # GeneralUser GS Electronic kit
SWING = 0.54
SRC = N.load(ROOT / "source" / "gnomeball.mid")
OFF = 1                       # song bar b plays source bar b + OFF

# name, first song bar, bars
FORM = [("intro", 0, 8), ("A1", 8, 16), ("turn", 24, 4), ("B", 28, 24), ("breakdown", 52, 8), ("A2", 60, 16),
        ("outro", 76, 4)]
BARS = 80
BREAK = (52, 60)
BALL = (58, 59.75)            # song bars: the bounces; the last beat of bar 59 is the whistle's
LEVELS = {"sample": -22, "whistle": -26, "riff": -25, "diva": -27, "bells": -29, "keys": -27, "grand": -26,
          "ball": -25, "ref": -27, "bass": -26, "sub": -33, "kick": -22, "snare": -27, "hats": -35, "cabasa": -38}

v = PT.v
PIANO = {"C": v("G3", "C4", "E4", "G4"), "F": v("A3", "C4", "F4", "A4"), "G": v("G3", "B3", "D4", "G4"),
         "D": v("A3", "D4", "F#4", "A4"), "G11": v("A3", "C4", "D4", "F4")}
ROOTS = {"C": PT.P["C2"], "F": PT.P["F1"], "G": PT.P["G1"], "D": PT.P["D2"], "G11": PT.P["G1"]}


def chord(s):
    """The source's harmony by source bar: C and F in A, G - D - F - C in B, a G11 / C vamp in the interludes."""
    if 13 <= s < 17 or 21 <= s < 25 or 65 <= s < 69 or 73 <= s < 77:
        return "F"
    if 25 <= s < 29 or 53 <= s < 61:
        return "G11" if s % 2 else "C"
    if 29 <= s < 37:
        return "G"
    if 37 <= s < 41:
        return "D"
    if 41 <= s < 45:
        return "F"
    return "C"


# Gnomeball's groove: four on the floor (a pickup kick on the a of 4 into each phrase), a 909 snare on 2 and 4 with
# a clap on the 4 (flammed on odd bars), a ride on every off-beat, closed hats on the e and a, and the source's own
# cabasa 16ths. B and A2 add a tambourine on 2 and 4; the turn drops the kick.
GAME = [
    dict(bus="kick", piece=PT.KICK, at=[0, 4, 8, 12], fill=[0, 4, 8, 12, 15], vel=(100, 108), accent={0: 114}),
    dict(bus="snare", piece=PT.SNARE, at=[4, 12], vel=(78, 88)),
    dict(bus="snare", piece=PT.CLAP, at=[12], odd=[11, 12], vel=(80, 90), accent={11: 52}),
    dict(bus="hats", piece=PT.RIDE, at=[2, 6, 10, 14], vel=(62, 74), dur=0.4),
    dict(bus="hats", piece=PT.CHH, at=[1, 3, 5, 7, 9, 11, 13, 15], vel=(26, 40), p=0.7),
]
INTRO = [dict(bus="hats", piece=PT.RIDE, at=[2, 6, 10, 14], vel=(44, 54), dur=0.4)]
WARM = INTRO + [dict(bus="kick", piece=PT.KICK, at=[0, 4, 8, 12], vel=(72, 80))]
LIFT = GAME + [dict(bus="hats", piece=PT.TAMB, at=[4, 12], vel=(54, 64))]
TURN = [ln for ln in GAME if ln["bus"] != "kick"]
RESTART = [dict(bus="kick", piece=PT.KICK, at=[0, 4, 8, 12], vel=(84, 94))]
# (song bar, bars, spec)
DRUMS = [(0, 4, INTRO), (4, 4, WARM), (8, 16, GAME), (24, 4, TURN), (28, 24, LIFT), (56, 3, RESTART),
         (60, 16, LIFT), (76, 3, GAME)]


def bar_s(b):
    return b * 4 * 60 / BPM


def place(tl, bus, rec, b0, b1, gain=1.0, tail=0.5, fin=2, fout=30):
    """Lay song bars b0..b1 from the record (source bars b0 + OFF ..)."""
    tl.put(bus, edge(rec.beats((b0 + OFF) * 4, (b1 + OFF) * 4, tail_beats=tail), fin, fout), b0 * 4, gain)


def src_notes(track, b0, b1, keep=None):
    """Source notes for song bars b0..b1, moved to song beats."""
    lo, hi = (b0 + OFF) * 4, (b1 + OFF) * 4
    return N.shift([x for x in SRC.tracks[track] if lo <= x.start < hi and (keep is None or x.pitch in keep)],
                   -OFF * 4, 0)


def sweep_highpass(x, f_of_t):
    return x - fx.sweep_lowpass(x, f_of_t)


def pea_whistle(n, blasts, f0=2850.0, seed=7):
    """A referee's pea whistle: a sine with a few harmonics, the pea's rattle as a fast irregular FM and AM, a short
    upward scoop at each attack and a little breath noise. blasts: [(start_s, dur_s)]."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=np.float32)
    for t0, dur in blasts:
        L = int((dur + 0.06) * SR)
        i0 = int(t0 * SR)
        if i0 >= n:
            continue
        L = min(L, n - i0)
        t = np.arange(L) / SR
        rate = 30 + 4 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6))
        rattle = np.sin(np.cumsum(2 * np.pi * rate / SR))
        f = f0 * (1 + 0.045 * rattle) * (1 - 0.04 * np.exp(-t / 0.03))
        ph = np.cumsum(2 * np.pi * f / SR)
        tone = np.sin(ph) + 0.18 * np.sin(2 * ph) + 0.05 * np.sin(3 * ph)
        breath = fx.bandpass(rng.standard_normal((L, 1)).astype(np.float32), 2000, 5200)[:, 0] * 0.12
        env = np.minimum(1, t / 0.015) * np.clip((dur - t) / 0.04 + 1, 0, 1)
        out[i0:i0 + L] += ((tone * (0.78 + 0.22 * rattle) + breath) * env).astype(np.float32)
    return np.stack([out, out], axis=1)


def bounces(b0, b1, d0=1.0, r=0.86, pitch=72, seed=5):
    """A dropped ball between song bars b0 and b1: each bounce comes r times sooner and a little quieter, and its knock
    rises as the bounces get shallower. Returns woodblock notes (song beats)."""
    rng = np.random.default_rng(seed)
    t, d, ns = b0 * 4.0, d0, []
    end = b1 * 4.0
    k = 0
    while t < end and d > 0.035:
        h = d / d0
        ns.append(N.Note(t, 0.12, pitch + int(round(5 * (1 - h))), int(np.clip(70 + 45 * h ** 0.5
                                                                                 + rng.integers(-4, 5), 30, 120))))
        t += d
        d *= r
        k += 1
    return ns


def thumps(ns, n):
    """The ball hitting the turf under each knock: a short falling sine, quieter as the bounces get shallower."""
    out = np.zeros(n, dtype=np.float32)
    L = int(0.09 * SR)
    tt = np.arange(L) / SR
    body = np.sin(2 * np.pi * np.cumsum(170 * (0.55 + 0.45 * np.exp(-tt / 0.02))) / SR) * np.exp(-tt / 0.03)
    for x in ns:
        i0 = int(x.start * 60 / BPM * SR)
        m = min(L, n - i0)
        if m > 0:
            out[i0:i0 + m] += body[:m] * (x.vel / 127) ** 1.5
    return np.stack([out, out], axis=1)


def build(out_dir: Path, stems: bool = False) -> dict:
    tl = Timeline(BPM, BARS * 4, tail_s=7.0)
    n = tl.n

    # --- the record (1:1) -----------------------------------------------------------------------------------
    rec = dict(bpm=BPM, semis=KEY, curve=None, swing=SWING)
    band = Record(SRC, ["strings", "trombone", "trumpet", "french horns", "marimba", "clarinet", "woodblock"],
                  vel={"strings": 0.8}, **rec)
    whistle = Record(SRC, ["whistle"], **rec)
    riff = Record(SRC, ["whistle"], programs={"whistle": 1}, **rec)                 # the tune on bright piano
    diva = Record(SRC, ["whistle"], programs={"whistle": 52}, **rec)                # ... sung by a choir
    bells = Record(SRC, ["xylophone", "music box"], **rec)

    place(tl, "sample", band, 0, BREAK[0], tail=1)
    place(tl, "sample", band, 56, BREAK[1], gain=0.8)                               # the band creeps back
    place(tl, "sample", band, 60, 79, tail=4)
    place(tl, "whistle", whistle, 8, 24, tail=1)
    place(tl, "whistle", whistle, 76, 79, tail=3)
    place(tl, "diva", diva, 28, 52, tail=2)
    place(tl, "diva", diva, 60, 76, gain=0.55, tail=2)
    place(tl, "riff", riff, BREAK[0], 59, tail=1, fout=60)
    tl.put("riff", edge(riff.beats((59 + OFF) * 4, (59 + OFF) * 4 + 3), 2, 40), 59 * 4)  # the last beat is silent
    place(tl, "riff", riff, 60, 79, tail=3)
    place(tl, "bells", bells, 28, 52, tail=2)

    # --- new parts ------------------------------------------------------------------------------------------
    keys, grand, bass = [], [], []
    for b in range(8, 79):
        c = chord(b + OFF)
        if BREAK[0] <= b < BREAK[1]:                    # the breakdown: a low held piano chord a bar
            grand += [N.Note(b * 4, 3.9, p - 12, 54) for p in PIANO[c][:2]] + \
                     [N.Note(b * 4 + 0.02, 3.9, p, 48) for p in PIANO[c]]
            continue
        if 24 <= b < 28:                                 # the turn: the chord on the and of 1, held
            keys += [N.Note(b * 4 + 0.5, 3.0, p, 70) for p in PIANO[c]]
        else:
            hits = [0, 3, 6, 10, 13] if b % 2 == 0 else [2, 6, 8, 11, 14]
            if 28 <= b < 52:                             # under the diva, only the off-beat half
                hits = [h for h in hits if h % 4]
            top = PIANO[c][-1] + 12
            keys += [N.Note(b * 4 + h / 4, 0.32, p, 84 + (8 if h in (0, 2) else 0)) for h in hits
                     for p in PIANO[c] + [top]]
        r = ROOTS[c]
        steps = [(0.5, 0), (1.5, 0), (2.5, 0), (2.75, 12), (3.5, 0)] if b % 2 == 0 else \
                [(0.5, 0), (1.5, 0), (1.75, 12), (2.5, 0), (3.5, 0), (3.75, 7)]
        bass += [N.Note(b * 4 + at, 0.3, r + up, 96) for at, up in steps]
    # the full-time chord
    keys += [N.Note(79 * 4, 12, p, 66) for p in PIANO["C"] + [PIANO["C"][-1] + 12]]
    grand += [N.Note(79 * 4, 12, p, 60) for p in v("C2", "C3", "G3")]
    bass += [N.Note(79 * 4, 6, PT.P["C2"], 90)]
    keys = N.shift(N.humanize(N.swing(keys, SWING), BPM, 4, 5, seed=6), 0, KEY)
    grand = N.shift(N.humanize(grand, BPM, 6, 4, seed=7), 0, KEY)
    bass = N.shift(N.humanize(N.swing(bass, SWING), BPM, 4, 4, seed=9, push_ms=2), 0, KEY)
    tl.put_full("keys", fluid([(keys, 1, 0, 4)], BPM, tail=8))                       # bright piano
    tl.put_full("grand", fluid([(grand, 0, 0, 5)], BPM, tail=8))
    tl.put_full("bass", fluid([(bass, 38, 0, 2)], BPM, tail=4))                      # synth bass 1
    tl.put_full("sub", PT.sub(bass, BPM, n))

    # the ball: the source's woodblock knock, and the turf under it
    ball = N.shift(bounces(*BALL), 0, KEY)
    tl.put_full("ball", fluid([(ball, 115, 0, 6)], BPM, tail=2))
    turf = thumps(ball, n)

    drums = {"kick": [], "snare": [], "hats": []}
    for b0, nb, spec in DRUMS:
        for piece, ns in PT.groove(nb, b0, spec, seed=50 + b0, swing_amt=SWING, bpm=BPM).items():
            drums[piece] += ns
    drums["hats"] += [N.Note(b * 4, 2.0, 49, 90) for b in (8, 28, 60)]               # crashes
    drums["hats"] += [N.Note(79 * 4, 3.0, 49, 84)]
    drums["kick"] += [N.Note(79 * 4, 0.5, PT.KICK, 110)]
    tl.kicks = [x.start * 60 / BPM for x in drums["kick"]]
    for piece, ns in drums.items():
        tl.put_full(piece, fluid([(ns, KIT, 128, 9)], BPM, tail=2.5))
    cab = [x for a, z in ((8, 52), (60, 79)) for x in src_notes("perc", a, z, keep={69})]
    tl.put_full("cabasa", fluid([(N.humanize(N.swing(cab, SWING), BPM, 5, 6, seed=11), KIT, 128, 9)], BPM, tail=1))

    # --- processing -----------------------------------------------------------------------------------------
    b = tl.bus
    t = np.arange(n) / SR

    def thin(tt):       # filter-house highpass: opens through the intro, rises again over the breakdown's build
        f = np.where(tt < bar_s(8), 1800 * (160 / 1800) ** np.clip(tt / bar_s(8), 0, 1) ** 0.8, 160.0)
        u = np.clip((tt - bar_s(56)) / (bar_s(59.75) - bar_s(56)), 0, 1)
        return np.where((tt > bar_s(56)) & (tt < bar_s(60)), 160 * (2200 / 160) ** u ** 1.6, f)

    def ride(tt):       # the record's lowpass: closes for the breakdown, shuts in the outro
        f = np.interp(tt, [0, bar_s(51.5), bar_s(52), bar_s(56), bar_s(59.9), bar_s(60)],
                      [5200, 5200, 1400, 1400, 4200, 5200])
        return np.where(tt > bar_s(79), 5200 * (500 / 5200) ** np.clip((tt - bar_s(79)) / 4, 0, 1), f)

    s = fx.highpass(b["sample"], 140, 2)
    s = fx.bitcrush(s, 12, 26040, mix=0.75)
    s = fx.saturate(fx.shelf_tilt(s, 800, -0.8), 1.5)
    s = sweep_highpass(fx.sweep_lowpass(s, ride), thin)
    b["sample"] = fx.wow_flutter(s, wow_ms=1.4, wow_hz=0.5, seed=41)
    wh = fx.lowpass(fx.highpass(b["whistle"], 500, 2), 6500, 2)
    b["whistle"] = fx.pan(fx.wow_flutter(fx.bitcrush(wh, 12, 26040, mix=0.4), seed=42), -0.15)
    rf = fx.shelf_tilt(fx.lowpass(fx.highpass(b["riff"], 180, 2), 6000, 2), 1000, -0.6)
    rf = fx.compress(fx.saturate(fx.detune_chorus(rf, cents=5, rate=0.6, mix=0.3), 1.3), -20, 3, 6, 120)
    b["riff"] = fx.bitcrush(rf, 12, 26040, mix=0.35)
    dv = fx.lowpass(fx.highpass(b["diva"], 280, 2), 5500, 2)
    b["diva"] = fx.widen(fx.detune_chorus(fx.saturate(dv, 1.2), cents=9, rate=0.3, mix=0.45), 1.5)
    bl = fx.bitcrush(fx.highpass(b["bells"], 400, 2), 12, 26040, mix=0.6)
    b["bells"] = fx.phaser(fx.lowpass(bl, 6500, 2), rate=BPM / 60 / 16, lo=500, hi=3800, mix=0.45)
    ky = fx.shelf_tilt(fx.lowpass(fx.highpass(b["keys"], 200, 2), 5800, 2), 1000, -0.5)
    ky = fx.compress(fx.saturate(ky, 1.4), -18, 3, 4, 90)
    b["keys"] = fx.sweep_lowpass(fx.bitcrush(ky, 12, 26040, mix=0.4), ride)
    b["grand"] = fx.lowpass(fx.highpass(b["grand"], 60, 2), 4200, 2)
    b["ball"] = fx.highpass(b["ball"], 300, 2) + turf * 0.5
    swing_pan = np.interp(np.arange(n), [tl.i(x.start) for x in ball] or [0],
                          [(-0.3 if k % 2 else 0.3) for k in range(len(ball))] or [0])
    b["ball"] = b["ball"] * np.stack([np.sqrt((1 - swing_pan) / 2), np.sqrt((1 + swing_pan) / 2)], axis=1) * 1.41
    blasts = [(bar_s(7.5), 1.05),                                               # kick-off
              (bar_s(59.75), 0.16), (bar_s(59.75) + 0.24, 0.16),                # play on
              (bar_s(79) + 0.5, 0.22), (bar_s(79) + 0.85, 0.22), (bar_s(79) + 1.2, 1.3)]  # full time
    b["ref"] = fx.pan(pea_whistle(n, blasts), 0.2)

    bb = fx.saturate(fx.lowpass(b["bass"], 1100, 2), 1.8)
    b["bass"] = fx.mono_below(fx.compress(bb, -20, 3, 10, 120), 300)
    b["sub"] = fx.lowpass(b["sub"], 100, 2)

    kk = fx.saturate(fx.lowpass(b["kick"], 6000, 2), 2.2)
    # the restart kick comes back muffled and opens up into the whistle
    kr = fx.sweep_lowpass(kk, lambda tt: np.where((tt > bar_s(56)) & (tt < bar_s(60)),
                                                  np.interp(tt, [bar_s(56), bar_s(59.75)], [300, 6000]), 20000))
    b["kick"] = fx.compress(kr, -14, 3, 3, 90)
    sn = fx.bitcrush(fx.highpass(b["snare"], 180, 2), 12, 26040)
    b["snare"] = fx.lowpass(fx.saturate(sn, 1.5), 9000, 2)
    hh = fx.bitcrush(fx.lowpass(fx.highpass(b["hats"], 800, 2), 7500, 2), 12, 26040)
    b["hats"] = fx.autopan(hh, rate=BPM / 60 / 16, depth=0.3)
    b["cabasa"] = fx.pan(fx.lowpass(fx.highpass(b["cabasa"], 2000, 2), 8000, 2), 0.35)

    lv = {k: fx.db(LEVELS[k]) / max(1e-9, fx.db(fx.gated_rms_db(b[k]))) for k in LEVELS}
    for k in LEVELS:
        b[k] = b[k] * lv[k]

    # space: a bright club plate on the piano, choir and bells; the breakdown piano in a long hall; a stadium slap on
    # the referee; a quarter-triplet ping-pong on the choir, an eighth echo on the riff
    plate = fx.reverb(b["riff"] * 0.4 + b["diva"] * 0.6 + b["bells"] * 0.4 + b["keys"] * 0.25 + b["snare"] * 0.2,
                      seconds=1.8, damp_hz=4200, predelay_ms=18, seed=43)
    gate_br = np.interp(t, [bar_s(51.8), bar_s(52), bar_s(59.8), bar_s(60)], [0, 1, 1, 0])[:, None]
    hall = fx.reverb((b["riff"] * gate_br + b["grand"]) * 0.7 + b["keys"] * 0.15, seconds=3.6, damp_hz=3600,
                     predelay_ms=35, seed=44)
    field = fx.reverb(b["ref"] + b["ball"] * 0.5, seconds=1.6, damp_hz=5000, predelay_ms=60, hp=600, seed=45)
    echo = fx.tape_delay(b["diva"], 2 / 3, BPM, feedback=0.32, repeats=4, lp=3400, pingpong=True)
    recho = fx.tape_delay(b["riff"], 0.5, BPM, feedback=0.25, repeats=3, lp=3000, pingpong=False)

    duck = fx.duck(n, tl.kicks, depth_db=3.0, attack_ms=5, release_ms=170)[:, None]
    music = (b["sample"] + b["whistle"] + b["riff"] + b["diva"] + b["bells"] + b["keys"]) * duck + b["grand"]
    low = (b["bass"] + b["sub"]) * fx.duck(n, tl.kicks, depth_db=5.0, release_ms=140)[:, None]
    drum_bus = fx.compress(b["kick"] + b["snare"] + b["hats"] + b["cabasa"], -16, 2.5, 8, 140, makeup_db=1)
    drum_bus = fx.tape(drum_bus, 1.3, 12000)

    # ambience: birds over the pitch in the intro, breakdown and outro; sparse vinyl and hiss
    secs = n / SR
    amt = np.interp(t, [0, bar_s(7), bar_s(9), bar_s(51), bar_s(53), bar_s(58), bar_s(60), bar_s(77), secs],
                    [1.0, 1.0, 0.25, 0.25, 0.9, 0.9, 0.25, 0.25, 1.0])[:, None]
    def fit(x):
        return np.pad(x[:n], ((0, n - len(x[:n])), (0, 0)))

    air = fit(AMB.birds(secs, -44, seed=46)) * amt
    vinyl = fit(AMB.vinyl(secs, -49, seed=47, crackle_rate=2.0))
    hiss = fit(AMB.hiss(secs, -52, seed=48))

    mix = (music + low + drum_bus * 0.85 + b["ball"] + b["ref"] + plate * 0.35 + hall * 0.5 + field * 0.4
           + echo * 0.25 + recho * 0.15 + air + vinyl + hiss)
    mix = fx.wow_flutter(mix, wow_ms=0.4, wow_hz=0.35, flutter_ms=0.04, drift_ms=0.3, seed=40)
    end = int((bar_s(BARS) + 6.0) * SR)
    mix = fx.fade(mix[:end], fin_s=0.5, fout_s=4.0)
    out = M.master(mix, -14.0, -1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{FILE}.wav"
    M.write(path, out, tags={"title": TITLE, "artist": "Gielinor Lo-Fi", "album": "Gielinor Lo-Fi",
                             "comment": "Lo-fi reinterpretation of 'Gnomeball' (Ian Taylor, Jagex) from Old School RuneScape"})
    if stems:
        sd = out_dir / "stems"
        sd.mkdir(exist_ok=True)
        import soundfile as sf
        for k, x in {**b, "plate": plate, "hall": hall, "field": field, "echo": echo, "air": air}.items():
            sf.write(str(sd / f"{k}.wav"), x[:end], SR, subtype="PCM_16")
    rep = M.report(out, BPM, [(nm, b0, b0 + nb) for nm, b0, nb in FORM])
    rep["file"] = str(path)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    print(json.dumps(build(ROOT / "build" / "gnomeball", stems="--stems" in sys.argv), indent=1))
