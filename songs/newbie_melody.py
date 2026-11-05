"""Gielinor Lo-Fi - Newbie Melody (after Ian Taylor's 'Newbie Melody', Old School RuneScape).

A first morning on Tutorial Island at 94 bpm in B mixolydian. The 4/4 MIDI is pressed 1:1 and varispeeded down a
semitone, so it plays at ~100 bpm, the source's own speed (it is the album's quickest lo-fi tune on purpose: it
drags any slower). The record re-voices the source: harp, a muted trumpet on the tune, celesta on the B melody, a
kalimba on the triplet bells. New parts: a Bell Tine EP, upright bass, an 808/909 kit whose hats move onto a
triplet grid when the source's bells and horns go into triplets. The one added move: the intro plays out of a tiny
speaker (mono, no lows, a little breakup) and the record opens to full range on the downbeat of A1; the outro
goes back into the speaker as the morning carries on.
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

TITLE = "Newbie Melody (Tutorial Island)"
FILE = "Newbie Melody (Tutorial Island)"
BPM = 94.0
KEY = -1                      # C mixolydian -> B mixolydian
KIT = 25                      # GeneralUser GS 808/909 kit
SWING = 0.58
SRC = N.load(ROOT / "source" / "newbie_melody.mid")

MAJ = [[("C69", 4)], [("Gm9", 4)]]           # the source rocks C / Gm every bar
MAJ2 = [[("C69", 4)], [("Gm11", 4)]]
MIN = [[("Cm9", 4)], [("Gm9", 4)]]

# name, first song bar, bars, source bar
FORM = [("intro", 0, 4, 0), ("A1", 4, 8, 4), ("B", 12, 8, 20), ("bells", 20, 8, 28), ("C", 28, 8, 36),
        ("minor", 36, 8, 52), ("A2", 44, 16, 60), ("outro", 60, 4, 78)]
BARS = 64
CHART = MAJ * 18 + MIN * 4 + MAJ * 4 + MAJ2 * 4 + MAJ * 2
LEVELS = {"sample": -20, "pad": -30, "horn": -24, "lead": -27, "bells": -26, "keys": -28, "glock": -33,
          "bass": -27, "sub": -34, "kick": -23, "snare": -26, "hats": -33}
OPEN = 4 * 4                  # beat where the small speaker opens up
CLOSE = (60 * 4, 63 * 4)      # and closes again

# Newbie's groove: a bouncy 808/909 step, the kick on 1, a+1 and 3 (plus the and of 3 on odd bars), snare on 2
# and 4, 8th hats with a 16th pickup into each beat 1, an open hat on the and of 2 now and then, a rim on the last
# 16th of odd bars.
MAIN = [
    dict(bus="kick", piece=PT.KICK, at=[0, 3, 8], odd=[0, 3, 8, 10], fill=[0, 3, 8, 10, 13], vel=(86, 98),
         accent={0: 108}),
    dict(bus="snare", piece=PT.SNARE, at=[4, 12], vel=94),
    dict(bus="snare", piece=PT.STICK, at=[], odd=[15], fill=[14, 15], vel=(44, 60), p=0.8),
    dict(bus="hats", piece=PT.CHH, at=[0, 2, 4, 6, 8, 10, 12, 14, 15], vel=(40, 52),
         accent={0: 66, 4: 62, 8: 66, 12: 62, 15: 34}),
    dict(bus="hats", piece=PT.OHH, at=[6], vel=(44, 52), p=0.3, dur=0.3),
]
# B: the celesta tune gets room: kick 1 and the and of 3, a sidestick backbeat, 8th hats
LIGHT = [
    dict(bus="kick", piece=PT.KICK, at=[0, 10], odd=[0, 7, 10], vel=(80, 92), accent={0: 100}),
    dict(bus="snare", piece=PT.STICK, at=[4, 12], vel=(82, 92)),
    dict(bus="hats", piece=PT.CHH, at=[0, 2, 4, 6, 8, 10, 12, 14], vel=(34, 46), p=0.9),
]
# bells and C: the source goes into triplets, so does the kit (a 12-step grid, never swung)
TRIP = [
    dict(bus="kick", piece=PT.KICK, at=[0, 5, 6], odd=[0, 5, 6, 11], vel=(84, 96), accent={0: 106}),
    dict(bus="snare", piece=PT.SNARE, at=[3, 9], vel=92),
    dict(bus="hats", piece=PT.CHH, at=list(range(12)), vel=(30, 42), accent={0: 60, 3: 56, 6: 60, 9: 56},
         p=0.85),
]
# the minor turn: half-time, the snare only on 3, a pedal hat on the 8ths
HALF = [
    dict(bus="kick", piece=PT.KICK, at=[0, 10], odd=[0, 3, 10], vel=(84, 94), accent={0: 104}),
    dict(bus="snare", piece=PT.SNARE, at=[8], vel=96),
    dict(bus="snare", piece=PT.CLAP, at=[8], vel=60),
    dict(bus="hats", piece=PT.PHH, at=[2, 6, 10, 14], vel=(46, 58)),
]
# A2: the full groove plus a clap on 4 and an open hat into each odd bar
A2 = MAIN + [dict(bus="snare", piece=PT.CLAP, at=[12], vel=56),
             dict(bus="hats", piece=PT.OHH, at=[], odd=[14], vel=(46, 54), dur=0.4)]
GROOVE = {"A1": (MAIN, 16), "B": (LIGHT, 16), "bells": (TRIP, 12), "C": (TRIP, 12), "minor": (HALF, 16),
          "A2": (A2, 16)}


def bar_s(b):
    return b * 4 * 60 / BPM


def place_bars(tl, bus, rec, src_bar, song_bar, nbars, gain=1.0):
    for k in range(nbars):
        tl.put(bus, edge(rec.bar(src_bar + k, tail_beats=0.2), 2, 25), (song_bar + k) * 4, gain)


def build(out_dir: Path, stems: bool = False) -> dict:
    tl = Timeline(BPM, BARS * 4, tail_s=7.0)
    n = tl.n

    # --- the record -----------------------------------------------------------------------------------------
    rec = dict(bpm=BPM, semis=KEY, curve=None, swing=SWING)
    hook = Record(SRC, ["harp", "pizz"], programs={"harp": 46, "pizz": 45}, vel={"pizz": 0.85}, **rec)
    pad = Record(SRC, ["slow strings", "choir"], programs={"choir": 52}, **rec)
    horn = Record(SRC, ["horn melody"], programs={"horn melody": 59}, **rec)               # muted trumpet
    lead = Record(SRC, ["crystal", "french horns", "strings"],
                  programs={"crystal": 8, "french horns": 60, "strings": 48}, vel={"strings": 0.8}, **rec)
    bells = Record(SRC, ["tubular bells"], programs={"tubular bells": 108}, vel={"tubular bells": 0.8}, **rec)

    for name, b0, nb, sb in FORM:
        place_bars(tl, "sample", hook, sb, b0, nb, 0.9 if name in ("intro", "outro") else 1.0)
        if name not in ("intro", "outro", "B"):
            place_bars(tl, "pad", pad, sb, b0, nb)
    place_bars(tl, "horn", horn, 4, 4, 8)                       # A1: the tune once
    place_bars(tl, "horn", horn, 60, 44, 16)                    # A2: twice
    place_bars(tl, "lead", lead, 20, 12, 8)                     # B: the celesta tune
    place_bars(tl, "lead", lead, 36, 28, 8)                     # C: triplet horns and strings
    place_bars(tl, "lead", lead, 52, 36, 8, 0.8)                # the minor turn: celesta octaves
    place_bars(tl, "bells", bells, 28, 20, 16)                  # bells and C: kalimba triplets

    # --- new parts ------------------------------------------------------------------------------------------
    keys = PT.comp(CHART[2:60], start_bar=2, vel=50, seed=4, bpm=BPM)
    keys += PT.comp([[("C69", 4)]] * 2, start_bar=60, vel=42, style="pad", bpm=BPM)
    keys += PT.comp([[("C69", 4)]], start_bar=63, vel=40, style="pad", bpm=BPM)
    keys = [N.Note(x.start, x.dur + (5 if x.start >= 63 * 4 else 0), x.pitch, x.vel) for x in keys]
    tl.put_full("keys", fluid([(N.shift(N.swing(keys, SWING), 0, KEY), 4, 12, 0)], BPM, tail=7))   # Bell Tine EP
    tune = N.top_line(N.window(SRC.tracks["horn melody"], 68, 76))
    glock = PT.melody_double(N.swing(N.shift(tune, 52 * 4), SWING), octave=1, vel=40)        # last 8 bars of A2
    tl.put_full("glock", fluid([(N.shift(glock, 0, KEY), 9, 0, 1)], BPM, tail=4))

    bass = PT.bass(CHART[4:60], start_bar=4, vel=86, seed=5, low=PT.P["E1"], high=PT.P["D#2"])
    bass += [N.Note(60 * 4, 3.5, PT.P["C2"], 70), N.Note(61 * 4, 3.5, PT.P["G1"], 66),
             N.Note(63 * 4, 6, PT.P["C2"], 72)]
    bass = N.shift(N.humanize(N.swing(bass, SWING), BPM, 6, 4, seed=6, push_ms=3), 0, KEY)
    tl.put_full("bass", fluid([(bass, 32, 0, 2)], BPM, tail=4))                         # upright
    tl.put_full("sub", PT.sub(bass, BPM, n))

    drums = {"kick": [], "snare": [], "hats": []}
    for name, b0, nb, _ in FORM:
        if name == "outro":
            g, steps, nb = [x for x in LIGHT if x["bus"] == "hats"], 16, 2
        elif name in GROOVE:
            g, steps = GROOVE[name]
        else:
            continue
        for piece, ns in PT.groove(nb, b0, g, seed=30 + b0, swing_amt=SWING, bpm=BPM, steps=steps).items():
            drums[piece] += ns
    tl.kicks = [x.start * 60 / BPM for x in drums["kick"]]
    for piece, ns in drums.items():
        tl.put_full(piece, fluid([(ns, KIT, 128, 9)], BPM, tail=2))

    # --- processing -----------------------------------------------------------------------------------------
    b = tl.bus

    def ride(t):        # the record filter only closes through the outro (the speaker does the intro)
        o0, o1 = bar_s(60), bar_s(63.5)
        return np.where(t > o0, 5400 * (380 / 5400) ** np.clip((t - o0) / (o1 - o0), 0, 1), 5400.0)

    s = fx.shelf_tilt(fx.highpass(b["sample"], 190, 2), 600, 2.0)      # the harp sits low; lift its pluck
    s = fx.bitcrush(s, 12, 26040, mix=0.7)
    s = fx.saturate(s, 1.35)
    s = fx.sweep_lowpass(s, ride)
    b["sample"] = fx.wow_flutter(s, wow_ms=1.4, wow_hz=0.5, seed=21)
    p = fx.lowpass(fx.highpass(b["pad"], 240, 2), 3800, 2)
    b["pad"] = fx.widen(fx.wow_flutter(p, wow_ms=1.4, wow_hz=0.5, seed=22), 1.5)
    hn = fx.saturate(fx.shelf_tilt(fx.lowpass(fx.highpass(b["horn"], 260, 2), 5000, 2), 700, 1.5), 1.4)
    b["horn"] = fx.wow_flutter(fx.bitcrush(hn, 12, 26040, mix=0.6), wow_ms=1.4, wow_hz=0.5, seed=23)
    ld = fx.saturate(fx.lowpass(fx.highpass(b["lead"], 280, 2), 5400, 2), 1.3)
    b["lead"] = fx.pan(fx.wow_flutter(fx.bitcrush(ld, 12, 26040, mix=0.5), seed=24), 0.15)
    bl = fx.bitcrush(fx.lowpass(fx.highpass(b["bells"], 300, 2), 6000, 2), 12, 26040, mix=0.6)
    b["bells"] = fx.autopan(fx.wow_flutter(bl, seed=25), rate=BPM / 60 / 8, depth=0.5)     # one sweep per 2 bars

    r = fx.detune_chorus(b["keys"], cents=7, rate=0.25, mix=0.3)
    r = fx.saturate(fx.lowpass(fx.highpass(r, 170, 2), 5200, 2), 1.25)
    b["keys"] = fx.wow_flutter(r, wow_ms=1.3, wow_hz=0.5, seed=26)
    b["glock"] = fx.pan(fx.bitcrush(fx.lowpass(b["glock"], 6500, 2), 12, 26040, mix=0.5), -0.25)

    bb = fx.saturate(fx.lowpass(b["bass"], 1100, 2), 1.4)
    b["bass"] = fx.mono_below(fx.compress(bb, -20, 3, 12, 130), 300)
    b["sub"] = fx.lowpass(b["sub"], 110, 2)

    kk = fx.saturate(fx.lowpass(b["kick"], 5000, 2), 2.0)
    b["kick"] = fx.compress(kk, -14, 3, 3, 90)
    sn = fx.bitcrush(fx.highpass(b["snare"], 170, 2), 12, 26040)
    b["snare"] = fx.lowpass(fx.saturate(sn, 1.5), 9000, 2)
    hh = fx.bitcrush(fx.lowpass(fx.highpass(b["hats"], 1000, 2), 9500, 2), 12, 26040)
    b["hats"] = fx.pan(hh, 0.25)

    for k, target in LEVELS.items():
        b[k] = fx.level(b[k], target)

    # space: a bright morning room, an 8th-note ping-pong echo on the trumpet and celesta
    send = (b["sample"] * 0.3 + b["keys"] * 0.4 + b["pad"] * 0.6 + b["horn"] * 0.35 + b["lead"] * 0.45
            + b["bells"] * 0.4 + b["glock"] * 0.5 + b["snare"] * 0.18)
    verb = fx.reverb(send, seconds=1.8, damp_hz=4200, predelay_ms=15, seed=27)
    echo = fx.tape_delay(b["horn"] + b["lead"] * 0.7 + b["glock"] * 0.6, 0.5, BPM, feedback=0.32, repeats=4,
                         lp=3200)

    duck = fx.duck(n, tl.kicks, depth_db=2.0, release_ms=200)[:, None]
    music = (b["sample"] + b["pad"] + b["horn"] + b["lead"] + b["bells"] + b["keys"] + b["glock"]) * duck
    music = music + verb * 0.5 + echo * 0.3
    low = (b["bass"] + b["sub"]) * fx.duck(n, tl.kicks, depth_db=3.5, release_ms=140)[:, None]
    drum_bus = fx.compress(b["kick"] + b["snare"] + b["hats"], -16, 2.5, 10, 150, makeup_db=1)
    drum_bus = fx.tape(drum_bus, 1.2, 11000)

    # the small speaker: the intro comes out of a little radio on the windowsill, the record opens up on the
    # downbeat of A1 (a 1/16 snap, no sweep) and the outro sinks back into it over three bars
    tiny = fx.small_speaker(music + low * 0.5, 480, 3400, 1.8)
    tiny = fx.level(tiny, fx.gated_rms_db(music[:tl.i(OPEN)]) + 3)
    t = np.arange(n) / SR
    o = tl.s(OPEN)
    wet = np.interp(t, [0, o - 0.04, o, tl.s(CLOSE[0]), tl.s(CLOSE[1]), t[-1]], [1, 1, 0, 0, 1, 1])[:, None]
    music = music * (1 - wet) + tiny * wet
    low = low * (1 - wet)

    # ambience: birds in the hedges (outside the speaker, so always full range), cassette hiss, a sparse vinyl tick
    secs = n / SR
    birds = AMB.birds(secs, -40, seed=41, rate=0.25)
    hiss = AMB.hiss(secs, -52, seed=42)
    vin = AMB.vinyl(secs, -48, seed=43, crackle_rate=2.0)

    mix = music + low + drum_bus * 0.85 + birds + hiss + vin
    mix = fx.wow_flutter(mix, wow_ms=0.45, wow_hz=0.4, flutter_ms=0.04, drift_ms=0.35, seed=29)
    end = int((bar_s(BARS) + 6.0) * SR)
    mix = fx.fade(mix[:end], fin_s=0.5, fout_s=4.0)
    out = M.master(mix, -14.0, -1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{FILE}.wav"
    M.write(path, out, tags={"title": TITLE, "artist": "Gielinor Lo-Fi", "album": "Gielinor Lo-Fi",
                             "comment": "Lo-fi reinterpretation of 'Newbie Melody' (Ian Taylor, Jagex) from Old School RuneScape"})
    if stems:
        sd = out_dir / "stems"
        sd.mkdir(exist_ok=True)
        import soundfile as sf
        for k, x in {**b, "verb": verb, "echo": echo, "birds": birds}.items():
            sf.write(str(sd / f"{k}.wav"), x[:end], SR, subtype="PCM_16")
    rep = M.report(out, BPM, [(nm, b0, b0 + nb) for nm, b0, nb, _ in FORM])
    rep["file"] = str(path)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    print(json.dumps(build(ROOT / "build" / "newbie_melody", stems="--stems" in sys.argv), indent=1))
