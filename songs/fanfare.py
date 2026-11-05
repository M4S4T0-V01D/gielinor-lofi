"""Gielinor Lo-Fi - Fanfare (after Ian Taylor's 'Fanfare', Old School RuneScape).

A summer afternoon in Falador Park: dusty lo-fi house at 116 bpm in Eb minor. The source is a 120 bpm march that
already has a kick on every beat, so it is pressed 1:1 (one source bar per song bar) and varispeeded down a
semitone, near its own speed. The record keeps the brass ostinato and re-voices the rest: the horn tune as a tenor
sax, the flute tune whistled, the EP figure on a music box, the sweep pad as a warm pad. New parts: a percussive
organ stabbing the off-beats into a hard sidechain pump, a rubbery synth bass with off-beat octaves in the calls,
a sub, and a 909-style dance kit. The move: at the first fanfare call the band in the park stops and the source's
own trumpets, horn and drums march past as a parade, from far left, past the bench, away to the right (distance
gain, air absorption, a small Doppler bend), with the fountain bubbling behind; the house beat slams back for B.
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

TITLE = "Fanfare (Falador Park)"
FILE = "Fanfare (Falador Park)"
BPM = 116.0
KEY = -1                      # E minor -> Eb minor
KIT = 26                      # GeneralUser GS dance kit
PARADE_KIT = 48               # orchestral kit: concert bass drum, snare, timpani for the source's toms
SWING = 0.56
SRC = N.load(ROOT / "source" / "fanfare.mid")

# name, first bar, bars (the record is laid 1:1, so song bar = source bar; the outro is new)
FORM = [("intro", 0, 8), ("A1", 8, 16), ("call1", 24, 4), ("B", 28, 16), ("call2", 44, 4), ("C", 48, 8),
        ("A2", 56, 8), ("call3", 64, 4), ("outro", 68, 4)]
BARS = 72
LEVELS = {"sample": -21, "pad": -31, "sax": -24, "whistle": -27, "mbox": -28, "brass": -24, "organ": -29,
          "parade": -20, "march": -36, "bass": -25, "sub": -32, "kick": -21, "snare": -27, "hats": -35}
CALL1 = (24, 28)


def chord(b):
    """The source's harmony: E minor, B minor in bars 16-19 and 36-39, A minor in 48-55."""
    if 16 <= b < 20 or 36 <= b < 40:
        return "Bm11"
    if 48 <= b < 56:
        return "Am9"
    return "Em9" if b % 2 == 0 else "Em11"


# Fanfare's groove: four on the floor, a clap on 2 and 4 with a snare under the 4, an open hat on every off-beat,
# shuffled 16th closed hats, a rim ghost on the e of 2 (and the a of 4 on odd bars). Each section changes it.
HOUSE = [
    dict(bus="kick", piece=PT.KICK, at=[0, 4, 8, 12], vel=(100, 108), accent={0: 112}),
    dict(bus="snare", piece=PT.CLAP, at=[4, 12], vel=(84, 94)),
    dict(bus="snare", piece=PT.SNARE, at=[12], vel=50),
    dict(bus="snare", piece=PT.STICK, at=[5], odd=[5, 15], fill=[5, 13, 15], vel=(30, 44), p=0.5),
    dict(bus="hats", piece=PT.OHH, at=[2, 6, 10, 14], vel=(60, 72), dur=0.3),
    dict(bus="hats", piece=PT.CHH, at=[0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15], vel=(24, 38), p=0.75,
         accent={4: 44, 12: 44}),
]
INTRO = [dict(bus="kick", piece=PT.KICK, at=[0, 4, 8, 12], vel=(70, 78)),
         dict(bus="hats", piece=PT.OHH, at=[2, 6, 10, 14], vel=(40, 50), dur=0.25)]
LIFT = HOUSE + [dict(bus="hats", piece=PT.SHAKER, at=list(range(16)), vel=(22, 34), accent={2: 46, 10: 46})]
BREAKDOWN = [dict(bus="snare", piece=PT.CLAP, at=[4, 12], vel=(70, 80)),
             dict(bus="hats", piece=PT.OHH, at=[2, 6, 10, 14], vel=(44, 54), dur=0.25),
             dict(bus="hats", piece=PT.SHAKER, at=list(range(16)), vel=(20, 30), p=0.8)]
GROOVE = {"intro": INTRO, "A1": HOUSE, "B": LIFT, "call2": LIFT, "C": BREAKDOWN, "A2": HOUSE, "call3": LIFT,
          "outro": INTRO}


def bar_s(b):
    return b * 4 * 60 / BPM


def place(tl, bus, rec, b0, b1, gain=1.0, tail=0.5):
    """Lay record bars b0..b1 at the same song bars (the record is 1:1)."""
    tl.put(bus, edge(rec.beats(b0 * 4, b1 * 4, tail_beats=tail), 2, 30), b0 * 4, gain)


def bars_of(track, b0, b1, keep=None):
    return [x for x in SRC.tracks[track] if b0 * 4 <= x.start < b1 * 4 and (keep is None or x.pitch in keep)]


def parade_pass(x, t0, t1, near=4.0, far=22.0):
    """A marching band walking past the listener from left to right between t0 and t1 seconds, closest halfway:
    inverse-distance gain, air absorption (the lowpass closes with distance), angle panning and a small Doppler bend
    (the propagation delay, scaled down so it bends the pitch by ~10 cents). Returns (direct, reverb send); the send
    is not distance-scaled, so the far band is mostly room."""
    n = len(x)
    t = np.arange(n) / SR
    u = np.clip((t - t0) / (t1 - t0), 0, 1)
    lat = far * (2 * u - 1)
    dist = np.sqrt(lat ** 2 + near ** 2)
    g = (near / dist).astype(np.float32)
    idx = np.arange(n) - dist / 343.0 * 0.4 * SR
    mono = np.interp(idx, np.arange(n), x.mean(axis=1)).astype(np.float32)[:, None].repeat(2, axis=1)

    def air(tt):
        uu = np.clip((tt - t0) / (t1 - t0), 0, 1)
        return 900 + 8100 * (near / np.sqrt((far * (2 * uu - 1)) ** 2 + near ** 2)) ** 1.2

    mono = fx.sweep_lowpass(mono, air)
    pan = lat / dist
    direct = mono * g[:, None] * np.stack([np.sqrt((1 - pan) / 2), np.sqrt((1 + pan) / 2)], axis=1)
    return direct.astype(np.float32), mono * 0.35


def build(out_dir: Path, stems: bool = False) -> dict:
    tl = Timeline(BPM, BARS * 4, tail_s=7.0)
    n = tl.n

    # --- the record (1:1) -----------------------------------------------------------------------------------
    rec = dict(bpm=BPM, semis=KEY, curve=None, swing=SWING)
    loop = Record(SRC, ["Track 4", "strings low"], programs={"Track 4": 61}, vel={"strings low": 0.7}, **rec)
    pad = Record(SRC, ["Track 6"], programs={"Track 6": 89}, **rec)                # warm pad for the sweep
    sax = Record(SRC, ["Track 3"], programs={"Track 3": 66}, **rec)                 # the horn tune on tenor sax
    whistle = Record(SRC, ["Track 7"], programs={"Track 7": 78}, **rec)
    mbox = Record(SRC, ["Track 8"], programs={"Track 8": 10}, **rec)                # music box
    brass = Record(SRC, ["piano1"], **rec)                                          # the trumpet calls

    place(tl, "sample", loop, 0, 68, tail=2)
    place(tl, "pad", pad, 28, 62, tail=2)
    place(tl, "sax", sax, 8, 24)
    place(tl, "sax", sax, 28, 68, tail=4)
    tl.put("whistle", edge(whistle.beats(27 * 4 + 3, 44 * 4, tail_beats=2), 2, 60), 27 * 4 + 3)
    place(tl, "mbox", mbox, 0, 8)
    place(tl, "mbox", mbox, 47, 56, tail=2)
    tl.put("mbox", edge(mbox.beats(0, 16, tail_beats=4), 2, 60), 68 * 4)          # the intro figure again
    place(tl, "brass", brass, 44, 48)
    place(tl, "brass", brass, 64, 68, tail=4)

    # --- the parade: the source's own call, horn and drums at bars 24-28 -----------------------------------
    pm = N.shift(bars_of("piano1", *CALL1), 0, KEY)
    ph = N.shift(bars_of("Track 3", *CALL1), 0, KEY)
    pd = bars_of("Track 10", *CALL1) + bars_of("Track 11", *CALL1)
    parade = fluid([(pm, 56, 0, 0), (ph, 60, 0, 1), (pd, PARADE_KIT, 128, 9)], BPM, tail=3)
    tl.put_full("parade", parade)
    # the parade's snare keeps marching, quietly, under A2 and the last call
    tl.put_full("march", fluid([(bars_of("Track 10", 56, 68, keep={38}), PARADE_KIT, 128, 9)], BPM, tail=2))

    # --- new parts ------------------------------------------------------------------------------------------
    org = []
    for b in range(8, 68):
        if CALL1[0] <= b < CALL1[1]:
            continue
        vc = [p for p in PT.VOICINGS[chord(b)] if p >= PT.P["C3"]]
        if 48 <= b < 56:                                   # the breakdown holds the chord
            org += [N.Note(b * 4, 3.9, p, 46) for p in vc]
            continue
        hits = [0.5, 1.5, 2.5, 3.5] if b % 2 == 0 else [0.5, 1.5, 2.5, 3.25, 3.75]
        org += [N.Note(b * 4 + h, 0.22, p, 62 + (6 if h == 0.5 else 0)) for h in hits for p in vc]
    org += [N.Note(68 * 4, 15, p, 44) for p in PT.VOICINGS["Em9"] if p >= PT.P["C3"]]
    org = N.shift(N.humanize(org, BPM, 4, 5, seed=6), 0, KEY)
    tl.put_full("organ", fluid([(org, 17, 0, 4)], BPM, tail=6))                        # percussive organ

    bass = [x for x in SRC.tracks["Track 5"] if x.start < 68 * 4 and not (CALL1[0] * 4 <= x.start < CALL1[1] * 4)]
    for b in list(range(44, 48)) + list(range(64, 68)):           # the calls have no bass: off-beat octaves
        bass += [N.Note(b * 4 + k + 0.5, 0.35, PT.P["E2"] if k % 2 else PT.P["E1"], 92) for k in range(4)]
    bass += [N.Note(68 * 4, 8, PT.P["E1"], 80)]
    bass = N.shift(N.humanize(N.swing(bass, SWING), BPM, 5, 4, seed=9, push_ms=3), 0, KEY)
    tl.put_full("bass", fluid([(bass, 39, 0, 2)], BPM, tail=4))                          # synth bass 2
    tl.put_full("sub", PT.sub(bass, BPM, n))

    drums = {"kick": [], "snare": [], "hats": []}
    for name, b0, nb in FORM:
        g = GROOVE.get(name)
        if name == "outro":
            nb = 2
        if not g:
            continue
        for piece, ns in PT.groove(nb, b0, g, seed=40 + b0, swing_amt=SWING, bpm=BPM).items():
            drums[piece] += ns
    drums["hats"] += [N.Note(b * 4, 2.0, 49, 88) for b in (8, 28, 44, 56, 64)]          # crashes
    tl.kicks = [x.start * 60 / BPM for x in drums["kick"]]
    for piece, ns in drums.items():
        tl.put_full(piece, fluid([(ns, KIT, 128, 9)], BPM, tail=2.5))

    # --- processing -----------------------------------------------------------------------------------------
    b = tl.bus

    def ride(t):        # record filter: opens through the intro, muffles under the parade, closes in the outro
        i1, c0, c1, o0, o1 = bar_s(8), bar_s(CALL1[0]), bar_s(CALL1[1]), bar_s(68), bar_s(71.5)
        f = np.where(t < i1, 500 * (4800 / 500) ** np.clip(t / i1, 0, 1) ** 1.4, 4800.0)
        f = np.where((t > c0 - 0.15) & (t < c1 - 0.05), 650.0, f)
        return np.where(t > o0, 4800 * (380 / 4800) ** np.clip((t - o0) / (o1 - o0), 0, 1), f)

    t = np.arange(n) / SR
    dip = np.interp(t, [bar_s(23.9), bar_s(24), bar_s(27.9), bar_s(28)], [1, 0.45, 0.45, 1])[:, None]
    s = fx.highpass(b["sample"], 170, 2)
    s = fx.bitcrush(s, 12, 26040, mix=0.75)
    s = fx.saturate(fx.shelf_tilt(s, 700, -1.0), 1.5)
    b["sample"] = fx.wow_flutter(fx.sweep_lowpass(s, ride), wow_ms=1.6, wow_hz=0.5, seed=31) * dip
    b["pad"] = fx.widen(fx.lowpass(fx.highpass(b["pad"], 240, 2), 3200, 2), 1.6)
    sx = fx.saturate(fx.shelf_tilt(fx.lowpass(fx.highpass(b["sax"], 160, 2), 4600, 2), 900, -0.7), 1.4)
    b["sax"] = fx.wow_flutter(fx.bitcrush(sx, 12, 26040, mix=0.5), wow_ms=1.6, wow_hz=0.5, seed=32)
    wh = fx.lowpass(fx.highpass(b["whistle"], 500, 2), 7000, 2)
    b["whistle"] = fx.pan(fx.wow_flutter(fx.bitcrush(wh, 12, 26040, mix=0.4), seed=33), 0.25)
    mb = fx.bitcrush(fx.highpass(b["mbox"], 300, 2), 12, 26040, mix=0.7)
    b["mbox"] = fx.widen(fx.detune_chorus(fx.sweep_lowpass(mb, ride), cents=8, rate=0.4, mix=0.4), 1.4)
    br = fx.saturate(fx.bitcrush(fx.lowpass(fx.highpass(b["brass"], 200, 2), 5000, 2), 12, 26040), 1.5)
    b["brass"] = fx.wow_flutter(br, seed=34)
    og = fx.detune_chorus(fx.lowpass(fx.highpass(b["organ"], 200, 2), 3600, 2), cents=7, rate=0.5, mix=0.4)
    b["organ"] = fx.sweep_lowpass(fx.saturate(og, 1.4), ride)
    direct, send = parade_pass(b["parade"], bar_s(CALL1[0]) - 0.2, bar_s(CALL1[1]) + 0.3)
    b["parade"] = direct
    b["march"] = fx.pan(fx.bitcrush(fx.highpass(b["march"], 220, 2), 12, 26040), -0.35)

    bb = fx.saturate(fx.lowpass(b["bass"], 900, 2), 1.8)
    b["bass"] = fx.mono_below(fx.compress(bb, -20, 3, 10, 120), 300)
    b["sub"] = fx.lowpass(b["sub"], 100, 2)

    kk = fx.saturate(fx.lowpass(b["kick"], 5000, 2), 2.0)
    b["kick"] = fx.compress(kk, -14, 3, 3, 90)
    sn = fx.bitcrush(fx.highpass(b["snare"], 180, 2), 12, 26040)
    b["snare"] = fx.lowpass(fx.saturate(sn, 1.6), 9000, 2)
    hh = fx.bitcrush(fx.lowpass(fx.highpass(b["hats"], 900, 2), 7000, 2), 12, 26040)
    b["hats"] = fx.autopan(hh, rate=BPM / 60 / 8, depth=0.35)                       # two bars a drift

    lv = {k: fx.db(LEVELS[k]) / max(1e-9, fx.db(fx.gated_rms_db(b[k]))) for k in LEVELS}
    for k in LEVELS:
        b[k] = b[k] * lv[k]
    send = send * lv["parade"]

    # space: a park bandstand plate on the sax, whistle and brass, a short room for the record, organ and drums;
    # a dotted-8th ping-pong on the whistle, a quarter echo on the music box; the parade gets its own wide reverb
    plate = fx.reverb(b["sax"] * 0.5 + b["whistle"] * 0.6 + b["brass"] * 0.5 + b["mbox"] * 0.5, seconds=2.4,
                      damp_hz=3400, predelay_ms=25, seed=35)
    room = fx.reverb(b["sample"] * 0.3 + b["organ"] * 0.35 + b["pad"] * 0.4 + b["snare"] * 0.25
                     + b["march"] * 0.4, seconds=1.2, damp_hz=3600, predelay_ms=12, seed=36)
    park = fx.reverb(send, seconds=2.8, damp_hz=2600, predelay_ms=40, hp=200, seed=37)
    echo = fx.tape_delay(b["whistle"], 0.75, BPM, feedback=0.35, repeats=4, lp=3000, pingpong=True)
    qecho = fx.tape_delay(b["mbox"], 1.0, BPM, feedback=0.3, repeats=3, lp=2600, pingpong=False)

    pump = fx.duck(n, tl.kicks, depth_db=7.0, attack_ms=4, release_ms=200)[:, None]   # the house pump
    duck = fx.duck(n, tl.kicks, depth_db=2.5, release_ms=200)[:, None]
    music = (b["sample"] + b["pad"] + b["sax"] + b["whistle"] + b["mbox"] + b["brass"]) * duck + b["organ"] * pump
    low = (b["bass"] + b["sub"]) * fx.duck(n, tl.kicks, depth_db=5.0, release_ms=150)[:, None]
    drum_bus = fx.compress(b["kick"] + b["snare"] + b["hats"], -16, 2.5, 8, 140, makeup_db=1)
    drum_bus = fx.tape(drum_bus, 1.3, 11000) + b["march"]

    # ambience: the fountain by the bench (louder when the band stops for the parade), sparse vinyl, hiss
    secs = n / SR
    amt = np.interp(t, [0, bar_s(7.5), bar_s(8.5), bar_s(23.5), bar_s(24.2), bar_s(27.5), bar_s(28.2),
                        bar_s(68), secs], [1.0, 1.0, 0.5, 0.5, 1.3, 1.3, 0.5, 0.5, 1.2])[:, None]
    air = AMB.fountain(secs, -42, seed=38, pan=-0.3) * amt
    vinyl = AMB.vinyl(secs, -48, seed=39, crackle_rate=2.0)
    hiss = AMB.hiss(secs, -52, seed=40)

    mix = (music + low + drum_bus * 0.85 + b["parade"] + park * 0.6 + plate * 0.4 + room * 0.4 + echo * 0.28
           + qecho * 0.22 + air + vinyl + hiss)
    mix = fx.wow_flutter(mix, wow_ms=0.5, wow_hz=0.35, flutter_ms=0.04, drift_ms=0.4, seed=30)
    end = int((bar_s(BARS) + 6.0) * SR)
    mix = fx.fade(mix[:end], fin_s=0.8, fout_s=4.5)
    out = M.master(mix, -14.0, -1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{FILE}.wav"
    M.write(path, out, tags={"title": TITLE, "artist": "Gielinor Lo-Fi", "album": "Gielinor Lo-Fi",
                             "comment": "Lo-fi reinterpretation of 'Fanfare' (Ian Taylor, Jagex) from Old School RuneScape"})
    if stems:
        sd = out_dir / "stems"
        sd.mkdir(exist_ok=True)
        import soundfile as sf
        for k, x in {**b, "plate": plate, "room": room, "park": park, "echo": echo, "air": air}.items():
            sf.write(str(sd / f"{k}.wav"), x[:end], SR, subtype="PCM_16")
    rep = M.report(out, BPM, [(nm, b0, b0 + nb) for nm, b0, nb in FORM])
    rep["file"] = str(path)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    print(json.dumps(build(ROOT / "build" / "fanfare", stems="--stems" in sys.argv), indent=1))
