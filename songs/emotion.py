"""Gielinor Lo-Fi - Emotion (after Ian Taylor's 'Emotion', Old School RuneScape).

A late night by the fire at 84 bpm in Bb minor. The 4/4 MIDI is pressed to a record 1:1 and varispeeded down one
semitone, so it plays at ~89 bpm, next to the source's 90. New parts: a Tine & FM EP stack, fretless bass, a brush
kit with a shaker groove, and vibes doubling the sitar in A3. The source's own tom pickups lead into both B
sections; the one added move is a granular freeze of the F# chord (a held breath) before B1.
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

TITLE = "Emotion (By the Hearth)"
FILE = "Emotion (By the Hearth)"
BPM = 84.0
KEY = -1                      # B minor -> Bb minor
KIT = 40                      # GeneralUser GS brush kit
SWING = 0.56                  # the record and the kit swing together (the harp's 16ths stay in the pocket)
SRC = N.load(ROOT / "source" / "emotion.mid")

A = [[("Bm9", 4)]] * 2 + [[("E9", 4)]] * 2
A2 = [[("Bm9", 4)]] * 2 + [[("F#7", 4)]] * 2
B = [[("Bmaj9", 3), ("Amaj9", 1)], [("Amaj9", 2), ("F#9", 2)], [("F#9", 4)]]     # the source's 3-bar group
T = [[("Bm11", 4)]] * 2 + [[("F#7b13", 4)]] * 2

# name, first song bar, bars
FORM = [("intro", 0, 4), ("A1", 4, 8), ("A2", 12, 8), ("B1", 20, 12), ("A3", 32, 8), ("horns", 40, 8),
        ("B2", 48, 12), ("outro", 60, 4)]
BARS = 64
CHART = A + A * 2 + A2 * 2 + B * 4 + A * 2 + T * 2 + B * 4 + A
# stem balance as gated RMS in dBFS before the master (the record leads)
LEVELS = {"sample": -21, "pad": -31, "flute": -25, "horn": -25, "keys": -27, "lead": -29, "bass": -27, "sub": -32,
          "kick": -23, "snare": -27, "hats": -36, "toms": -29, "freeze": -27}

FREEZE = (19 * 4 + 2.5, 20 * 4)          # the held breath: everything out but the frozen F# chord

# Emotion's groove: a brush kit with a shaker in place of hats, the kick stepping 1, a+2, 3 (1, e+1, 3, a+3 on odd
# bars), the brush slap on 2 and 4 with a clap layered on 4, sidestick ghosts and a pedal hat on 1 and 3.
FULL = [
    dict(bus="kick", piece=PT.KICK, at=[0, 6, 10], odd=[0, 3, 8, 11], fill=[0, 6, 10, 13], vel=(84, 96),
         accent={0: 106}),
    dict(bus="snare", piece=PT.SNARE, at=[4, 12], vel=96),
    dict(bus="snare", piece=PT.CLAP, at=[12], vel=58),
    dict(bus="snare", piece=PT.STICK, at=[7, 15], fill=[7, 14, 15], vel=(26, 40), p=0.4),
    dict(bus="hats", piece=PT.SHAKER, at=list(range(16)), vel=(28, 44), accent={2: 80, 6: 80, 10: 80, 14: 80},
         p=0.92),
    dict(bus="hats", piece=PT.PHH, at=[0, 8], vel=(42, 54)),
]
# the B sections: kick on 1 and the and of 3, sidestick backbeat, the shaker only on the off-8ths
LIGHT = [
    dict(bus="kick", piece=PT.KICK, at=[0, 10], odd=[0, 7, 11], vel=(80, 92), accent={0: 100}),
    dict(bus="snare", piece=PT.STICK, at=[4, 12], vel=(84, 94)),
    dict(bus="hats", piece=PT.SHAKER, at=[2, 6, 10, 14], vel=(62, 78)),
    dict(bus="hats", piece=PT.PHH, at=[0, 8], vel=(36, 46)),
]
# the horn section opens up: an open hat on the last 8th of odd bars
HORNS = FULL + [dict(bus="hats", piece=PT.OHH, at=[], odd=[14], vel=(44, 52), dur=0.4)]
GROOVE = {"A1": FULL, "A2": FULL, "B1": LIGHT, "A3": FULL, "horns": HORNS, "B2": LIGHT}


def bar_s(b):
    return b * 4 * 60 / BPM


def place_bars(tl, bus, rec, src_bar, song_bar, nbars, gain=1.0):
    """Lay record bars on the grid one chop per bar (each chop de-clicked, a short tail into the next)."""
    for k in range(nbars):
        tl.put(bus, edge(rec.bar(src_bar + k, tail_beats=0.2), 2, 25), (song_bar + k) * 4, gain)


def freeze(src, seconds, grain_ms=150.0, hop_ms=30.0, seed=40):
    """Hold a moment of the record: Hann grains drawn at random from `src`, overlapped into a steady cloud."""
    rng = np.random.default_rng(seed)
    g, hop = int(grain_ms * SR / 1000), int(hop_ms * SR / 1000)
    win = np.hanning(g)[:, None]
    out = np.zeros((int(seconds * SR) + g, 2), dtype=np.float32)
    for i0 in range(0, len(out) - g, hop):
        s = rng.integers(0, len(src) - g)
        out[i0:i0 + g] += src[s:s + g] * win
    return out * (hop / (g * 0.5))


def hearth(seconds, level_db=-40.0, seed=31):
    """A fire in the grate: a low flickering roar, soft wide pops and a rare bright snap (sparse, so never rain)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    knots = rng.uniform(0.55, 1.0, int(seconds * 3) + 2)
    flicker = np.interp(t, np.arange(len(knots)) / 3, knots)[:, None]
    roar = fx.lowpass(fx.highpass(AMB._pink(n, rng), 60, 2), 420, 2) * flicker
    pops = fx.lowpass(fx.highpass(AMB._clicks(n, 2.2, 1.0, rng, (8, 30)), 250, 2), 2600, 2)
    snaps = fx.bandpass(AMB._clicks(n, 0.2, 2.5, rng, (2, 6)), 900, 5000, 1)
    x = roar * 0.6 + pops * 1.2 + snaps * 0.6
    return (x / max(1e-9, np.sqrt(np.mean(x ** 2))) * fx.db(level_db)).astype(np.float32)


def build(out_dir: Path, stems: bool = False) -> dict:
    tl = Timeline(BPM, BARS * 4, tail_s=7.0)
    n = tl.n

    # --- the record -----------------------------------------------------------------------------------------
    rec = dict(bpm=BPM, semis=KEY, curve=None, swing=SWING)
    hook = Record(SRC, ["harp", "syn vox", "sitar", "music box"], vel={"harp": 0.8}, **rec)
    pad = Record(SRC, ["slow strings", "choir"], **rec)
    flute = Record(SRC, ["flute", "steel guitar"], **rec)
    horn = Record(SRC, ["trumpet", "epiano1"], vel={"epiano1": 0.7}, **rec)
    toms = Record(SRC, ["drums"], **rec)

    place_bars(tl, "sample", hook, 0, 0, 4, 0.9)                         # intro: harp and voices, filtered open
    for src_bar, song_bar, nb in ((8, 4, 8), (16, 12, 8), (24, 20, 12), (36, 32, 8), (44, 40, 8), (60, 48, 12),
                                  (0, 60, 4)):
        place_bars(tl, "sample", hook, src_bar, song_bar, nb, 0.85 if song_bar == 60 else 1.0)
        if song_bar != 60:
            place_bars(tl, "pad", pad, src_bar, song_bar, nb)
    place_bars(tl, "flute", flute, 24, 20, 12)                            # B1 and B2: the flute's own tune
    place_bars(tl, "flute", flute, 60, 48, 12)
    place_bars(tl, "horn", horn, 36, 32, 8)                               # A3: the source's epiano
    place_bars(tl, "horn", horn, 44, 40, 8)                               # the trumpet section, at its own speed
    place_bars(tl, "toms", toms, 22, 18, 2)                               # the source's tom pickups into B1 / B2
    place_bars(tl, "toms", toms, 50, 46, 2)

    hold = freeze(hook.beats(23 * 4, 23 * 4 + 2), (FREEZE[1] - FREEZE[0] + 3) * 60 / BPM)
    tt = np.arange(len(hold)) / SR
    bs = 60 / BPM
    env = np.minimum(1, tt / (1.2 * bs)) * np.where(tt < 2.5 * bs, 1, np.exp(-(tt - 2.5 * bs) / 1.1))
    tl.put("freeze", hold * env[:, None], FREEZE[0] - 0.5)

    # --- new parts ------------------------------------------------------------------------------------------
    def clear(ns):
        return [x for x in ns if not (FREEZE[0] - 0.05 <= x.start < FREEZE[1])]

    keys = clear(PT.comp(CHART[2:63], start_bar=2, vel=52, seed=8, bpm=BPM))
    keys = [N.Note(x.start, min(x.dur, FREEZE[0] - x.start) if x.start < FREEZE[0] < x.end else x.dur, x.pitch, x.vel)
            for x in keys]
    keys += PT.comp([[("Bm9", 4)]], start_bar=63, vel=48, style="pad", bpm=BPM)
    keys = [N.Note(x.start, x.dur + (6 if x.start >= 63 * 4 else 0), x.pitch, x.vel) for x in keys]
    tl.put_full("keys", fluid([(N.shift(keys, 0, KEY), 4, 11, 0)], BPM, tail=8))       # Tine & FM EPs
    sitar = N.top_line(N.window(SRC.tracks["sitar"], 36, 44))
    lead = PT.melody_double(N.swing(N.shift(sitar, 32 * 4), SWING), octave=1, vel=46)    # A3 vibes double
    tl.put_full("lead", fluid([(N.shift(lead, 0, KEY), 11, 0, 1)], BPM, tail=4))

    bass = PT.bass(CHART[4:63], start_bar=4, vel=84, seed=9)
    bass = clear([x for x in bass if x.start < 62 * 4]) + [N.Note(63 * 4, 6, PT.P["B1"], 78)]
    bass = N.humanize(N.swing(bass, SWING), BPM, 6, 4, seed=10, push_ms=4)
    bass = N.shift(bass, 0, KEY)
    tl.put_full("bass", fluid([(bass, 35, 0, 2)], BPM, tail=4))                       # fretless
    tl.put_full("sub", PT.sub(bass, BPM, n))

    drums = {"kick": [], "snare": [], "hats": []}
    for name, b0, nb in FORM:
        lines = GROOVE.get(name)
        if name == "outro":
            lines, nb = [x for x in LIGHT if x["bus"] == "hats"], 2
        if not lines:
            continue
        for piece, ns in PT.groove(nb, b0, lines, seed=20 + b0, swing_amt=SWING, bpm=BPM).items():
            drums[piece] += clear(ns)
    drums["snare"] = [N.Note(x.start + 0.03, x.dur, x.pitch, x.vel) for x in drums["snare"]]   # the brush drags
    tl.kicks = [x.start * 60 / BPM for x in drums["kick"]]
    for piece, ns in drums.items():
        tl.put_full(piece, fluid([(ns, KIT, 128, 9)], BPM, tail=2))

    # --- processing -----------------------------------------------------------------------------------------
    b = tl.bus
    i0, i1 = tl.i(FREEZE[0]), tl.i(FREEZE[1])
    gap = np.ones(n, dtype=np.float32)            # the held breath: the record and pads out, the cloud alone
    fd = int(0.04 * SR)
    gap[i0:i1] = 0.0
    gap[i0 - fd:i0] = np.linspace(1, 0, fd)
    gap[i1 - fd // 2:i1] = np.linspace(0, 1, fd // 2)
    for k in ("sample", "pad", "toms"):
        b[k] *= gap[:, None]

    def ride(t):        # record filter: opens through the intro, closes through the outro
        f = np.full_like(t, 5600.0)
        i1s, o0, o1 = bar_s(4), bar_s(60), bar_s(63.5)
        f = np.where(t < i1s, 420 * (5600 / 420) ** np.clip(t / i1s, 0, 1) ** 1.4, f)
        f = np.where(t > o0, 5600 * (320 / 5600) ** np.clip((t - o0) / (o1 - o0), 0, 1), f)
        return f

    s = fx.highpass(b["sample"], 150, 2)
    s = fx.bitcrush(s, 12, 26040, mix=0.7)
    s = fx.saturate(s, 1.4)
    s = fx.sweep_lowpass(s, ride)
    b["sample"] = fx.wow_flutter(s, wow_ms=2.0, wow_hz=0.4, seed=12)        # cassette: a slower, deeper wow
    p = fx.lowpass(fx.highpass(b["pad"], 220, 2), 3600, 2)
    b["pad"] = fx.widen(fx.wow_flutter(p, wow_ms=2.0, wow_hz=0.4, seed=13), 1.5)
    fl = fx.saturate(fx.lowpass(fx.highpass(b["flute"], 260, 2), 5600, 2), 1.3)
    b["flute"] = fx.wow_flutter(fx.bitcrush(fl, 12, 26040, mix=0.5), wow_ms=2.0, wow_hz=0.4, seed=14)
    hn = fx.saturate(fx.lowpass(fx.highpass(b["horn"], 240, 2), 4600, 2), 1.5)
    b["horn"] = fx.wow_flutter(fx.bitcrush(hn, 12, 26040, mix=0.6), wow_ms=2.0, wow_hz=0.4, seed=15)
    b["toms"] = fx.saturate(fx.lowpass(b["toms"], 6000, 2), 1.4)
    b["freeze"] = fx.widen(fx.lowpass(fx.highpass(b["freeze"], 200, 2), 4200, 2), 1.6)

    r = fx.detune_chorus(b["keys"], cents=9, rate=0.18, mix=0.35)
    r = fx.saturate(fx.lowpass(fx.highpass(r, 180, 2), 5000, 2), 1.3)
    b["keys"] = fx.wow_flutter(r, wow_ms=1.6, wow_hz=0.4, seed=16)
    b["lead"] = fx.pan(fx.wow_flutter(fx.lowpass(b["lead"], 5000, 2), seed=17), -0.2)

    bb = fx.saturate(fx.lowpass(b["bass"], 900, 2), 1.5)
    b["bass"] = fx.mono_below(fx.compress(bb, -20, 3, 15, 140), 300)
    b["sub"] = fx.lowpass(b["sub"], 110, 2)

    kk = fx.saturate(fx.lowpass(b["kick"], 4000, 2), 2.2)
    b["kick"] = fx.compress(kk, -14, 3, 3, 90)
    sn = fx.bitcrush(fx.highpass(b["snare"], 180, 2), 12, 26040)
    b["snare"] = fx.lowpass(fx.saturate(sn, 1.6), 8500, 2)
    hh = fx.bitcrush(fx.lowpass(fx.highpass(b["hats"], 900, 2), 9000, 2), 12, 26040)
    b["hats"] = fx.pan(hh, -0.3)

    for k, target in LEVELS.items():
        b[k] = fx.level(b[k], target)

    # space: a warm small room, a long plate-ish tail for the freeze, a quarter-note echo on flute and horn
    send = (b["sample"] * 0.3 + b["keys"] * 0.45 + b["pad"] * 0.6 + b["flute"] * 0.45 + b["horn"] * 0.4
            + b["lead"] * 0.5 + b["snare"] * 0.2 + b["toms"] * 0.3)
    verb = fx.reverb(send, seconds=2.0, damp_hz=2800, predelay_ms=18)
    bloom = fx.reverb(b["freeze"], seconds=5.0, damp_hz=3200, predelay_ms=40, seed=11)
    echo = fx.tape_delay(b["flute"] + b["horn"] * 0.7 + b["lead"] * 0.6, 1.0, BPM, feedback=0.3, repeats=3,
                         lp=2400)

    duck = fx.duck(n, tl.kicks, depth_db=2.0, release_ms=220)[:, None]
    music = (b["sample"] + b["pad"] + b["flute"] + b["horn"] + b["keys"] + b["lead"]) * duck
    low = (b["bass"] + b["sub"]) * fx.duck(n, tl.kicks, depth_db=3.5, release_ms=150)[:, None]
    drum_bus = fx.compress(b["kick"] + b["snare"] + b["hats"] + b["toms"], -16, 2.5, 12, 160, makeup_db=1)
    drum_bus = fx.tape(drum_bus, 1.25, 10500)

    # ambience: the fire, louder where the beat is out; cassette hiss, no vinyl
    secs = n / SR
    t = np.arange(n) / SR
    fire_amt = np.interp(t, [0, bar_s(3.5), bar_s(4.5), bar_s(19.5), bar_s(20), bar_s(20.5), bar_s(60), secs],
                         [1.0, 1.0, 0.5, 0.5, 0.9, 0.5, 0.5, 1.1])[:, None]
    fire = hearth(secs, -40, seed=31) * fire_amt
    hiss = AMB.hiss(secs, -52, seed=32)

    mix = music + low + drum_bus * 0.85 + verb * 0.5 + b["freeze"] * 0.8 + bloom * 0.7 + echo * 0.3 + fire + hiss
    mix = fx.wow_flutter(mix, wow_ms=0.5, wow_hz=0.35, flutter_ms=0.04, drift_ms=0.4, seed=19)
    end = int((bar_s(BARS) + 6.5) * SR)
    mix = fx.fade(mix[:end], fin_s=1.5, fout_s=4.0)
    out = M.master(mix, -14.0, -1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{FILE}.wav"
    M.write(path, out, tags={"title": TITLE, "artist": "Gielinor Lo-Fi", "album": "Gielinor Lo-Fi",
                             "comment": "Lo-fi reinterpretation of 'Emotion' (Ian Taylor, Jagex) from Old School RuneScape"})
    if stems:
        sd = out_dir / "stems"
        sd.mkdir(exist_ok=True)
        import soundfile as sf
        for k, x in {**b, "verb": verb, "bloom": bloom, "echo": echo, "fire": fire}.items():
            sf.write(str(sd / f"{k}.wav"), x[:end], SR, subtype="PCM_16")
    rep = M.report(out, BPM, [(nm, b0, b0 + nb) for nm, b0, nb in FORM])
    rep["file"] = str(path)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    print(json.dumps(build(ROOT / "build" / "emotion", stems="--stems" in sys.argv), indent=1))
