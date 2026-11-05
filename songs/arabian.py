"""Gielinor Lo-Fi - Arabian (after Ian Taylor's 'Arabian', Old School RuneScape).

Al Kharid after dark: Middle Eastern lo-fi hip-hop at 72 bpm in Bb phrygian. The 150 bpm MIDI is pressed half-time
(each source bar is two song beats) and varispeeded down a semitone, so the tune itself runs at ~152 bpm, its own
speed, over a 72 bpm head-nod beat. The record re-voices the source: the guitar ostinato stays, the pan flute is
pressed as a breathy ney (shakuhachi), the kalimba B section as a santur (dulcimer). New parts: a phased Rhodes, a
synth bass and sub, a shenai doubling the ney in A3 into a ping-pong echo, an oud doubling the ostinato in A3, and a
kit with a darbuka layer (maqsum in the A sections, a 3+3+2 malfuf in B), a riq that wanders across the field and
a finger-cymbal every four bars. The move: the band drops out and the source's own drum tracks play alone under an
oud taqsim on a held drone, until a darbuka roll throws the beat back in for B2.
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

TITLE = "Arabian (Al Kharid)"
FILE = "Arabian (Al Kharid)"
BPM = 72.0
KEY = -1                      # B phrygian -> Bb phrygian
KIT = 16                      # GeneralUser GS power kit
SWING = 0.56
SRC = N.load(ROOT / "source" / "arabian.mid")
DOUM, TEK, KA, ZILL, CRASH = 64, 63, 62, 81, 49    # low conga / open hi conga / mute hi conga as darbuka, triangle

A = [[("Bm11", 4)], [("Bphr", 4)], [("Bm11", 4)], [("Em9", 4)]] * 2
B = [[("Em9", 4)], [("Em11", 4)]] * 2

# name, first song bar, bars, source bar (two source bars per song bar)
FORM = [("intro", 0, 4, 4), ("A1", 4, 8, 12), ("B1", 12, 4, 28), ("A2", 16, 8, 44), ("break", 24, 4, None),
        ("B2", 28, 8, 60), ("A3", 36, 8, 12), ("outro", 44, 4, 76)]
BARS = 48
CHART = [[("Bm11", 4)], [("Bphr", 4)]] * 2 + A + B + A + [[("Bphr", 4)]] * 4 + B * 2 + A + [[("Em9", 4)]] * 4
LEVELS = {"sample": -21, "pad": -31, "ney": -24, "santur": -25, "shenai": -29, "oud": -27, "keys": -29,
          "perc": -24, "bass": -27, "sub": -33, "kick": -22, "snare": -26, "hats": -36, "darb": -27, "riq": -33}
BREAK = (24 * 4, 28 * 4)

# Arabian's groove: a head-nod kick on 1, e+2 and the and of 3 (1, a+1, and of 3 on odd bars) and a fat snare on 2
# and 4, quiet off-8th hats; on top a darbuka maqsum (doum on 1 and 3, tek on e+1, e+2 and 4) with ka ghosts, and
# a riq on the off-8ths. The B sections play a 3+3+2 malfuf on the darbuka under handclaps.
MAQSUM = [
    dict(bus="kick", piece=PT.KICK, at=[0, 7, 10], odd=[0, 3, 10], fill=[0, 3, 10, 13], vel=(88, 100),
         accent={0: 110}),
    dict(bus="snare", piece=PT.SNARE, at=[4, 12], vel=96),
    dict(bus="hats", piece=PT.CHH, at=[2, 6, 10, 14], vel=(34, 44)),
    dict(bus="darb", piece=DOUM, at=[0, 8], odd=[0, 6, 8], vel=(80, 92)),
    dict(bus="darb", piece=TEK, at=[2, 6, 12], fill=[2, 6, 12, 13, 14, 15], vel=(70, 86)),
    dict(bus="darb", piece=KA, at=[4, 10, 14, 15], vel=(34, 50), p=0.6),
    dict(bus="riq", piece=PT.TAMB, at=[2, 6, 10, 14], vel=(56, 70)),
    dict(bus="riq", piece=PT.TAMB, at=[3, 11], vel=(26, 36), p=0.4),
]
MALFUF = [
    dict(bus="kick", piece=PT.KICK, at=[0, 10], odd=[0, 6, 10], vel=(84, 96), accent={0: 104}),
    dict(bus="snare", piece=PT.CLAP, at=[4, 12], vel=(72, 84)),
    dict(bus="darb", piece=DOUM, at=[0], vel=(82, 92)),
    dict(bus="darb", piece=TEK, at=[6, 12], fill=[6, 12, 14], vel=(72, 86)),
    dict(bus="darb", piece=KA, at=[2, 4, 8, 10, 14], vel=(30, 46), p=0.5),
    dict(bus="riq", piece=PT.TAMB, at=[4, 12], vel=(66, 76)),
]
# A3: everything, plus a clap on the 4 and an open hat into each odd bar
BIG = MAQSUM + [dict(bus="snare", piece=PT.CLAP, at=[12], vel=58),
                dict(bus="hats", piece=PT.OHH, at=[], odd=[14], vel=(44, 52), dur=0.4)]
GROOVE = {"A1": MAQSUM, "B1": MALFUF, "A2": MAQSUM, "B2": MALFUF, "A3": BIG}

# the oud taqsim over the break (source key, B phrygian; song beats from the start of the break). Long notes get a
# tremolo; (beat, beats, note, vel, tremolo)
TAQSIM = [
    (0.0, 1.5, "B2", 84, True), (1.5, 0.25, "C3", 70, False), (1.75, 0.25, "D3", 72, False),
    (2.0, 1.0, "C3", 78, True), (3.0, 0.5, "B2", 70, False), (3.5, 0.5, "A2", 66, False),
    (4.0, 2.0, "B2", 82, True), (6.0, 0.25, "D3", 70, False), (6.25, 0.25, "E3", 72, False),
    (6.5, 0.5, "F#3", 80, False), (7.0, 1.0, "E3", 74, True),
    (8.0, 0.5, "G3", 84, False), (8.5, 0.5, "F#3", 76, False), (9.0, 0.25, "E3", 70, False),
    (9.25, 0.25, "D3", 68, False), (9.5, 0.5, "C3", 74, False), (10.0, 2.0, "D3", 80, True),
    (12.0, 0.75, "C3", 78, False), (12.75, 0.25, "D3", 66, False), (13.0, 0.5, "C3", 72, False),
    (13.5, 0.5, "A2", 68, False), (14.0, 1.5, "B2", 86, True),
]


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


def tremolo(notes, rate=8):
    """Oud tremolo: long notes become fast repeated picks (rate per beat), the first one strongest."""
    out = []
    for x, trem in notes:
        if not trem:
            out.append(x)
            continue
        k = int(x.dur * rate)
        for j in range(k):
            out.append(N.Note(x.start + j / rate, 1 / rate * 1.1, x.pitch, max(30, int(x.vel * (1 if j == 0 else 0.7)))))
    return out


def build(out_dir: Path, stems: bool = False) -> dict:
    tl = Timeline(BPM, BARS * 4, tail_s=7.0)
    n = tl.n

    # --- the record (half-time) -----------------------------------------------------------------------------
    rec = dict(bpm=BPM, semis=KEY, dst_bar=2, curve=None, swing=SWING)
    loop = Record(SRC, ["piano1", "Track 3"], programs={"piano1": 24, "Track 3": 24}, vel={"Track 3": 0.8}, **rec)
    pad = Record(SRC, ["strings low"], programs={"strings low": 49}, **rec)
    ney = Record(SRC, ["Track 5"], programs={"Track 5": 77}, **rec)                   # shakuhachi as a ney
    santur = Record(SRC, ["Track 6"], programs={"Track 6": 15}, **rec)                # dulcimer as a santur
    perc = Record(SRC, ["Track 10", "Track 11"], **rec)                               # the source's own drums

    for name, b0, nb, sb in FORM:
        if sb is None:
            continue
        if name in ("intro", "A1", "A2", "A3"):
            place_bars(tl, "sample", loop, sb, b0, nb, 0.9 if name == "intro" else 1.0)
        if name not in ("intro", "outro"):
            place_bars(tl, "pad", pad, sb, b0, nb)
        if name in ("B1", "B2", "outro"):
            place_bars(tl, "santur", santur, sb, b0, nb)
    for b0 in (4, 16, 36):                     # the ney's tune, with its pickup from the bar before
        sb = 44 if b0 == 16 else 12
        x = ney.beats((sb - 1) * 2, (sb + 16) * 2, tail_beats=1.5)
        tl.put("ney", edge(x, 2, 60), b0 * 4 - 2)
    place_bars(tl, "perc", perc, 0, 24, 2)                                         # the break: the source's drums
    place_bars(tl, "perc", perc, 84, 26, 2)

    # --- new parts ------------------------------------------------------------------------------------------
    def no_break(ns):
        return [x for x in ns if not (BREAK[0] <= x.start < BREAK[1])]

    keys = PT.comp(CHART[2:24] + CHART[28:44], start_bar=2, vel=48, seed=6, bpm=BPM)
    keys = [N.Note(x.start + (4 * 4 if x.start >= 24 * 4 else 0), x.dur, x.pitch, x.vel) for x in keys]
    keys += PT.comp([[("Em9", 4)]] * 2, start_bar=44, vel=42, style="pad", bpm=BPM)
    keys += [N.Note(x.start, x.dur + 5, x.pitch, x.vel) for x in PT.comp([[("Em9", 4)]], 47, 40, "pad", bpm=BPM)]
    tl.put_full("keys", fluid([(N.shift(N.swing(keys, SWING), 0, KEY), 4, 0, 0)], BPM, tail=7))       # Rhodes

    tune = [x for x in SRC.tracks["Track 5"] if 11 * 4 <= x.start < 28 * 4]
    shenai = PT.melody_double(from_src(tune, 12, 36 * 4), octave=-1, vel=58)
    tl.put_full("shenai", fluid([(N.shift(shenai, 0, KEY), 111, 0, 1)], BPM, tail=4))
    ost = [x for x in SRC.tracks["piano1"] if 12 * 4 <= x.start < 28 * 4]
    oud = [(N.Note(x.start, x.dur, x.pitch - 12, 70), False) for x in from_src(ost, 12, 36 * 4)]
    oud += [(N.Note(BREAK[0] + bt, d, PT.P[p], v), tr) for bt, d, p, v, tr in TAQSIM]
    oud = N.humanize(tremolo(oud), BPM, 5, 4, seed=7)
    tl.put_full("oud", fluid([(N.shift(oud, 0, KEY), 106, 0, 3)], BPM, tail=4))        # shamisen as an oud

    bass = no_break(PT.bass(CHART[4:44], start_bar=4, vel=84, seed=8))
    bass = [x for x in bass if x.start < 44 * 4] + [N.Note(44 * 4, 7.5, PT.P["E1"], 76),
                                                   N.Note(46 * 4, 6, PT.P["E1"], 70)]
    bass = N.shift(N.humanize(N.swing(bass, SWING), BPM, 6, 4, seed=9, push_ms=4), 0, KEY)
    tl.put_full("bass", fluid([(bass, 38, 0, 2)], BPM, tail=4))                         # synth bass 1
    drone = [N.Note(BREAK[0], 15.5, PT.P["B1"] + KEY, 70)]
    tl.put_full("sub", PT.sub(bass + drone, BPM, n))

    drums = {"kick": [], "snare": [], "hats": [], "darb": [], "riq": []}
    for name, b0, nb, _ in FORM:
        g = GROOVE.get(name)
        if name == "outro":
            g, nb = [x for x in MALFUF if x["bus"] in ("darb", "riq")], 2
        if not g:
            continue
        for piece, ns in PT.groove(nb, b0, g, seed=40 + b0, swing_amt=SWING, bpm=BPM).items():
            drums[piece] += ns
        drums["riq"] += [N.Note((b0 + k) * 4, 1.5, ZILL, 64) for k in range(0, nb, 4)]     # the zill
    # the roll out of the break: tek and ka alternating in 32nds over the last beat, rising, into a crash
    roll = BREAK[1] - 1
    drums["darb"] += [N.Note(roll + j / 8, 0.1, TEK if j % 2 == 0 else KA, 46 + j * 8) for j in range(8)]
    drums["snare"] += [N.Note(roll + 0.5 + j / 8, 0.1, PT.SNARE, 50 + j * 14) for j in range(4)]
    drums["hats"] += [N.Note(BREAK[1], 2.0, CRASH, 92)]
    tl.kicks = [x.start * 60 / BPM for x in drums["kick"]]
    for piece, ns in drums.items():
        tl.put_full(piece, fluid([(ns, KIT, 128, 9)], BPM, tail=2.5))

    # --- processing -----------------------------------------------------------------------------------------
    b = tl.bus

    def ride(t):        # record filter: opens through the intro, closes through the outro
        i1s, o0, o1 = bar_s(4), bar_s(44), bar_s(47.5)
        f = np.where(t < i1s, 450 * (5200 / 450) ** np.clip(t / i1s, 0, 1) ** 1.3, 5200.0)
        return np.where(t > o0, 5200 * (340 / 5200) ** np.clip((t - o0) / (o1 - o0), 0, 1), f)

    s = fx.highpass(b["sample"], 160, 2)
    s = fx.bitcrush(s, 12, 26040, mix=0.75)
    s = fx.saturate(s, 1.5)
    b["sample"] = fx.wow_flutter(fx.sweep_lowpass(s, ride), wow_ms=1.8, wow_hz=0.45, seed=31)
    b["pad"] = fx.widen(fx.lowpass(fx.highpass(b["pad"], 220, 2), 3400, 2), 1.6)
    ny = fx.saturate(fx.lowpass(fx.highpass(b["ney"], 300, 2), 5600, 2), 1.3)
    b["ney"] = fx.wow_flutter(fx.bitcrush(ny, 12, 26040, mix=0.5), wow_ms=1.8, wow_hz=0.45, seed=32)
    st = fx.bitcrush(fx.lowpass(fx.highpass(b["santur"], 250, 2), 5800, 2), 12, 26040, mix=0.6)
    st = fx.sweep_lowpass(st, ride)
    b["santur"] = fx.widen(fx.detune_chorus(st, cents=6, rate=0.3, mix=0.3), 1.3)
    sh = fx.saturate(fx.bandpass(b["shenai"], 350, 3200, 2), 1.6)                 # a reed through a dusty speaker
    b["shenai"] = fx.pan(fx.wow_flutter(sh, seed=33), 0.3)
    b["oud"] = fx.pan(fx.saturate(fx.lowpass(fx.highpass(b["oud"], 110, 2), 4200, 2), 1.4), -0.3)
    b["perc"] = fx.saturate(fx.bitcrush(fx.lowpass(b["perc"], 7000, 2), 12, 26040), 1.5)

    r = fx.phaser(b["keys"], rate=BPM / 60 / 8, lo=280, hi=2600, stages=4, feedback=0.4, mix=0.7)   # 2 bars a sweep
    r = fx.saturate(fx.lowpass(fx.highpass(r, 180, 2), 4800, 2), 1.3)
    b["keys"] = fx.wow_flutter(r, wow_ms=1.5, wow_hz=0.45, seed=34)

    bb = fx.saturate(fx.lowpass(b["bass"], 800, 2), 1.7)
    b["bass"] = fx.mono_below(fx.compress(bb, -20, 3, 12, 140), 300)
    b["sub"] = fx.lowpass(b["sub"], 100, 2)

    kk = fx.saturate(fx.lowpass(b["kick"], 4500, 2), 2.3)
    b["kick"] = fx.compress(kk, -14, 3, 3, 100)
    sn = fx.bitcrush(fx.highpass(b["snare"], 160, 2), 12, 26040)
    b["snare"] = fx.lowpass(fx.saturate(sn, 1.7), 8500, 2)
    b["hats"] = fx.pan(fx.bitcrush(fx.lowpass(fx.highpass(b["hats"], 900, 2), 9000, 2), 12, 26040), 0.2)
    dk = fx.saturate(fx.bitcrush(fx.highpass(b["darb"], 120, 2), 12, 26040), 1.4)
    b["darb"] = fx.pan(dk, -0.2)
    rq = fx.lowpass(fx.highpass(b["riq"], 1500, 2), 10000, 2)
    b["riq"] = fx.autopan(rq, rate=BPM / 60 / 4, depth=0.7)                                 # one bar a sweep

    for k, target in LEVELS.items():
        b[k] = fx.level(b[k], target)

    # space: a long dark plate on the ney and shenai, a room for the rest; a dotted-8th echo on the ney and a
    # quarter-note ping-pong on the shenai; the break gets a wider room
    plate = fx.reverb(b["ney"] * 0.7 + b["shenai"] * 0.6 + b["oud"] * 0.4, seconds=3.6, damp_hz=3000,
                      predelay_ms=35, seed=35)
    room = fx.reverb(b["sample"] * 0.3 + b["keys"] * 0.45 + b["pad"] * 0.5 + b["santur"] * 0.45
                     + b["snare"] * 0.2 + b["darb"] * 0.25 + b["perc"] * 0.3, seconds=1.7, damp_hz=3200,
                     predelay_ms=15, seed=36)
    echo = fx.tape_delay(b["ney"], 0.75, BPM, feedback=0.35, repeats=4, lp=2600)
    pong = fx.tape_delay(b["shenai"] + b["oud"] * 0.5, 1.0, BPM, feedback=0.4, repeats=5, lp=3000, pingpong=True)

    duck = fx.duck(n, tl.kicks, depth_db=2.2, release_ms=230)[:, None]
    music = (b["sample"] + b["pad"] + b["ney"] + b["santur"] + b["shenai"] + b["oud"] + b["keys"]) * duck
    low = (b["bass"] + b["sub"]) * fx.duck(n, tl.kicks, depth_db=4.0, release_ms=160)[:, None]
    drum_bus = fx.compress(b["kick"] + b["snare"] + b["hats"] + b["darb"] + b["riq"], -16, 2.5, 10, 160,
                           makeup_db=1)
    drum_bus = fx.tape(drum_bus, 1.3, 10500) + b["perc"]

    # ambience: night air over the desert: low wind with no gusts, crickets, hiss; both come up in the break
    secs = n / SR
    t = np.arange(n) / SR
    amt = np.interp(t, [0, bar_s(3.5), bar_s(4.5), bar_s(23.5), bar_s(24.2), bar_s(27.5), bar_s(28.2),
                        bar_s(44), secs], [1.0, 1.0, 0.6, 0.6, 1.2, 1.2, 0.6, 0.6, 1.2])[:, None]
    air = (AMB.wind(secs, -44, seed=37) + AMB.crickets(secs, -43, seed=38)) * amt
    hiss = AMB.hiss(secs, -52, seed=39)

    mix = (music + low + drum_bus * 0.85 + plate * 0.45 + room * 0.45 + echo * 0.28 + pong * 0.3 + air + hiss)
    mix = fx.wow_flutter(mix, wow_ms=0.5, wow_hz=0.35, flutter_ms=0.04, drift_ms=0.4, seed=30)
    end = int((bar_s(BARS) + 6.5) * SR)
    mix = fx.fade(mix[:end], fin_s=1.0, fout_s=4.5)
    out = M.master(mix, -14.0, -1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{FILE}.wav"
    M.write(path, out, tags={"title": TITLE, "artist": "Gielinor Lo-Fi", "album": "Gielinor Lo-Fi",
                             "comment": "Lo-fi reinterpretation of 'Arabian' (Ian Taylor, Jagex) from Old School RuneScape"})
    if stems:
        sd = out_dir / "stems"
        sd.mkdir(exist_ok=True)
        import soundfile as sf
        for k, x in {**b, "plate": plate, "room": room, "echo": echo, "pong": pong, "air": air}.items():
            sf.write(str(sd / f"{k}.wav"), x[:end], SR, subtype="PCM_16")
    rep = M.report(out, BPM, [(nm, b0, b0 + nb) for nm, b0, nb, _ in FORM])
    rep["file"] = str(path)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    print(json.dumps(build(ROOT / "build" / "arabian", stems="--stems" in sys.argv), indent=1))
