"""Gielinor Lo-Fi 02 - Autumn Voyage (after Ian Taylor's 'Autumn Voyage', Old School RuneScape).

A long walk through falling leaves at 72 bpm in C# minor. The original (4/4, E minor, 120 bpm) is pressed to a
'record' and varispeeded down three semitones, but each source bar is laid as two song beats, so the tune runs a
little faster than the original instead of dragging; the chords change twice a bar. The crystal-piano theme
carries the A sections, the vibes take is the B, and the source's own pedal turn leads into its key change (here
C# minor to A minor), which a gust of wind carries over. No tape stops, spin-downs, reversed pickups, echo throws or
timpani: those belong to the other tracks.
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

TITLE = "Autumn Voyage"
FILE = "02 - Autumn Voyage"
BPM = 72.0
KEY = -3                      # E minor -> C# minor (and C minor -> A minor after the key change)
KIT = 8                       # GeneralUser GS room kit
SRC = N.load(ROOT / "source" / "autumn_voyage.mid")

# source tracks (GM programs): piano1 98 crystal theme, strings low 4 EP pad + bass, Track 3 60 horn, Track 4 6
# harpsichord, clarinet 71, string 51, Track 7 24 guitar, Track 8 49 strings, Track 9 8 celesta, Track 10 drums,
# Track 11 52 choir, Track 12 11 vibes. Record bar k = source bars 2k and 2k+1.

# two chords a bar (one per source bar), in source key
E_A = [[("Em9", 2), ("Am9", 2)], [("Am11", 2), ("D69", 2)], [("Em9", 2), ("Am9", 2)], [("Cmaj9", 2), ("Em11", 2)]]
C_A = [[("Cm9", 2), ("Fm9", 2)], [("Fm11", 2), ("Bb69", 2)], [("Cm9", 2), ("Fm9", 2)], [("Abmaj9", 2), ("Cm11", 2)]]
E_PED = [[("Em9", 2), ("Em/D", 2)], [("Cmaj9", 2), ("Em/B", 2)]]
C_PED = [[("Cm9", 2), ("Cm/Bb", 2)], [("Abmaj9", 2), ("Cm/G", 2)]]

FORM = [("intro", 0, 4), ("A1", 4, 8), ("B", 12, 8), ("turn", 20, 4), ("A2", 24, 8), ("C", 32, 4),
        ("A3", 36, 8), ("outro", 44, 4)]
BARS = 48
CHART = E_PED * 2 + E_A * 2 + E_A * 2 + E_PED * 2 + C_A * 2 + C_A + C_A * 2 + C_PED * 2
LEVELS = {"sample": -21, "counter": -28, "strings": -30, "rhodes": -28, "lead": -29, "bass": -28, "sub": -33,
          "kick": -23, "snare": -27, "toms": -30, "hats": -38}

# the voyage's own groove: a rowing stroke - the floor tom pulls on the last 16th into each bar, the kick lands
# with the oar, straight 8th hats (never 16ths, which read as rain); the snare keeps a plain 2 and 4
FULL = [dict(bus="kick", piece=PT.KICK, at=[0, 7, 10], odd=[0, 10], fill=[0, 7, 10, 11], accent={0: 104},
             vel=(82, 94)),
        dict(bus="snare", piece=PT.SNARE, at=[4, 12], vel=(88, 98)),
        dict(bus="snare", piece=PT.SNARE, at=[3, 11], vel=(22, 34), p=0.3),
        dict(bus="toms", piece=PT.LO_FLOOR, at=[15], odd=[14, 15], fill=[], vel=(58, 72)),
        dict(bus="toms", piece=PT.LO_TOM, at=[], fill=[13, 14, 15], vel=(66, 80)),
        dict(bus="hats", piece=PT.CHH, at=[0, 2, 4, 6, 8, 10, 12, 14], accent={2: 70, 6: 70, 10: 70, 14: 70},
             vel=(46, 56)),
        dict(bus="hats", piece=PT.OHH, at=[], fill=[14], vel=60, dur=0.5)]
HALF = [dict(bus="kick", piece=PT.KICK, at=[0], fill=[0, 10], vel=96),
        dict(bus="toms", piece=PT.LO_FLOOR, at=[8, 15], vel=(62, 74)),
        dict(bus="snare", piece=PT.STICK, at=[12], vel=(80, 90)),
        dict(bus="hats", piece=PT.CHH, at=[0, 4, 8, 12], vel=(40, 52))]
LIGHT = [dict(bus="kick", piece=PT.KICK, at=[0], vel=88), dict(bus="hats", piece=PT.CHH, at=[0, 8], vel=(38, 48))]
GROOVE = {"A1": FULL, "B": FULL, "turn": HALF, "A2": FULL, "C": HALF, "A3": FULL, "outro": LIGHT}


def bar_s(b):
    return b * 4 * 60 / BPM


def place_bars(tl, bus, rec, rec_bar, song_bar, nbars, gain=1.0):
    for k in range(nbars):
        tl.put(bus, edge(rec.bar(rec_bar + k, tail_beats=0.2), 2, 25), (song_bar + k) * 4, gain)


def bass_line(chart, start_bar, seed):
    """A root under each chord, and an octave pickup walking into the second one."""
    rng = np.random.default_rng(seed)
    lo = PT.P["G1"]

    def reg(pc):
        return lo + (pc - lo) % 12
    out = []
    for b, bar in enumerate(chart):
        base = (start_bar + b) * 4
        for ci, (name, beats) in enumerate(bar):
            r = reg(PT.root(name))
            out.append(N.Note(base + ci * 2, 1.4, r, 90 if ci == 0 else 82))
            if ci == 0 and rng.random() < 0.7:
                out.append(N.Note(base + 1.5, 0.4, r + 12, 58))
    return out


def build(out_dir: Path, stems: bool = False) -> dict:
    tl = Timeline(BPM, BARS * 4, tail_s=7.0)
    n = tl.n

    # --- the record -----------------------------------------------------------------------------------------
    rec = dict(semis=KEY, dst_bar=2, curve=None)
    theme = Record(SRC, ["piano1", "strings low"], BPM, vel={"strings low": 0.6}, **rec)   # crystal piano, pad
    melody = Record(SRC, ["piano1"], BPM, **rec)
    comp = Record(SRC, ["Track 3", "Track 4", "Track 8"], BPM, **rec)    # horn and harpsichord stabs, strings
    vibes = Record(SRC, ["Track 12", "strings low"], BPM, vel={"strings low": 0.65}, **rec)
    pads = Record(SRC, ["strings low", "string"], BPM, **rec)
    counter = Record(SRC, ["Track 7", "Track 9"], BPM, **rec)            # guitar and celesta counter-line
    clar = Record(SRC, ["clarinet", "Track 11", "strings low"], BPM, **rec)

    place_bars(tl, "strings", pads, 14, 0, 4, 1.2)                       # intro: the pedal, filtered open
    place_bars(tl, "sample", theme, 2, 4, 8)                             # A1
    place_bars(tl, "sample", comp, 6, 8, 4, 0.7)
    place_bars(tl, "sample", vibes, 10, 12, 4)                           # B: the vibes take, then the theme on top
    place_bars(tl, "sample", vibes, 10, 16, 4, 0.85)
    place_bars(tl, "sample", melody, 2, 16, 4, 0.7)
    place_bars(tl, "strings", pads, 14, 20, 4)                           # turn: the pedal into the key change
    place_bars(tl, "sample", theme, 18, 24, 4)                           # A2 in A minor
    place_bars(tl, "sample", theme, 18, 28, 4)
    place_bars(tl, "sample", comp, 18, 28, 4, 0.7)
    place_bars(tl, "sample", theme, 22, 32, 4, 0.85)                     # C: the piano comps, the Rhodes sings
    place_bars(tl, "sample", theme, 26, 36, 4)                           # A3: everything
    place_bars(tl, "sample", comp, 26, 36, 4, 0.7)
    place_bars(tl, "counter", counter, 26, 36, 4)
    place_bars(tl, "sample", theme, 18, 40, 4, 0.9)
    place_bars(tl, "counter", counter, 26, 40, 4, 0.7)
    place_bars(tl, "strings", pads, 18, 40, 4, 0.6)
    place_bars(tl, "sample", clar, 30, 44, 4, 0.75)                      # outro: clarinet over the choir
    tl.put("sample", edge(clar.bar(33, tail_beats=4.0), 2, 400), 47 * 4, 0.6)

    # --- new parts ------------------------------------------------------------------------------------------
    rhodes = PT.comp(CHART[4:20] + CHART[24:44], start_bar=4, vel=52, seed=7, bpm=BPM)
    rhodes = [x for x in rhodes if x.start < 20 * 4] + \
        [N.Note(x.start + 4 * 4, x.dur, x.pitch, x.vel) for x in rhodes if x.start >= 20 * 4]
    rhodes += PT.comp(CHART[20:24], start_bar=20, vel=44, style="pad", bpm=BPM)
    rhodes += PT.comp(CHART[44:48], start_bar=44, vel=42, style="pad", bpm=BPM)
    rhodes = [N.Note(x.start, x.dur + (6 if x.start >= 47 * 4 + 2 else 0), x.pitch, x.vel) for x in rhodes]
    # C: the A-minor melody (source 36-43) an octave down on the Rhodes, over the source's own piano comping
    line = N.remap(N.top_line(N.window(SRC.tracks["piano1"], 36, 44)), 4, 2, None)
    lead = PT.melody_double(N.shift(line, 32 * 4), octave=-1, vel=50)
    tl.put_full("rhodes", fluid([(N.shift(rhodes, 0, KEY), 4, 0, 0)], BPM, tail=8))
    tl.put_full("lead", fluid([(N.shift(lead, 0, KEY), 4, 0, 1)], BPM, tail=4))

    bass = bass_line(CHART[4:20], 4, 8) + bass_line(CHART[24:44], 24, 9)
    bass += [N.Note(b * 4, 3.6, PT.P["E2"] if b < 24 else PT.P["C2"], 76) for b in range(20, 24)]   # pedal
    bass += [N.Note(44 * 4 + k * 4, 3.6 if k < 3 else 8, PT.P["C2"], 72) for k in range(4)]
    bass = N.humanize(bass, BPM, 6, 4, seed=10, push_ms=6)
    bass = N.shift(bass, 0, KEY)
    tl.put_full("bass", fluid([(bass, 33, 0, 2)], BPM, tail=4))
    tl.put_full("sub", PT.sub(bass, BPM, n))

    drums = {"kick": [], "snare": [], "hats": [], "toms": []}
    for name, b0, nb in FORM:
        feel = GROOVE.get(name)
        if not feel:
            continue
        if name == "outro":
            nb = 2
        for piece, ns in PT.groove(nb, b0, feel, seed=50 + b0, swing_amt=0.56, bpm=BPM).items():
            drums[piece] += ns
    tl.kicks = [x.start * 60 / BPM for x in drums["kick"]]
    for piece, ns in drums.items():
        tl.put_full(piece, fluid([(ns, KIT, 128, 9)], BPM, tail=2))

    # --- processing -----------------------------------------------------------------------------------------
    b = tl.bus

    def ride(t):        # opens through the intro, dips for the turn, closes through the outro
        f = np.full_like(t, 5200.0)
        i1, o0, o1 = bar_s(4), bar_s(44), bar_s(47.5)
        f = np.where(t < i1, 400 * (5200 / 400) ** np.clip(t / i1, 0, 1) ** 1.6, f)
        f = np.where((t > bar_s(20)) & (t < bar_s(24)), 2600.0, f)
        f = np.where(t > o0, 5200 * (350 / 5200) ** np.clip((t - o0) / (o1 - o0), 0, 1), f)
        return f

    s = fx.highpass(b["sample"], 140, 2)
    s = fx.bitcrush(s, 12, 26040, mix=0.85)
    s = fx.saturate(s, 1.5)
    s = fx.sweep_lowpass(s, ride)
    b["sample"] = fx.wow_flutter(s, seed=22) * 1.1

    c = fx.bitcrush(fx.highpass(b["counter"], 250, 2), 12, 26040, mix=0.6)
    b["counter"] = fx.pan(fx.wow_flutter(fx.lowpass(fx.saturate(c, 1.2), 5000, 2), seed=23), -0.25) * 0.7
    st = fx.highpass(b["strings"], 160, 2)
    st = fx.sweep_lowpass(fx.bitcrush(st, 12, 26040, mix=0.5), lambda t: np.minimum(ride(t), 3400.0))
    b["strings"] = fx.widen(fx.wow_flutter(st, seed=24), 1.4) * 0.5

    r = fx.detune_chorus(b["rhodes"], cents=7, mix=0.3)
    r = fx.saturate(fx.lowpass(fx.highpass(r, 190, 2), 5200, 2), 1.4)
    b["rhodes"] = fx.wow_flutter(r, wow_ms=1.2, seed=25) * 0.75
    b["lead"] = fx.wow_flutter(fx.saturate(fx.lowpass(b["lead"], 4000, 2), 1.3), seed=26) * 0.6

    bs = fx.saturate(fx.lowpass(b["bass"], 650, 2), 1.8)
    b["bass"] = fx.mono_below(fx.compress(bs, -20, 3, 15, 120), 300) * 0.9
    b["sub"] = fx.lowpass(b["sub"], 110, 2) * 0.8

    kk = fx.saturate(fx.lowpass(b["kick"], 4500, 2), 2.0)
    b["kick"] = fx.compress(kk, -14, 3, 3, 90) * 1.1
    sn = fx.bitcrush(fx.highpass(b["snare"], 170, 2), 12, 26040)
    b["snare"] = fx.lowpass(fx.saturate(sn, 1.6), 7500, 2) * 0.9
    tm = fx.saturate(fx.lowpass(fx.highpass(b["toms"], 60, 2), 5000, 2), 1.5)
    b["toms"] = fx.pan(tm, -0.2)
    hh = fx.saturate(fx.lowpass(fx.highpass(b["hats"], 400, 2), 5000, 2), 1.2)       # clean and dark: no fizz
    b["hats"] = fx.pan(hh, 0.25) * 0.45

    for k, target in LEVELS.items():
        b[k] = fx.level(b[k], target)

    # space: an open-air reverb, wide and soft; the counter-line and the lead get the tape echo
    send = b["sample"] * 0.4 + b["rhodes"] * 0.5 + b["counter"] * 0.6 + b["strings"] * 0.6 + b["snare"] * 0.2 + \
        b["toms"] * 0.25 + b["lead"] * 0.45
    verb = fx.reverb(send, seconds=2.6, damp_hz=3200, predelay_ms=24)
    echo = fx.tape_delay(b["counter"] + b["lead"], 0.75, BPM, feedback=0.32, repeats=3, lp=2800)

    duck = fx.duck(n, tl.kicks, depth_db=2.5, release_ms=200)[:, None]
    music = (b["sample"] + b["counter"] + b["strings"] + b["rhodes"] + b["lead"]) * duck
    low = (b["bass"] + b["sub"]) * fx.duck(n, tl.kicks, depth_db=4, release_ms=140)[:, None]
    drum_bus = fx.compress(b["kick"] + b["snare"] + b["toms"] + b["hats"], -16, 2.5, 12, 160, makeup_db=1)
    drum_bus = fx.tape(drum_bus, 1.3, 11000)

    # ambience: it's dry and windy, not raining - wind through the trees, a big gust carrying the key change,
    # and a breath of tape hiss; no vinyl
    wind = AMB.wind(n / SR, -37, seed=33, gusts=(bar_s(24) - 0.5,))
    hiss = AMB.hiss(n / SR, -63, seed=44)

    mix = music + low + drum_bus * 0.85 + verb * 0.6 + echo * 0.3 + wind + hiss
    mix = fx.wow_flutter(mix, wow_ms=0.4, flutter_ms=0.03, drift_ms=0.3, seed=29)
    end = int((bar_s(BARS) + 6.5) * SR)
    mix = fx.fade(mix[:end], fin_s=1.5, fout_s=4.0)
    out = M.master(mix, -14.0, -1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{FILE}.wav"
    M.write(path, out, tags={"title": TITLE, "artist": "Gielinor Lo-Fi", "album": "Gielinor Lo-Fi", "track": "2",
                             "comment": "Lo-fi reinterpretation of 'Autumn Voyage' (Ian Taylor, Jagex) from Old School RuneScape"})
    if stems:
        sd = out_dir / "stems"
        sd.mkdir(exist_ok=True)
        import soundfile as sf
        for k, x in {**b, "verb": verb, "echo": echo, "wind": wind}.items():
            sf.write(str(sd / f"{k}.wav"), x[:end], SR, subtype="PCM_16")
    rep = M.report(out, BPM, [(nm, b0, b0 + nb) for nm, b0, nb in FORM])
    rep["file"] = str(path)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    print(json.dumps(build(ROOT / "build" / "autumn_voyage", stems="--stems" in sys.argv), indent=1))
