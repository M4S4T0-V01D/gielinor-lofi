"""Gielinor Lo-Fi 04 - Book of Spells (after Ian Taylor's 'Book of Spells', Old School RuneScape).

Late night in the Wizards' Tower at 80 bpm in Bb dorian. The original (4/4, C dorian, 110 bpm) is pressed to a
'record', swung, varispeeded down two semitones and chopped: the guitar/piano theme carries the A sections, the
vibes-and-ocarina G section is the bridge, and the pizzicato G-minor riff is the turnaround. No tape stop or
reversed pickups (those are Harmony's): the turn rides the source's own timpani roll into A3, and A4 throws a
vibes beat into the tape echo.
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

TITLE = "Book of Spells (Wizards' Tower)"
FILE = "04 - Book of Spells (Wizards' Tower)"
BPM = 80.0
KEY = -2                      # C dorian -> Bb dorian
KIT = 32                      # GeneralUser GS jazz kit
SWING = 0.57                  # the record's straight 16ths pulled toward the drums' 0.59
SRC = N.load(ROOT / "source" / "book_of_spells.mid")

# source tracks (GM programs): 1 pad, 2 slow strings, 3 nylon guitar, 4 horn, 5 pizz, 6 ocarina, 7 choir,
# 8 timpani, 9 celesta, 10 drums, 11 vibes, 12 piano
T = {k: f"Track {k}" for k in range(1, 13)}

# the theme's 8 bars (source 33-40 / 43-50 / 69-76): the cadence is C major, or G major before the G section
A_C = [[("Cm9", 4)], [("F13", 4)], [("Cm9", 4)], [("F9", 4)], [("Cm9", 4)], [("Ebmaj9", 4)], [("Dm7", 4)],
       [("Cadd9", 4)]]
A_G = A_C[:7] + [[("G5", 4)]]
INTRO = [[("F9", 4)], [("Cm9", 4)], [("Ebmaj9", 4)], [("Dm7", 4)]]
B = ([[("G5", 4)]] * 3 + [[("Ebmaj7", 4)]]) * 2
TURN = [[("Gm9", 4)]] * 4
OUTRO = [[("Cm9", 4)], [("Ebmaj9", 4)], [("Dm7", 4)], [("Cadd9", 4)]]

FORM = [("intro", 0, 4), ("A1", 4, 8), ("A2", 12, 8), ("B", 20, 8), ("turn", 28, 4),
        ("A3", 32, 8), ("A4", 40, 8), ("outro", 48, 4)]
BARS = 52
LEVELS = {"sample": -21, "bell": -27, "strings": -30, "rhodes": -27, "lead": -29, "bass": -28, "sub": -33,
          "kick": -23, "snare": -27, "hats": -37}
CHART = INTRO + A_C + A_G + B + TURN + A_G + A_C + OUTRO

# the tower's own groove: a jazz ride (spang-a-lang) over a sidestick and a lazy two-step kick - no 16th hats,
# which read as rain under the record
RIDE = dict(bus="hats", piece=PT.RIDE, at=[0, 4, 6, 8, 12, 14], accent={0: 84, 4: 92, 8: 84, 12: 92}, vel=(52, 64),
            dur=0.5)
FULL = [dict(bus="kick", piece=PT.KICK, at=[0, 9], odd=[0, 6, 9], fill=[0, 9, 14], accent={0: 102}, vel=(80, 92)),
        dict(bus="snare", piece=PT.STICK, at=[4, 12], vel=(90, 100)),
        dict(bus="snare", piece=PT.SNARE, at=[7, 15], fill=[13, 15], vel=(24, 38), p=0.3),
        dict(bus="hats", piece=PT.PHH, at=[4, 12], vel=(46, 58)), RIDE]
HALF = [dict(bus="kick", piece=PT.KICK, at=[0], fill=[0, 14], vel=96),
        dict(bus="snare", piece=PT.STICK, at=[8], vel=(84, 94)),
        dict(bus="hats", piece=PT.RIDE_BELL, at=[0, 8], vel=(50, 62), dur=0.8)]
LIGHT = [dict(bus="kick", piece=PT.KICK, at=[0], vel=90), dict(RIDE, at=[0, 8], dur=1.0)]
GROOVE = {"A1": FULL, "A2": FULL, "B": HALF, "turn": FULL, "A3": FULL, "A4": FULL, "outro": LIGHT}


def bar_s(b):
    return b * 4 * 60 / BPM


def place_bars(tl, bus, rec, src_bar, song_bar, nbars, gain=1.0):
    for k in range(nbars):
        tl.put(bus, edge(rec.bar(src_bar + k, tail_beats=0.2), 2, 25), (song_bar + k) * 4, gain)


def build(out_dir: Path, stems: bool = False) -> dict:
    tl = Timeline(BPM, BARS * 4, tail_s=7.0)
    n = tl.n

    # --- the record -----------------------------------------------------------------------------------------
    rec = dict(semis=KEY, swing=SWING)
    theme = Record(SRC, [T[3], T[12]], BPM, **rec)                       # guitar + piano melody
    pads = Record(SRC, [T[1], T[2]], BPM, **rec)                         # new-age pad + slow strings
    bell = Record(SRC, [T[9]], BPM, **rec)                               # celesta counter-line
    spell = Record(SRC, [T[4], T[5], T[6], T[11]], BPM, **rec)           # horn, pizz, ocarina arps, vibes
    riff = Record(SRC, [T[5], T[6], T[7]], BPM, **rec)                   # pizz riff, ocarina stabs, choir
    vibes = Record(SRC, [T[11], T[3]], BPM, **rec)                       # the theme again on vibes
    timp = Record(SRC, [T[8]], BPM, **rec)                               # the source's one timpani roll (bar 21)

    place_bars(tl, "strings", pads, 6, 0, 4, 1.2)                        # intro: strings alone, filtered open
    place_bars(tl, "sample", theme, 33, 4, 8)                            # A1
    place_bars(tl, "strings", pads, 33, 4, 8, 0.8)
    place_bars(tl, "sample", theme, 43, 12, 8, 0.9)                      # A2: + celesta, cadence on G
    place_bars(tl, "bell", bell, 43, 12, 8)
    place_bars(tl, "strings", pads, 43, 12, 8, 0.8)
    place_bars(tl, "sample", spell, 52, 20, 8, 0.72)                     # B: the G section
    place_bars(tl, "sample", riff, 90, 28, 4, 0.9)                       # turn: the pizz riff, played through
    tl.put("sample", edge(timp.beats(21 * 4 + 0.5, 21 * 4 + 2.3, tail_beats=0.5), 3, 40), 31 * 4 + 2.2, 2.0)
    place_bars(tl, "sample", theme, 43, 32, 8)                           # A3: everything
    place_bars(tl, "bell", bell, 43, 32, 8, 0.9)
    place_bars(tl, "strings", pads, 43, 32, 8)
    # A4: the vibes take; bar 5 keeps only its first beat and throws it into the tape echo (added after levels)
    for k in (0, 1, 2, 3, 5, 6, 7):
        tl.put("sample", edge(vibes.bar(69 + k, tail_beats=0.2), 2, 25), (40 + k) * 4)
    tl.put("sample", edge(vibes.beats(73 * 4, 73 * 4 + 1, tail_beats=0.15), 2, 60), 44 * 4)
    place_bars(tl, "strings", pads, 69, 40, 8, 0.9)
    place_bars(tl, "sample", theme, 37, 48, 4, 0.85)                     # outro: filtered closed, ends major
    place_bars(tl, "strings", pads, 37, 48, 4, 0.8)

    # --- new parts ------------------------------------------------------------------------------------------
    rhodes = PT.comp(CHART[2:51], start_bar=2, vel=54, seed=7, bpm=BPM)
    rhodes = [x for x in rhodes if not (20 * 4 <= x.start < 28 * 4)]
    rhodes += PT.comp(B, start_bar=20, vel=46, style="pad", bpm=BPM)     # B: let the arps sparkle
    rhodes += PT.comp([[("Cadd9", 4)]], start_bar=51, vel=50, style="pad", bpm=BPM)
    rhodes = [N.Note(x.start, x.dur + (6 if x.start >= 51 * 4 else 0), x.pitch, x.vel) for x in rhodes]
    vibe_line = N.top_line(N.window(SRC.tracks[T[11]], 69, 77))
    lead = PT.melody_double(N.shift(vibe_line, 40 * 4), octave=-1, vel=48)
    tl.put_full("rhodes", fluid([(N.shift(rhodes, 0, KEY), 4, 0, 0)], BPM, tail=8))
    tl.put_full("lead", fluid([(N.shift(lead, 0, KEY), 4, 0, 1)], BPM, tail=4))

    bass = PT.bass(CHART[4:28], start_bar=4, seed=8) + PT.bass(CHART[32:51], start_bar=32, seed=9)
    bass = [x for x in bass if x.start < 50 * 4]
    # the turn: the bass takes the original pizzicato riff (its lower octave)
    bass += [N.Note(x.start + 28 * 4, x.dur * 1.6, x.pitch, 92)
             for x in N.window(SRC.tracks[T[5]], 90, 94) if x.pitch <= 40 and x.start < 3 * 4 + 2]
    bass += [N.Note(51 * 4, 6, PT.P["C2"], 80)]
    bass = N.humanize(N.swing(bass, 0.58), BPM, 6, 4, seed=10, push_ms=6)
    bass = N.shift(bass, 0, KEY)
    tl.put_full("bass", fluid([(bass, 33, 0, 2)], BPM, tail=4))
    tl.put_full("sub", PT.sub(bass, BPM, n))

    drums = {"kick": [], "snare": [], "hats": []}
    for name, b0, nb in FORM:
        feel = GROOVE.get(name)
        if name == "outro":
            nb = 2
        if not feel:
            continue
        for piece, ns in PT.groove(nb, b0, feel, seed=30 + b0, swing_amt=0.62, bpm=BPM).items():
            drums[piece] += ns
    tl.kicks = [x.start * 60 / BPM for x in drums["kick"]]
    for piece, ns in drums.items():
        tl.put_full(piece, fluid([(ns, KIT, 128, 9)], BPM, tail=2))

    # --- processing -----------------------------------------------------------------------------------------
    b = tl.bus

    def ride(t):        # opens through the intro, a touch brighter for the spell, closes through the outro
        f = np.full_like(t, 5000.0)
        i1, o0, o1 = bar_s(4), bar_s(48), bar_s(51.5)
        f = np.where(t < i1, 380 * (5000 / 380) ** np.clip(t / i1, 0, 1) ** 1.6, f)
        f = np.where((t > bar_s(20)) & (t < bar_s(28)), 5600.0, f)
        f = np.where((t > bar_s(28)) & (t < bar_s(31.5)), 3000.0, f)
        f = np.where(t > o0, 5000 * (300 / 5000) ** np.clip((t - o0) / (o1 - o0), 0, 1), f)
        return f

    s = fx.highpass(b["sample"], 140, 2)
    s = fx.bitcrush(s, 12, 26040, mix=0.85)
    s = fx.saturate(s, 1.6)
    s = fx.sweep_lowpass(s, ride)
    b["sample"] = fx.wow_flutter(s, seed=12) * 1.1

    bl = fx.bitcrush(fx.highpass(b["bell"], 300, 2), 12, 26040, mix=0.7)
    b["bell"] = fx.wow_flutter(fx.lowpass(fx.saturate(bl, 1.2), 5500, 2), seed=13) * 0.7
    st = fx.highpass(b["strings"], 180, 2)
    st = fx.sweep_lowpass(fx.bitcrush(st, 12, 26040, mix=0.5), lambda t: np.minimum(ride(t), 3200.0))
    b["strings"] = fx.widen(fx.wow_flutter(st, seed=14), 1.4) * 0.5

    r = fx.detune_chorus(b["rhodes"], cents=7, mix=0.3)
    r = fx.saturate(fx.lowpass(fx.highpass(r, 190, 2), 5500, 2), 1.4)
    b["rhodes"] = fx.wow_flutter(r, wow_ms=1.2, seed=15) * 0.75
    b["lead"] = fx.wow_flutter(fx.saturate(fx.lowpass(b["lead"], 4200, 2), 1.3), seed=16) * 0.6

    bs = fx.saturate(fx.lowpass(b["bass"], 650, 2), 1.8)
    b["bass"] = fx.mono_below(fx.compress(bs, -20, 3, 15, 120), 300) * 0.9
    b["sub"] = fx.lowpass(b["sub"], 110, 2) * 0.8

    kk = fx.saturate(fx.lowpass(b["kick"], 4500, 2), 2.0)
    b["kick"] = fx.compress(kk, -14, 3, 3, 90) * 1.1
    sn = fx.bitcrush(fx.highpass(b["snare"], 170, 2), 12, 26040)
    b["snare"] = fx.lowpass(fx.saturate(sn, 1.7), 8000, 2) * 0.9
    hh = fx.saturate(fx.lowpass(fx.highpass(b["hats"], 400, 2), 6000, 2), 1.2)       # ride left clean: no crush fizz
    b["hats"] = fx.pan(hh, 0.3) * 0.45

    for k, target in LEVELS.items():
        b[k] = fx.level(b[k], target)

    # space: a bigger, stonier room than the bank; the celesta and the vibes double get the tape echo
    send = b["sample"] * 0.4 + b["rhodes"] * 0.5 + b["bell"] * 0.6 + b["strings"] * 0.6 + b["snare"] * 0.25 + b["lead"] * 0.4
    verb = fx.reverb(send, seconds=3.0, damp_hz=2800, predelay_ms=30)
    echo = fx.tape_delay(b["bell"] + b["lead"] * 0.8, 0.75, BPM, feedback=0.38, repeats=4, lp=2600)
    t0, t1 = tl.i(44 * 4), tl.i(44 * 4 + 1.15)
    thrown = np.zeros_like(b["sample"])
    thrown[t0:t1] = b["sample"][t0:t1]
    echo += fx.tape_delay(thrown, 0.75, BPM, feedback=0.55, repeats=5, lp=2400) * 2.2

    duck = fx.duck(n, tl.kicks, depth_db=2.5, release_ms=200)[:, None]
    music = (b["sample"] + b["bell"] + b["strings"] + b["rhodes"] + b["lead"]) * duck
    low = (b["bass"] + b["sub"]) * fx.duck(n, tl.kicks, depth_db=4, release_ms=140)[:, None]
    drum_bus = fx.compress(b["kick"] + b["snare"] + b["hats"], -16, 2.5, 12, 160, makeup_db=1)
    drum_bus = fx.tape(drum_bus, 1.3, 11000)

    # ambience: no rain in the tower, so no vinyl either (even sparse crackle over its surface noise reads as rain);
    # only a breath of tape hiss
    hiss = AMB.hiss(n / SR, -63, seed=43)

    mix = music + low + drum_bus * 0.85 + verb * 0.6 + echo * 0.35 + hiss
    mix = fx.wow_flutter(mix, wow_ms=0.4, flutter_ms=0.03, drift_ms=0.3, seed=19)
    end = int((bar_s(BARS) + 6.5) * SR)
    mix = fx.fade(mix[:end], fin_s=1.5, fout_s=4.0)
    out = M.master(mix, -14.0, -1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{FILE}.wav"
    M.write(path, out, tags={"title": TITLE, "artist": "Gielinor Lo-Fi", "album": "Gielinor Lo-Fi", "track": "4",
                             "comment": "Lo-fi reinterpretation of 'Book of Spells' (Ian Taylor, Jagex) from Old School RuneScape"})
    if stems:
        sd = out_dir / "stems"
        sd.mkdir(exist_ok=True)
        import soundfile as sf
        for k, x in {**b, "verb": verb, "echo": echo}.items():
            sf.write(str(sd / f"{k}.wav"), x[:end], SR, subtype="PCM_16")
    rep = M.report(out, BPM, [(nm, b0, b0 + nb) for nm, b0, nb in FORM])
    rep["file"] = str(path)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    print(json.dumps(build(ROOT / "build" / "book_of_spells", stems="--stems" in sys.argv), indent=1))
