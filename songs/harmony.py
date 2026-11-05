"""Gielinor Lo-Fi 01 - Harmony (after Ian Taylor's 'Harmony', Old School RuneScape).

Rainy Lumbridge at 78 bpm in G minor. The original 3/4 MIDI is pressed to a 'record' (3+3+2 remap, varispeeded
down two semitones) and chopped; Rhodes, bass and drums are new parts written against Harmony's own chords.
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
from engine.sampler import Record, Timeline, edge, reverse  # noqa: E402

TITLE = "Harmony (Rainy Lumbridge)"
BPM = 78.0
KEY = -2                      # everything lands in G minor
KIT = 32                      # GeneralUser GS jazz kit
SRC = N.load(ROOT / "source" / "harmony.mid")

A = [[("Am9", 4)], [("Cmaj9", 4)], [("Em11", 4)], [("Fmaj7", 3), ("G13", 1)],
     [("Am9", 4)], [("Em7", 4)], [("G69", 4)], [("Am11", 4)]]
B = [[("Amaj9", 4)]] * 2 + [[("Cmaj9", 4)]] * 2 + [[("Fmaj9", 4)]] * 2 + [[("G69", 4)]] * 2
TURN = [[("Am9", 4)], [("Am11", 4)], [("G69", 4)], [("Am9", 4)]]

# name, first song bar, bars
FORM = [("intro", 0, 4), ("A1", 4, 8), ("A2", 12, 8), ("B", 20, 8), ("turn", 28, 4),
        ("A3", 32, 8), ("A4", 40, 8), ("outro", 48, 4)]
BARS = 52
# stem balance as gated RMS in dBFS before the master (the sample leads; the hook must stay on top)
LEVELS = {"sample": -21, "oboe": -25, "strings": -31, "rhodes": -27, "lead": -28, "bass": -27, "sub": -32,
          "kick": -23, "snare": -27, "hats": -37}
CHART = A[:4] + A + A + B + TURN + A + A + A[:3] + [[("Am9", 4)]]


def bar_s(b):
    return b * 4 * 60 / BPM


def place_bars(tl, bus, rec, src_bar, song_bar, nbars, gain=1.0):
    """Lay record bars on the grid one chop per bar (each chop de-clicked, a short tail into the next)."""
    for k in range(nbars):
        tl.put(bus, edge(rec.bar(src_bar + k, tail_beats=0.2), 2, 25), (song_bar + k) * 4, gain)


def build(out_dir: Path, stems: bool = False) -> dict:
    tl = Timeline(BPM, BARS * 4, tail_s=7.0)
    n = tl.n

    # --- the record -----------------------------------------------------------------------------------------
    hook = Record(SRC, ["harpsichord", "kalimba"], BPM, semis=KEY)
    oboe = Record(SRC, ["oboe"], BPM, semis=KEY)
    strings = Record(SRC, ["slo strings", "strings"], BPM, semis=KEY)
    arp = Record(SRC, ["piccolo", "pizz", "slo strings"], BPM, semis=KEY, vel={"slo strings": 0.7})
    turn = Record(SRC, ["pizz", "slo strings"], BPM, semis=KEY)
    # the trumpet fanfare near its own speed: each 3/4 bar as two triplet beats (the waltz stretch made it drag)
    horn = Record(SRC, ["trumpet"], BPM, semis=KEY, dst_bar=2, curve=None)

    place_bars(tl, "sample", hook, 0, 0, 4, 0.9)                         # intro (filtered open below)
    tl.put("sample", edge(reverse(hook.beats(0, 2))), 3 * 4 + 2, 0.5)     # reversed pickup into A1
    place_bars(tl, "sample", hook, 0, 4, 8)                               # A1
    place_bars(tl, "sample", hook, 8, 12, 8, 0.85)                        # A2: the oboe phrase over the hook
    place_bars(tl, "oboe", oboe, 8, 12, 8)
    tl.put("sample", edge(reverse(arp.beats(40 * 4, 40 * 4 + 2))), 19 * 4 + 2, 0.45)
    place_bars(tl, "sample", arp, 40, 20, 8, 0.8)                         # B: the piccolo lift (A C F G)
    place_bars(tl, "sample", turn, 48, 28, 4, 0.9)                        # turnaround: pizz + strings
    tl.put("sample", edge(horn.beats(96, 104, 0.8), 2, 40), 28 * 4, 0.9)   # the fanfare twice, the last
    tl.put("sample", edge(horn.beats(96, 104, 2.0), 2, 60), 30 * 4, 0.9)   # held note rings over the stop
    place_bars(tl, "sample", hook, 52, 32, 8)                             # A3: everything
    place_bars(tl, "oboe", oboe, 52, 32, 8, 0.9)
    place_bars(tl, "strings", strings, 52, 32, 8)
    # A4: the hook flipped - straight, then a stutter on bar 4 and a reversed swell out of bar 7
    for k in (0, 1, 2, 3, 5, 6):
        tl.put("sample", edge(hook.bar(k, tail_beats=0.2), 2, 25), (40 + k) * 4)
    for off in (0.0, 1.5):
        tl.put("sample", edge(hook.beats(16, 17.5), 2, 20), 44 * 4 + off)
    tl.put("sample", edge(hook.beats(19, 20, 0.2), 2, 25), 44 * 4 + 3)
    tl.put("sample", edge(hook.beats(28, 30), 2, 20), 47 * 4)
    tl.put("sample", edge(reverse(hook.beats(28, 30)), 5, 5), 47 * 4 + 2, 0.7)
    place_bars(tl, "sample", hook, 0, 48, 4, 0.85)                        # outro (filtered closed below)

    # --- new parts ------------------------------------------------------------------------------------------
    rhodes = PT.comp(CHART[2:51], start_bar=2, vel=54, seed=4, bpm=BPM)
    rhodes = [x for x in rhodes if not (31 * 4 + 2 <= x.start < 32 * 4)]                    # space for the stop
    rhodes += PT.comp([[("Am9", 4)]], start_bar=51, vel=50, style="pad", bpm=BPM)
    rhodes = [N.Note(x.start, x.dur + (6 if x.start >= 51 * 4 else 0), x.pitch, x.vel) for x in rhodes]
    hook_line = N.top_line(N.window(N.remap(SRC.tracks["harpsichord"], 3), 0, 8))
    lead = PT.melody_double(N.shift(hook_line, 40 * 4), vel=50)                             # A4 Rhodes double
    tl.put_full("rhodes", fluid([(N.shift(rhodes, 0, KEY), 4, 0, 0)], BPM, tail=8))
    tl.put_full("lead", fluid([(N.shift(lead, 0, KEY), 4, 0, 1)], BPM, tail=4))

    bass = PT.bass(CHART[4:51], start_bar=4, seed=5)
    bass = [x for x in bass if not (31 * 4 + 2 <= x.start < 32 * 4) and x.start < 50 * 4]
    bass += [N.Note(51 * 4, 6, PT.P["A1"], 80)]
    bass = N.humanize(N.swing(bass, 0.58), BPM, 6, 4, seed=6, push_ms=6)
    bass = N.shift(bass, 0, KEY)
    tl.put_full("bass", fluid([(bass, 33, 0, 2)], BPM, tail=4))
    tl.put_full("sub", PT.sub(bass, BPM, n))

    drums = {"kick": [], "snare": [], "hats": []}
    for name, b0, nb in FORM:
        feel = {"A1": "full", "A2": "full", "B": "light", "turn": "half", "A3": "full", "A4": "full"}.get(name)
        if name == "outro":
            feel, nb = "light", 2
        if not feel:
            continue
        for piece, ns in PT.boom_bap(nb, b0, feel, seed=10 + b0, bpm=BPM).items():
            if name == "turn":
                ns = [x for x in ns if x.start < 31 * 4 + 2]
            drums[piece] += ns
    tl.kicks = [x.start * 60 / BPM for x in drums["kick"]]
    for piece, ns in drums.items():
        tl.put_full(piece, fluid([(ns, KIT, 128, 9)], BPM, tail=2))

    # --- processing -----------------------------------------------------------------------------------------
    b = tl.bus
    stop0, stop1 = tl.i(31 * 4 + 2), tl.i(32 * 4)
    for k in ("sample", "oboe", "strings", "rhodes", "bass", "sub"):
        b[k][stop0:stop1] = fx.tape_stop(b[k][stop0:stop1], (stop1 - stop0) / SR)

    def ride(t):        # sample filter: opens through the intro, closes through the outro, ducks in the turn
        f = np.full_like(t, 5200.0)
        i1, o0, o1 = bar_s(4), bar_s(48), bar_s(51.5)
        f = np.where(t < i1, 380 * (5200 / 380) ** np.clip(t / i1, 0, 1) ** 1.6, f)
        f = np.where(t > o0, 5200 * (300 / 5200) ** np.clip((t - o0) / (o1 - o0), 0, 1), f)
        f = np.where((t > bar_s(28)) & (t < bar_s(31.5)), 3200.0, f)
        return f

    s = fx.highpass(b["sample"], 140, 2)
    s = fx.bitcrush(s, 12, 26040, mix=0.85)
    s = fx.saturate(s, 1.6)
    s = fx.sweep_lowpass(s, ride)
    s = fx.wow_flutter(s, seed=2)
    b["sample"] = s * 1.1

    o = fx.bitcrush(fx.highpass(b["oboe"], 220, 2), 12, 26040, mix=0.6)
    b["oboe"] = fx.wow_flutter(fx.lowpass(fx.saturate(o, 1.3), 4800, 2), seed=3) * 0.75
    st = fx.lowpass(fx.highpass(b["strings"], 200, 2), 3200, 2)
    b["strings"] = fx.widen(fx.wow_flutter(st, seed=4), 1.4) * 0.5

    r = fx.detune_chorus(b["rhodes"], cents=7, mix=0.3)
    r = fx.saturate(fx.lowpass(fx.highpass(r, 190, 2), 5500, 2), 1.4)
    b["rhodes"] = fx.wow_flutter(r, wow_ms=1.2, seed=5) * 0.75
    b["lead"] = fx.wow_flutter(fx.saturate(fx.lowpass(b["lead"], 4200, 2), 1.3), seed=6) * 0.6

    bs = fx.saturate(fx.lowpass(b["bass"], 650, 2), 1.8)
    b["bass"] = fx.mono_below(fx.compress(bs, -20, 3, 15, 120), 300) * 0.9
    b["sub"] = fx.lowpass(b["sub"], 110, 2) * 0.8

    kk = fx.saturate(fx.lowpass(b["kick"], 4500, 2), 2.0)
    b["kick"] = fx.compress(kk, -14, 3, 3, 90) * 1.1
    sn = fx.bitcrush(fx.highpass(b["snare"], 170, 2), 12, 26040)
    b["snare"] = fx.lowpass(fx.saturate(sn, 1.7), 8000, 2) * 0.9
    hh = fx.bitcrush(fx.lowpass(fx.highpass(b["hats"], 600, 2), 7000, 2), 12, 26040)
    b["hats"] = fx.pan(hh, 0.25) * 0.45

    for k, target in LEVELS.items():
        b[k] = fx.level(b[k], target)

    # space: one room for everything, a dotted-eighth tape echo on the oboe and lead
    send = b["sample"] * 0.35 + b["rhodes"] * 0.5 + b["oboe"] * 0.5 + b["strings"] * 0.6 + b["snare"] * 0.25 + b["lead"] * 0.4
    verb = fx.reverb(send, seconds=2.4, damp_hz=3000, predelay_ms=22)
    echo = fx.tape_delay(b["oboe"] + b["lead"] * 0.8, 0.75, BPM, feedback=0.35, repeats=4, lp=2600)

    duck = fx.duck(n, tl.kicks, depth_db=2.5, release_ms=200)[:, None]
    music = (b["sample"] + b["oboe"] + b["strings"] + b["rhodes"] + b["lead"]) * duck
    low = (b["bass"] + b["sub"]) * fx.duck(n, tl.kicks, depth_db=4, release_ms=140)[:, None]
    drum_bus = fx.compress(b["kick"] + b["snare"] + b["hats"], -16, 2.5, 12, 160, makeup_db=1)
    drum_bus = fx.tape(drum_bus, 1.3, 11000)

    # ambience: rain in the window throughout, heavier where the beat is out
    secs = n / SR
    t = np.arange(n) / SR
    rain_amt = np.interp(t, [0, bar_s(3), bar_s(4.5), bar_s(28), bar_s(31.5), bar_s(32), bar_s(48), secs],
                         [1.0, 1.0, 0.45, 0.45, 0.8, 0.45, 0.45, 1.1])[:, None]
    rain = AMB.rain(secs, -30, seed=21) * rain_amt
    vinyl = AMB.vinyl(secs, -40, seed=22)
    hiss = AMB.hiss(secs, -56, seed=23)

    mix = music + low + drum_bus * 0.85 + verb * 0.55 + echo * 0.35 + rain + vinyl + hiss
    mix = fx.wow_flutter(mix, wow_ms=0.4, flutter_ms=0.03, drift_ms=0.3, seed=9)
    end = int((bar_s(BARS) + 6.5) * SR)
    mix = fx.fade(mix[:end], fin_s=1.5, fout_s=4.0)
    out = M.master(mix, -14.0, -1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "01 - Harmony (Rainy Lumbridge).wav"
    M.write(path, out, tags={"title": TITLE, "artist": "Gielinor Lo-Fi", "album": "Gielinor Lo-Fi", "track": "1",
                             "comment": "Lo-fi reinterpretation of 'Harmony' (Ian Taylor, Jagex) from Old School RuneScape"})
    if stems:
        sd = out_dir / "stems"
        sd.mkdir(exist_ok=True)
        import soundfile as sf
        for k, x in {**b, "verb": verb, "echo": echo, "rain": rain}.items():
            sf.write(str(sd / f"{k}.wav"), x[:end], SR, subtype="PCM_16")
    rep = M.report(out, BPM, [(nm, b0, b0 + nb) for nm, b0, nb in FORM])
    rep["file"] = str(path)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    print(json.dumps(build(ROOT / "build" / "harmony", stems="--stems" in sys.argv), indent=1))
