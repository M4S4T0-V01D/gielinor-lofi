"""Gielinor Lo-Fi 05 - Sea Shanty (Port Sarim) (after Ian Taylor's 'Sea Shanty', Old School RuneScape).

Evening on the Port Sarim docks in 12/8 at 76 bpm, D minor. The original (3/4, E minor, 130 bpm) is pressed to a
'record' (each 3/4 bar as two triplet beats, so the tune keeps its pace; varispeeded down two semitones) and
chopped: the accordion tune carries the A sections, the flute joins on A2 and A3, and the harmonica's A-minor dance
is the bridge. Surf and gulls open and close it.
Its own moves (no tape stop or reversed chops, those are Harmony's): the source's tubular bell rung as a ship's
bell into the dance, the turn going under the water and surfacing on a wave crash, and A4 swaying like a deck.
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

TITLE = "Sea Shanty (Port Sarim)"
FILE = "05 - Sea Shanty (Port Sarim)"
BPM = 76.0
KEY = -2                      # E minor -> D minor
KIT = 32                      # GeneralUser GS jazz kit
SRC = N.load(ROOT / "source" / "sea_shanty.mid")

# source bars (3/4): 0-7 the tune, 8-15 its second half, 16-31 both again with the flute on top, 32-47 a busier
# accordion variation (... F Em D D), 48-63 the A-minor dance (harmonica from 56), 64-79 the tune once more.
# Each 3/4 bar is pressed into two song beats as triplets (dst_bar=2), so the tune keeps its pace (12/8 at 76)
# and one song bar holds two source bars. The chords follow the source's bass, one per source bar.
ROOTS = ("EBEEEBEE" "CBCBCBCD") * 2 + "EDEDFEDD" "EDEDFEDA" + "AD" * 8 + "EBEEEBEE" "CBCBCBCE"


def chord(s):
    r = ROOTS[s]
    if 48 <= s < 64:
        return {"A": "Am9", "D": "Dm9"}[r]
    return {"E": "Em9" if s % 2 == 0 else "Em11", "B": "B7b13" if s % 4 == 1 else "B7", "C": "Cmaj9",
            "D": "D69", "F": "Fmaj7", "A": "Am9"}[r]


def chart(s0, nsrc):
    return [[(chord(s0 + i), 2), (chord(s0 + i + 1), 2)] for i in range(0, nsrc, 2)]


FORM = [("intro", 0, 4), ("A1", 4, 8), ("A2", 12, 8), ("B", 20, 8), ("turn", 28, 4),
        ("A3", 32, 8), ("A4", 40, 8), ("outro", 48, 4)]
BARS = 52
LEVELS = {"sample": -21, "flute": -25, "rhodes": -27, "lead": -29, "bass": -27, "sub": -32,
          "kick": -23, "snare": -27, "hats": -40}
CHART = (chart(8, 8) + chart(0, 16) + chart(16, 16) + chart(48, 16) + chart(36, 4) * 2 + chart(64, 16)
         + chart(32, 16) + chart(72, 8))

# the shanty's own groove, on a 12/8 triplet grid: a stomping kick, snare on 2 and 4 with a clap on 4, a shuffled
# tambourine and a woodblock knock (after the source's woodblock). The dance gets four on the floor.
TAMB = dict(bus="hats", piece=PT.TAMB, at=[0, 2, 3, 5, 6, 8, 9, 11], accent={0: 74, 3: 82, 6: 74, 9: 82}, vel=(36, 50))
WOOD = dict(bus="hats", piece=PT.LO_WOOD, at=[2, 8], odd=[2, 7, 8], vel=(52, 66), p=0.75)
FULL = [dict(bus="kick", piece=PT.KICK, at=[0, 6], odd=[0, 5, 6], fill=[0, 5, 6, 11], accent={0: 104}, vel=(84, 94)),
        dict(bus="snare", piece=PT.SNARE, at=[3, 9], vel=(92, 102)),
        dict(bus="snare", piece=PT.CLAP, at=[9], vel=(60, 72)),
        dict(bus="snare", piece=PT.SNARE, at=[8, 11], fill=[10, 11], vel=(24, 36), p=0.3), TAMB, WOOD]
DANCE = [dict(bus="kick", piece=PT.KICK, at=[0, 3, 6, 9], fill=[0, 3, 6, 9, 11], vel=(88, 98)),
         dict(bus="snare", piece=PT.CLAP, at=[3, 9], vel=(84, 96)),
         dict(TAMB, at=list(range(12))), dict(WOOD, at=[2, 5, 8, 11], odd=[2, 5, 8], p=0.85)]
LIGHT = [dict(bus="kick", piece=PT.KICK, at=[0], vel=90), dict(TAMB, at=[0, 6], accent={})]
GROOVE = {"A1": FULL, "A2": FULL, "B": DANCE, "turn": FULL, "A3": FULL, "A4": FULL, "outro": LIGHT}


def bar_s(b):
    return b * 4 * 60 / BPM


def place_bars(tl, bus, rec, src_bar, song_bar, nbars, gain=1.0):
    for k in range(nbars):
        tl.put(bus, edge(rec.bar(src_bar + k, tail_beats=0.2), 2, 25), (song_bar + k) * 4, gain)


def render(stomp: bool = False) -> dict:
    """The band and the harbour, before vinyl, hiss and the master. stomp=True is the first half of The Scallywag
    Stomp: no outro; bar 48 turns on the dominant (A7, shared by D minor and D major) and Sea Shanty 2 takes the
    band from bar 49."""
    tl = Timeline(BPM, BARS * 4, tail_s=8.0)
    n = tl.n

    # --- the record -----------------------------------------------------------------------------------------
    # record bar k = source bars 2k and 2k+1
    rec = dict(semis=KEY, dst_bar=2, curve=None)
    squeeze = Record(SRC, ["accordion", "pizz"], BPM, vel={"pizz": 0.8}, **rec)   # the tune
    flute = Record(SRC, ["flute"], BPM, **rec)
    dance = Record(SRC, ["harmonica", "accordion", "pizz", "tubular bells"], BPM,
                   vel={"accordion": 0.8, "tubular bells": 0.7}, **rec)
    ship_bell = Record(SRC, ["tubular bells"], BPM, **rec)             # its strike on source bar 47

    place_bars(tl, "sample", squeeze, 4, 0, 4, 0.9)                    # intro: the second half, from down the dock
    place_bars(tl, "sample", squeeze, 0, 4, 8)                         # A1: the tune, both halves
    place_bars(tl, "sample", squeeze, 8, 12, 8, 0.85)                  # A2: again, the flute on top
    place_bars(tl, "flute", flute, 8, 12, 8)
    strike = edge(ship_bell.beats(47 * 2, 47 * 2 + 2.5), 1, 200)
    tl.put("sample", strike, 19 * 4 + 2, 0.7)                          # ding-ding into the dance
    tl.put("sample", strike, 19 * 4 + 2.5, 0.55)
    place_bars(tl, "sample", dance, 24, 20, 8, 0.85)                   # B: the dance, harmonica in its second half
    for k, rb in enumerate((18, 19, 18, 19)):                          # turn: F Em D D twice, under in bar 31
        tl.put("sample", edge(squeeze.bar(rb, tail_beats=0.2), 2, 25), (28 + k) * 4, 0.9)
    place_bars(tl, "sample", squeeze, 32, 32, 8)                       # A3: the tune's last pass, flute on top
    place_bars(tl, "flute", flute, 8, 32, 8, 0.9)
    place_bars(tl, "sample", squeeze, 16, 40, 8)                       # A4: the variation, swaying (below)
    if stomp:                                                          # the turn: the tune's own B7 bars, which are
        tl.put("sample", edge(squeeze.beats(2, 4.2), 2, 30), 48 * 4, 0.9)          # also Sea Shanty 2's dominant
        tl.put("sample", edge(squeeze.beats(10, 12.3), 2, 60), 48 * 4 + 2, 0.9)
    else:
        place_bars(tl, "sample", squeeze, 36, 48, 4, 0.85)             # outro: the filter closes on the tune

    # --- new parts ------------------------------------------------------------------------------------------
    rhodes = PT.comp(CHART[2:51], start_bar=2, vel=54, seed=5, bpm=BPM)
    rhodes = [x for x in rhodes if not (20 * 4 <= x.start < 28 * 4)]
    rhodes += PT.comp(chart(48, 16), start_bar=20, vel=46, style="pad", bpm=BPM)   # B: room for the harmonica
    turn = [[("B7b13", 2), ("B7", 2)]]                                 # stomp: the dominant both keys share
    if stomp:
        rhodes = [x for x in rhodes if x.start < 48 * 4 - 0.1]
        rhodes += PT.comp(turn, start_bar=48, vel=56, seed=9, bpm=BPM)
    else:
        rhodes += PT.comp([[("Em9", 4)]], start_bar=51, vel=50, style="pad", bpm=BPM)
    rhodes = [N.Note(x.start, x.dur + (6 if x.start >= 51 * 4 else 0), x.pitch, x.vel) for x in rhodes]
    tune = N.top_line(N.window(N.remap(SRC.tracks["accordion"], 3, 2, None), 16, 24))
    lead = PT.melody_double(N.shift(tune, 40 * 4), vel=48)                         # A4 Rhodes double
    tl.put_full("rhodes", fluid([(N.shift(rhodes, 0, KEY), 4, 0, 0)], BPM, tail=8))
    tl.put_full("lead", fluid([(N.shift(lead, 0, KEY), 4, 0, 1)], BPM, tail=4))

    bass = PT.bass(CHART[4:51], start_bar=4, seed=6)
    bass = [x for x in bass if x.start < (48 * 4 - 0.1 if stomp else 50 * 4)]
    bass += PT.bass(turn, start_bar=48, seed=8) if stomp else [N.Note(51 * 4, 6, PT.P["E1"], 80)]
    bass = N.humanize(N.swing(bass, 0.58), BPM, 6, 4, seed=7, push_ms=6)
    bass = N.shift(bass, 0, KEY)
    tl.put_full("bass", fluid([(bass, 33, 0, 2)], BPM, tail=4))
    tl.put_full("sub", PT.sub(bass, BPM, n))

    drums = {"kick": [], "snare": [], "hats": []}
    for name, b0, nb in FORM:
        feel = GROOVE.get(name)
        if name == "outro":
            nb = 2
        if not feel or (stomp and name == "outro"):
            continue
        for piece, ns in PT.groove(nb, b0, feel, seed=50 + b0, steps=12, bpm=BPM).items():
            drums[piece] += ns
    if stomp:                                                          # a fill bar over the turn
        for piece, ns in PT.groove(1, 48, FULL, seed=98, fill_every=1, steps=12, bpm=BPM).items():
            drums[piece] += ns
    tl.kicks = [x.start * 60 / BPM for x in drums["kick"]]
    for piece, ns in drums.items():
        tl.put_full(piece, fluid([(ns, KIT, 128, 9)], BPM, tail=2))

    # --- processing -----------------------------------------------------------------------------------------
    b = tl.bus

    def ride(t):        # closed under the surf at first, opens into A1, a little darker for the dance
        f = np.full_like(t, 5000.0)
        i0, i1, o0, o1 = bar_s(0.5), bar_s(4), bar_s(48), bar_s(51.5)
        f = np.where(t < i1, 300 * (5000 / 300) ** np.clip((t - i0) / (i1 - i0), 0, 1) ** 1.6, f)
        f = np.where((t > bar_s(20)) & (t < bar_s(28)), 4400.0, f)
        f = np.where((t > bar_s(28)) & (t < bar_s(31.5)), 3000.0, f)
        if not stomp:
            f = np.where(t > o0, 5000 * (300 / 5000) ** np.clip((t - o0) / (o1 - o0), 0, 1), f)
        return f

    t = np.arange(n) / SR
    # A4's second half leans with the swell: ~25 cents of pitch roll per 4 s wave; the drums keep the deck steady
    lean = np.interp(t, [bar_s(43.5), bar_s(44.5), bar_s(47.5), bar_s(48)], [0, 19, 19, 0])
    band, verb, echo = mix_band(b, tl.kicks, n, BPM, ride, lean)
    secs = (n + 0.5) / SR         # so int(secs * SR) == n in the generators
    ambience = harbour(n, secs, t, stomp)
    # the turn goes under: bar 31 sinks to ~350 Hz like a head under water, and surfaces on A3 with the crash
    def under(tt):
        u = np.clip((tt - bar_s(31)) / bar_s(1), 0, 1) ** 1.5
        return np.where((tt >= bar_s(31)) & (tt < bar_s(32)), 14000 * (350 / 14000) ** u, 18000.0)
    band = fx.sweep_lowpass(band, under)
    return dict(band=band, **ambience, b=b, verb=verb, echo=echo, n=n, secs=secs)


def mix_band(b, kicks, n, bpm, ride, lean=None):
    """The band's chains, room and sidechain (shared with The Scallywag Stomp). Levels the buses in b in place;
    returns (band, verb, echo)."""
    s = fx.highpass(b["sample"], 140, 2)
    s = fx.bitcrush(s, 12, 26040, mix=0.85)
    s = fx.saturate(s, 1.6)
    s = fx.sweep_lowpass(s, ride)
    b["sample"] = fx.wow_flutter(s, seed=22) * 1.1

    fl = fx.bitcrush(fx.highpass(b["flute"], 280, 2), 12, 26040, mix=0.6)
    b["flute"] = fx.wow_flutter(fx.lowpass(fx.saturate(fl, 1.3), 5000, 2), seed=23) * 0.75

    r = fx.detune_chorus(b["rhodes"], cents=7, mix=0.3)
    r = fx.saturate(fx.lowpass(fx.highpass(r, 190, 2), 5500, 2), 1.4)
    b["rhodes"] = fx.wow_flutter(r, wow_ms=1.2, seed=25) * 0.75
    b["lead"] = fx.wow_flutter(fx.saturate(fx.lowpass(b["lead"], 4200, 2), 1.3), seed=26) * 0.6

    bs = fx.saturate(fx.lowpass(b["bass"], 650, 2), 1.8)
    b["bass"] = fx.mono_below(fx.compress(bs, -20, 3, 15, 120), 300) * 0.9
    b["sub"] = fx.lowpass(b["sub"], 110, 2) * 0.8

    kk = fx.saturate(fx.lowpass(b["kick"], 4500, 2), 2.0)
    b["kick"] = fx.compress(kk, -14, 3, 3, 90) * 1.1
    sn = fx.bitcrush(fx.highpass(b["snare"], 170, 2), 12, 26040)
    b["snare"] = fx.lowpass(fx.saturate(sn, 1.7), 8000, 2) * 0.9
    hh = fx.saturate(fx.lowpass(fx.highpass(b["hats"], 500, 2), 4500, 4), 1.2)       # a dull tambourine, not fizz
    b["hats"] = fx.pan(hh, -0.25) * 0.45

    for k, target in LEVELS.items():
        b[k] = fx.level(b[k], target)

    # space: a harbour tavern room; the flute and the Rhodes double get the tape echo
    send = b["sample"] * 0.35 + b["rhodes"] * 0.5 + b["flute"] * 0.5 + b["snare"] * 0.25 + b["lead"] * 0.4
    verb = fx.reverb(send, seconds=2.4, damp_hz=3200, predelay_ms=25)
    echo = fx.tape_delay(b["flute"] + b["lead"] * 0.8, 0.75, bpm, feedback=0.35, repeats=4, lp=2600)

    duck = fx.duck(n, kicks, depth_db=2.5, release_ms=200)[:, None]
    music = (b["sample"] + b["flute"] + b["rhodes"] + b["lead"]) * duck
    low = (b["bass"] + b["sub"]) * fx.duck(n, kicks, depth_db=4, release_ms=140)[:, None]
    if lean is not None:
        music, low = fx.sway(music, lean, 0.25), fx.sway(low, lean, 0.25)
    drum_bus = fx.compress(b["kick"] + b["snare"] + b["hats"], -16, 2.5, 12, 160, makeup_db=1)
    drum_bus = fx.tape(drum_bus, 1.3, 11000)
    return music + low + drum_bus * 0.85 + verb * 0.6 + echo * 0.35, verb, echo


def harbour(n, secs, t, stomp):
    """Surf and gulls up front, back behind the music, then back again for the outro."""
    sea_end = ([bar_s(48), secs], [0.5, 0.5]) if stomp else ([bar_s(48), bar_s(50), secs],
                                                                               [0.5, 1.0, 1.0])
    sea_amt = np.interp(t, [0, bar_s(3), bar_s(4.5), bar_s(31), bar_s(32), bar_s(33)] + sea_end[0],
                        [1.0, 1.0, 0.5, 0.5, 0.8, 0.5] + sea_end[1])[:, None]
    sea = AMB.waves(secs, -27, seed=51, crashes=(bar_s(32),)) * sea_amt
    gull_times = (0.6, bar_s(2) + 0.3) + ((bar_s(48) + 0.4,) if stomp else (bar_s(49) + 0.5, bar_s(51) + 1.0))
    gulls = AMB.gulls(secs, -31, seed=52, rate=1 / 30, times=gull_times) * np.minimum(1.0, sea_amt * 1.4)
    return dict(sea=sea, gulls=gulls)


def build(out_dir: Path, stems: bool = False) -> dict:
    r = render()
    b, verb, echo, sea, gulls, secs = r["b"], r["verb"], r["echo"], r["sea"], r["gulls"], r["secs"]
    vinyl = AMB.vinyl(secs, -47, seed=53, crackle_rate=3.0)        # dense crackle would read as rain
    hiss = AMB.hiss(secs, -56, seed=54)
    mix = r["band"] + sea + gulls + vinyl + hiss
    mix = fx.wow_flutter(mix, wow_ms=0.4, flutter_ms=0.03, drift_ms=0.3, seed=29)
    end = int((bar_s(BARS) + 7.0) * SR)
    mix = fx.fade(mix[:end], fin_s=1.0, fout_s=4.5)
    out = M.master(mix, -14.0, -1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{FILE}.wav"
    M.write(path, out, tags={"title": TITLE, "artist": "Gielinor Lo-Fi", "album": "Gielinor Lo-Fi", "track": "5",
                             "comment": "Lo-fi reinterpretation of 'Sea Shanty' (Ian Taylor, Jagex) from Old School RuneScape"})
    if stems:
        sd = out_dir / "stems"
        sd.mkdir(exist_ok=True)
        import soundfile as sf
        for k, x in {**b, "verb": verb, "echo": echo, "sea": sea, "gulls": gulls}.items():
            sf.write(str(sd / f"{k}.wav"), x[:end], SR, subtype="PCM_16")
    rep = M.report(out, BPM, [(nm, b0, b0 + nb) for nm, b0, nb in FORM])
    rep["file"] = str(path)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    print(json.dumps(build(ROOT / "build" / "sea_shanty", stems="--stems" in sys.argv), indent=1))
