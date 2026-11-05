"""Gielinor Lo-Fi - Flute Salad (Italo) (after Ian Taylor's 'Flute Salad', Old School RuneScape).

Italo piano house: 120 bpm in F# minor. The source is a 100 bpm G minor tune over a two-bar vamp (Gm | D/A, and
Em | B/F# in the second half of each phrase), with a funky syncopated bass, a harpsichord comp and the GM latin kit
(cowbell on 1 and 3, bongos, congas, agogo, cabasa) - most of an Italo record already. Its real downbeat is the
MIDI's beat 2, so every source bar is read one beat late. The record keeps the harpsichord, trombone and strings;
the flute tune is played as the classic Italo piano hook - the melody on top of block chords with a low octave, on
an electric grand - and the flute stays on it in A1 and the back half of A2. The source bass moves to a synth bass
with a sub, the source percussion plays on the 808 kit, and a new Standard 3 kit plays straight 16th hats with a
gated-reverb clap. The move is the salad spinner: in the ocarina break the pads and piano play through a rotary
speaker that turns slowly, then spins up to full speed over the four bars into the drop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

from engine import ambience as AMB, fx, master as M, notes as N, parts as PT  # noqa: E402
from engine.render import SR, fluid  # noqa: E402
from engine.sampler import Record, Timeline, edge  # noqa: E402

TITLE = "Flute Salad (Italo)"
FILE = "Flute Salad (Italo)"
BPM = 120.0
KEY = -1                      # G minor -> F# minor
KIT = 2                       # GeneralUser GS Standard 3 kit
PERC_KIT = 25                 # 808/909 kit, for the source's latin percussion
SWING = 0.52
SRC = N.load(ROOT / "source" / "flute_salad.mid")
PICK = 1                      # source bar s starts on source beat 4 s + PICK

# name, song bar, source bar, bars
SECTS = [("intro", 0, 0, 8), ("A1", 8, 0, 16), ("B", 24, 16, 12), ("break", 36, 28, 8), ("A2", 44, 36, 12),
         ("outro", 56, 40, 8), ("end", 64, 48, 1)]
BARS = 65
BREAK = (36, 44)
SPIN = (40, 44)               # the rotor spins up
LEVELS = {"sample": -26, "flute": -26, "piano": -23, "ep": -28, "pad": -30, "ocarina": -25, "bass": -24,
          "sub": -30, "kick": -20, "snare": -25, "hats": -31, "perc": -30}

v = PT.v
TONES = {"Gm": {7, 10, 2}, "D": {2, 6, 9}, "Em": {4, 7, 11}, "B": {11, 3, 6}}
PADS = {"Gm": v("G3", "A#3", "D4", "F4"), "D": v("F#3", "A3", "D4", "E4"), "Em": v("G3", "B3", "E4", "F#4"),
        "B": v("F#3", "B3", "D#4", "F#4"), "Em9": v("E3", "G3", "B3", "F#4"), "Cmaj9": v("E3", "G3", "B3", "D4"),
        "D69": v("F#3", "A3", "B3", "E4")}
LOW = {"Gm": "G2", "D": "A2", "Em": "E2", "B": "F#2", "Em9": "E2", "Cmaj9": "C2", "D69": "D2"}


def chord(s):
    """The source's harmony by source bar: Gm | D/A, or Em | B/F# in the second half of each phrase; the ocarina
    break gets its own pads."""
    if 28 <= s < 36:
        return ["Em9", "Cmaj9", "Em9", "D69"][(s - 28) % 4]
    if s >= 48:
        return "Gm"
    minor = 4 <= s % 8 < 8 if s < 16 else (24 <= s < 28 or 40 <= s < 44)
    return ("Em", "B") [s % 2] if minor else ("Gm", "D")[s % 2]


def song_of(sect):
    return next(x for x in SECTS if x[0] == sect)


def src_bar(b):
    """The source bar under song bar b."""
    for _, sb, s0, nb in SECTS:
        if sb <= b < sb + nb:
            return s0 + b - sb
    return 48


def lay(tl, bus, rec, b0, s0, nb, gain=1.0, tail=0.5, fin=3, fout=30):
    """Lay nb source bars from source bar s0 at song bar b0."""
    a = s0 * 4 + PICK
    tl.put(bus, edge(rec.beats(a, a + nb * 4, tail_beats=tail), fin, fout), b0 * 4, gain)


def src_notes(track, b0, s0, nb, keep=None):
    a = s0 * 4 + PICK
    return N.shift([x for x in SRC.tracks[track] if a <= x.start < a + nb * 4 and (keep is None or x.pitch in keep)],
                   b0 * 4 - a, 0)


def hook(b0, s0, nb, vel=0.85):
    """The Italo piano hook: each flute note on top, the next two chord tones below it and the melody an octave
    down, all struck together."""
    out = []
    for x in src_notes("piano1", b0, s0, nb):
        if x.dur < 0.08:
            continue
        tones = TONES[chord(src_bar(int(x.start // 4)))]
        below = [p for p in range(x.pitch - 1, x.pitch - 13, -1) if p % 12 in tones][:2]
        d = max(0.2, x.dur * 1.4)
        out += [N.Note(x.start, d, p, int(x.vel * vel * (1.0 if p == x.pitch else 0.82)))
                for p in [x.pitch, x.pitch - 12] + below]
    return out


def rotary(x, rate):
    """A rotary speaker: the horn (above 800 Hz) and the drum spin at rate(t) Hz, the drum lagging behind with more
    inertia; each rotor bends the pitch a little (Doppler) and sweeps its level across two mics."""
    n = len(x)
    t = np.arange(n) / SR
    r = rate(t)
    m = x.mean(axis=1)
    hi = fx.highpass(m[:, None], 800, 2)[:, 0]
    lo = m - hi
    out = np.zeros((n, 2), dtype=np.float32)
    idx = np.arange(n, dtype=np.float64)
    for band, tau, depth_ms, am in ((hi, 0.35, 0.6, 0.75), (lo, 1.4, 0.8, 0.4)):
        a = np.exp(-1 / (tau * SR))
        rr, _ = lfilter([1 - a], [1, -a], r, zi=[r[0] * a])
        ph = np.cumsum(2 * np.pi * rr / SR)
        y = np.interp(idx - (1 + np.sin(ph)) * depth_ms * SR / 1000, idx, band)
        out[:, 0] += y * (1 + am * np.sin(ph))
        out[:, 1] += y * (1 + am * np.sin(ph + np.pi / 2))
    return out * 0.75


def bar_s(b):
    return b * 4 * 60 / BPM


# Flute Salad's groove: four on the floor (the 4 drops out of fill bars for a clap pickup), a clap on 2 and 4 over a
# snare on the 4 with a skip clap on the a of 2 in odd bars, straight 16th closed hats accented on the off-beats
# and an open hat only on the and of 4. B adds a sidestick; the break keeps only hats until the clap returns.
HATS = [dict(bus="hats", piece=PT.CHH, at=[s for s in range(16) if s != 14], vel=(30, 42),
             accent={2: 72, 6: 72, 10: 72})]
OPEN = [dict(bus="hats", piece=PT.OHH, at=[14], vel=(64, 74), dur=0.45)]
ITALO = HATS + OPEN + [
    dict(bus="kick", piece=PT.KICK, at=[0, 4, 8, 12], fill=[0, 4, 8], vel=(104, 112), accent={0: 118}),
    dict(bus="snare", piece=PT.CLAP, at=[4, 12], odd=[4, 7, 12], vel=(88, 98), accent={7: 56}),
    dict(bus="snare", piece=PT.CLAP, at=[], fill=[4, 12, 13, 14, 15], vel=(70, 92)),
    dict(bus="snare", piece=PT.SNARE, at=[12], vel=(66, 76)),
]
INTRO = HATS
WARM = HATS + OPEN + [dict(bus="kick", piece=PT.KICK, at=[0, 4, 8, 12], vel=(88, 96))]
LIFT = ITALO + [dict(bus="snare", piece=PT.STICK, at=[6, 15], vel=(44, 56))]
AIR = [dict(bus="hats", piece=PT.CHH, at=[2, 6, 10, 14], vel=(34, 44))]
CALL = HATS + [dict(bus="snare", piece=PT.CLAP, at=[4, 12], vel=(80, 92))]
# (song bar, bars, spec)
DRUMS = [(0, 4, INTRO), (4, 4, WARM), (8, 16, ITALO), (24, 12, LIFT), (36, 4, AIR), (40, 4, CALL), (44, 12, ITALO),
         (56, 8, ITALO)]


def build(out_dir: Path, stems: bool = False) -> dict:
    tl = Timeline(BPM, BARS * 4, tail_s=7.0)
    n = tl.n

    # --- the records ----------------------------------------------------------------------------------------
    rec = dict(bpm=BPM, semis=KEY, curve=None, swing=SWING)
    band = Record(SRC, ["strings low", "tromb", "strings"], vel={"strings": 0.85}, **rec)
    flute = Record(SRC, ["piano1"], programs={"piano1": 73}, **rec)
    ep = Record(SRC, ["Track 6"], programs={"Track 6": 5}, **rec)                   # DX-style electric piano
    oca = Record(SRC, ["ocarina", "Track 7"], **rec)

    for name, b0, s0, nb in SECTS:
        if name == "break":
            continue
        lay(tl, "sample", band, b0, s0, nb, tail=4 if name == "end" else 0.5)
    lay(tl, "flute", flute, 8, 0, 16, tail=1)
    lay(tl, "flute", flute, 48, 40, 8, tail=2, fin=40)
    lay(tl, "ep", ep, 24, 16, 12, tail=2)
    lay(tl, "ocarina", oca, 36, 28, 8, tail=3)

    # --- new parts ------------------------------------------------------------------------------------------
    piano, pad, bass = [], [], []
    piano += hook(8, 0, 16) + hook(44, 36, 12) + hook(56, 40, 8, vel=0.7)
    for b in range(0, BARS):
        c = chord(src_bar(b))
        if 24 <= b < 36:                                 # B: the chord held on 1, struck again on the and of 2
            piano += [N.Note(b * 4 + at, d, p, vel) for at, d, vel in ((0, 1.4, 80), (1.5, 2.3, 70))
                      for p in PADS[c] + [PT.P[LOW[c]] + 12]]
            pad += [N.Note(b * 4, 4.0, p, 64) for p in PADS[c]]
        elif BREAK[0] <= b < BREAK[1]:                   # the break: a chord a bar for the spinner
            piano += [N.Note(b * 4, 3.95, p, 60) for p in PADS[c] + [PT.P[LOW[c]]]]
            pad += [N.Note(b * 4, 4.0, p, 70) for p in PADS[c] + [PADS[c][-1] + 12]]
        elif b >= 56:
            pad += [N.Note(b * 4, 4.0, p, 52) for p in PADS[c]]
    end = 64 * 4
    piano += [N.Note(end, 10, p, 76) for p in PADS["Gm"] + v("G2", "G4", "D5")]
    pad += [N.Note(end, 10, p, 60) for p in PADS["Gm"]]
    for name, b0, s0, nb in SECTS:                       # the source's own bass, from the intro's fifth bar
        if name in ("break", "end"):
            continue
        ns = src_notes("bass", b0, s0, nb)
        bass += [x for x in ns if x.start >= 4 * 4]
    bass += [N.Note(end, 4, PT.P["G1"], 100)]
    piano = N.shift(N.humanize(piano, BPM, 4, 5, seed=6), 0, KEY)
    pad = N.shift(N.humanize(pad, BPM, 8, 4, seed=7), 0, KEY)
    bass = N.shift(N.humanize(N.swing(bass, SWING), BPM, 3, 4, seed=9), 0, KEY)
    tl.put_full("piano", fluid([(piano, 2, 0, 4)], BPM, tail=8))                     # electric grand
    tl.put_full("pad", fluid([(pad, 50, 0, 5)], BPM, tail=8))                        # synth strings
    tl.put_full("bass", fluid([(bass, 39, 0, 2)], BPM, tail=4))                      # synth bass 2
    tl.put_full("sub", PT.sub(bass, BPM, n))

    drums = {"kick": [], "snare": [], "hats": []}
    for b0, nb, spec in DRUMS:
        for piece, ns in PT.groove(nb, b0, spec, seed=60 + b0, swing_amt=SWING, bpm=BPM).items():
            drums[piece] += ns
    drums["hats"] += [N.Note(b * 4, 2.0, 49, 92) for b in (8, 24, 44)] + [N.Note(end, 3.0, 49, 86)]
    drums["kick"] += [N.Note(end, 0.5, PT.KICK, 112)]
    tl.kicks = [x.start * 60 / BPM for x in drums["kick"]]
    for piece, ns in drums.items():
        tl.put_full(piece, fluid([(ns, KIT, 128, 9)], BPM, tail=2.5))
    perc = []
    for name, b0, s0, nb in SECTS:                       # the source's latin kit: all but the intro's first bars
        if name != "end":
            perc += [x for x in src_notes("Track 10", b0, s0, nb) if x.start >= 2 * 4]
    perc = N.humanize(N.swing(perc, SWING), BPM, 4, 6, seed=11)
    tl.put_full("perc", fluid([(perc, PERC_KIT, 128, 9)], BPM, tail=1.5))

    # --- processing -----------------------------------------------------------------------------------------
    b = tl.bus
    t = np.arange(n) / SR

    def ride(tt):       # the band's lowpass: opens over the intro, darkens for the break, closes in the outro
        f = np.interp(tt, [0, bar_s(8), bar_s(36), bar_s(36.2), bar_s(43.5), bar_s(44), bar_s(60), bar_s(64)],
                      [380, 5200, 5200, 2400, 2400, 5200, 5200, 1200])
        return np.where(tt > bar_s(64.5), 1200 * (400 / 1200) ** np.clip((tt - bar_s(64.5)) / 3, 0, 1), f)

    s = fx.highpass(b["sample"], 150, 2)
    s = fx.bitcrush(s, 12, 26040, mix=0.75)
    s = fx.saturate(fx.shelf_tilt(s, 800, -0.6), 1.4)
    b["sample"] = fx.wow_flutter(fx.sweep_lowpass(s, ride), wow_ms=1.2, wow_hz=0.5, seed=51)
    fl = fx.lowpass(fx.highpass(b["flute"], 450, 2), 7000, 2)
    b["flute"] = fx.pan(fx.wow_flutter(fx.bitcrush(fl, 12, 26040, mix=0.35), seed=52), 0.12)
    pn = fx.shelf_tilt(fx.lowpass(fx.highpass(b["piano"], 160, 2), 12000, 2), 1000, 0.6)
    pn = fx.detune_chorus(fx.saturate(pn, 1.3), cents=7, rate=0.8, mix=0.4)
    pn = fx.compress(pn, -20, 3.5, 3, 80, makeup_db=1)
    b["piano"] = fx.sweep_lowpass(fx.bitcrush(pn, 12, 26040, mix=0.3), ride)
    e = fx.lowpass(fx.highpass(b["ep"], 250, 2), 5000, 2)
    b["ep"] = fx.autopan(fx.detune_chorus(e, cents=6, rate=0.4, mix=0.4), rate=BPM / 60 / 8, depth=0.35)
    b["pad"] = fx.widen(fx.detune_chorus(fx.lowpass(fx.highpass(b["pad"], 200, 2), 4500, 2), cents=10, rate=0.2,
                                         mix=0.5), 1.6)
    oc = fx.lowpass(fx.highpass(b["ocarina"], 500, 2), 6500, 2)
    b["ocarina"] = fx.wow_flutter(fx.bitcrush(oc, 12, 26040, mix=0.4), wow_ms=2.0, seed=53)

    bb = fx.saturate(fx.lowpass(b["bass"], 1300, 2), 1.7)
    b["bass"] = fx.mono_below(fx.compress(bb, -20, 3, 8, 110), 300)
    b["sub"] = fx.lowpass(b["sub"], 100, 2)
    b["kick"] = fx.compress(fx.saturate(fx.lowpass(b["kick"], 7000, 2), 2.0), -14, 3, 3, 90)
    sn = fx.bitcrush(fx.highpass(b["snare"], 200, 2), 12, 26040)
    b["snare"] = fx.lowpass(fx.saturate(sn, 1.4), 9500, 2)
    b["hats"] = fx.bitcrush(fx.lowpass(fx.highpass(b["hats"], 900, 2), 15000, 2), 12, 26040, mix=0.35)
    pc = fx.lowpass(fx.highpass(b["perc"], 250, 2), 9000, 2)
    b["perc"] = fx.autopan(fx.saturate(pc, 1.2), rate=BPM / 60 / 4, depth=0.5)

    lv = {k: fx.db(LEVELS[k]) / max(1e-9, fx.db(fx.gated_rms_db(b[k]))) for k in LEVELS}
    for k in LEVELS:
        b[k] = b[k] * lv[k]

    # the salad spinner: pads, piano and the record through a rotary speaker over the break - slow, then spun up
    i0, i1 = tl.i(BREAK[0] * 4), tl.i(BREAK[1] * 4)
    lead = int(0.05 * SR)

    def rate(tt):
        tt = tt + bar_s(BREAK[0]) - lead / SR
        u = np.clip((tt - bar_s(SPIN[0])) / (bar_s(SPIN[1] - 0.25) - bar_s(SPIN[0])), 0, 1)
        return 0.8 * (7.5 / 0.8) ** (u ** 1.5)

    for k, wet in (("pad", 1.0), ("piano", 1.0), ("sample", 0.8)):
        seg = b[k][i0 - lead:i1 + lead]
        spun = rotary(seg, rate)
        xf = np.clip(np.minimum(np.arange(len(seg)), len(seg) - 1 - np.arange(len(seg))) / lead, 0, 1)[:, None]
        b[k][i0 - lead:i1 + lead] = seg * (1 - xf * wet) + spun * xf * wet

    # space: a bright plate on piano, flute and EP; a long hall under the break; a gated room on the claps; a
    # dotted-quarter echo on the flute and a quarter ping-pong on the ocarina
    plate = fx.reverb(b["piano"] * 0.35 + b["flute"] * 0.5 + b["ep"] * 0.5 + b["pad"] * 0.3, seconds=1.7,
                      damp_hz=4800, predelay_ms=15, seed=54)
    hall = fx.reverb(b["ocarina"] + b["pad"] * 0.6 + b["piano"] * 0.3, seconds=4.0, damp_hz=3800, predelay_ms=40,
                     seed=55)
    claps = [x.start * 60 / BPM for x in drums["snare"] if x.pitch == PT.CLAP and x.vel > 70]
    gate = np.zeros(n, dtype=np.float32)
    hold, rel = int(0.13 * SR), int(0.018 * SR)
    shape = np.concatenate([np.ones(hold), np.linspace(1, 0, rel)]).astype(np.float32)
    for c in claps:
        i = int(c * SR)
        m = min(len(shape), n - i)
        if m > 0:
            gate[i:i + m] = np.maximum(gate[i:i + m], shape[:m])
    gated = fx.reverb(b["snare"], seconds=1.4, damp_hz=7000, predelay_ms=4, hp=300, seed=56) * gate[:, None]
    fecho = fx.tape_delay(b["flute"], 1.5, BPM, feedback=0.3, repeats=4, lp=3600, pingpong=False)
    oecho = fx.tape_delay(b["ocarina"], 1.0, BPM, feedback=0.45, repeats=6, lp=4200, pingpong=True)

    duck = fx.duck(n, tl.kicks, depth_db=2.5, attack_ms=5, release_ms=160)[:, None]
    music = (b["sample"] + b["flute"] + b["piano"] + b["ep"] + b["pad"]) * duck + b["ocarina"]
    music = music - 0.45 * fx.bandpass(music, 700, 1800)          # a mid scoop, room for the claps and the bass
    low = (b["bass"] + b["sub"]) * fx.duck(n, tl.kicks, depth_db=4.0, release_ms=130)[:, None]
    drum_bus = fx.compress(b["kick"] + b["snare"] + b["hats"] + b["perc"] + gated * 0.6, -16, 2.5, 8, 140,
                           makeup_db=1)
    drum_bus = fx.tape(drum_bus, 1.3, 13000)

    secs = n / SR

    def fit(x):
        return np.pad(x[:n], ((0, n - len(x[:n])), (0, 0)))

    vinyl = fit(AMB.vinyl(secs, -50, seed=57, crackle_rate=1.5))
    hiss = fit(AMB.hiss(secs, -52, seed=58))

    mix = (music + low + drum_bus * 0.85 + plate * 0.3 + hall * 0.45 + fecho * 0.12 + oecho * 0.3 + vinyl + hiss)
    mix = fx.wow_flutter(mix, wow_ms=0.35, wow_hz=0.3, flutter_ms=0.04, drift_ms=0.3, seed=50)
    stop = int((bar_s(BARS) + 6.0) * SR)
    mix = fx.fade(mix[:stop], fin_s=0.3, fout_s=4.0)
    out = M.master(mix, -14.0, -1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{FILE}.wav"
    M.write(path, out, tags={"title": TITLE, "artist": "Gielinor Lo-Fi", "album": "Gielinor Lo-Fi",
                             "comment": "Lo-fi reinterpretation of 'Flute Salad' (Ian Taylor, Jagex) from Old School RuneScape"})
    if stems:
        sd = out_dir / "stems"
        sd.mkdir(exist_ok=True)
        import soundfile as sf
        for k, x in {**b, "plate": plate, "hall": hall, "gated": gated, "fecho": fecho, "oecho": oecho}.items():
            sf.write(str(sd / f"{k}.wav"), x[:stop], SR, subtype="PCM_16")
    rep = M.report(out, BPM, [(nm, b0, b0 + nb) for nm, b0, _, nb in SECTS])
    rep["file"] = str(path)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    print(json.dumps(build(ROOT / "build" / "flute_salad", stems="--stems" in sys.argv), indent=1))
