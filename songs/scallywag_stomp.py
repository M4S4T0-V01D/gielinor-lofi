"""Gielinor Lo-Fi - The Scallywag Stomp (after Ian Taylor's 'Sea Shanty' and 'Sea Shanty 2', Old School RuneScape).

The whole night at Port Sarim, played by one band. Sea Shanty (Port Sarim) plays as it is up to the end of its
swaying A4; then it turns on its own B7 bars, which in the band's D minor are A7, the dominant of D major too. The
band resolves it into D major, and Sea Shanty's tune, turned major, trades lines with Sea Shanty 2's (4/4, A major,
100 bpm, chopped and played as a 12/8 jig) until Sea Shanty 2 takes over; over the first four bars the same
players speed up from 76 to 88 while the harbour falls away behind them. Same record treatment
(the horn tune on the accordion, the guitars on pizzicato, the accordion's bridge on harmonica, the piccolo on
flute), the same Rhodes, bassist, jazz kit, grooves, chains and room.
Its own moves: the accelerando morph across the shared dominant, the two tunes swapping lines in major, tankards drummed on the table into the bridge,
stop-time in the bridge, two tankards clashing for the last tune, and the door opening on the sea at the end.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from engine import ambience as AMB, fx, master as M, notes as N, parts as PT  # noqa: E402
from engine.render import SR, fluid  # noqa: E402
from engine.sampler import Record, Timeline, edge  # noqa: E402
from songs import sea_shanty as S1  # noqa: E402

TITLE = "The Scallywag Stomp"
FILE = "The Scallywag Stomp"
BPM = 88.0                    # the second half, once it has sped up; the first is Sea Shanty at 76
SRC = N.load(ROOT / "source" / "sea_shanty2.mid")


def jt(t):
    """Swing a straight 4/4 position into the jig: each beat's eighths become triplets (2:1), its 16ths fall in."""
    b = math.floor(t)
    f = t - b
    return b + (f * 4 / 3 if f < 0.5 else 2 / 3 + (f - 0.5) * 2 / 3)


def jig(notes, semis=0):
    out = []
    for x in notes:
        s, e = jt(x.start), jt(x.start + x.dur)
        out.append(N.Note(s, max(0.05, e - s), x.pitch + semis, x.vel))
    return out


# the morph: 76 -> 88 bpm, linear in beats over the first four bars, then 88. W(beat) is seconds from bar 0;
# the second half is rendered at 60 bpm (a beat is a second) with every note moved through W.
RAMP = 16
_K = 60 * RAMP / (BPM - 76)


def W(b):
    if b <= RAMP:
        return _K * math.log((76 + (BPM - 76) * b / RAMP) / 76)
    return _K * math.log(BPM / 76) + (b - RAMP) * 60 / BPM


def warp(notes):
    return [N.Note(W(x.start), W(x.start + x.dur) - W(x.start), x.pitch, x.vel) for x in notes]


def bar_s(b):
    return W(b * 4)


# source bars (4/4, one song bar each): 0-3 the horn alone, 4-19 the tune with the band (nylon from 11), 20-31 the
# oom-pah without it (piccolo 24-31), 32-39 the accordion's bridge over A7 (bandoneon 36-39), 40-47 the tune with
# marimba, 48-51 the last line over A / E. Chords per source bar, named here in the song's D.
ROOTS = "AAAAAAAAEEAAAAAAEEAA" + "A" * 12 + "C" * 8 + "AAAAEEAA" + "AGAG"
PT.VOICINGS.update({"A9": PT.v("A2", "G3", "B3", "C#4", "E4"), "A13": PT.v("A2", "G3", "C#4", "F#4"),
                    "D13": PT.v("D3", "C4", "F#4", "B4"), "A/C#": PT.v("C#3", "G3", "A3", "E4"),
                    "Bm7": PT.v("B2", "A3", "D4", "F#4")})

# the swap: the front of the jig trades lines between the two songs. Song bars 0-3 and 8-11 are Sea Shanty's
# tune (its song bars 0-3, then 4-7) turned major, which makes its "F F E F" Sea Shanty 2's "F# E F#"; song bars
# 4-7 and 12-15 answer with Sea Shanty 2's (source bars 4-11), and from bar 16 Sea Shanty 2 runs on from its
# bar 12. Its horn-alone bars 0-3 are cut.
SWAP = {0: 0, 1: 1, 2: 2, 3: 3, 8: 4, 9: 5, 10: 6, 11: 7}                 # song bar -> Sea Shanty song bar
ORDER = [None] * 4 + [4, 5, 6, 7] + [None] * 4 + [8, 9, 10, 11] + list(range(12, 52))  # -> Sea Shanty 2 bar
MAJOR = {5: 6, 0: 1, 10: 11}                                              # D minor -> D major: F, C, Bb raised


def chord(s):
    return {"A": "D69" if s % 2 == 0 else "Dadd9", "E": "A9" if s % 2 == 0 else "A13",
            "C": "D13" if s % 2 == 0 else "Dadd9", "G": "A/C#"}[ROOTS[s]]


def swap_chord(s):
    """Sea Shanty's chord on its source bar s (E minor), in D and turned major."""
    return {"E": "D69" if s % 4 < 2 else "Dadd9", "B": "A13", "C": "Bm7", "D": "A/C#"}[S1.ROOTS[s]]


def bar_chart(k):
    if k in SWAP:
        j = SWAP[k]
        return [(swap_chord(2 * j), 2), (swap_chord(2 * j + 1), 2)]
    return [(chord(ORDER[k]), 4)]


def chart(s0, nb):
    return [bar_chart(k) for k in range(s0, s0 + nb)]


def chop(notes):
    """Sea Shanty 2's notes (in source beats) laid out in ORDER."""
    out = []
    for k, s in enumerate(ORDER):
        if s is not None:
            out += [replace(x, start=x.start + (k - s) * 4) for x in notes if s * 4 <= x.start < s * 4 + 4]
    return out


def swap_line(track, semis):
    """Sea Shanty's track (song beats, as triplets) on SWAP's bars, turned major, transposed by semis."""
    src = N.remap(S1.SRC.tracks[track], 3, 2, None)
    out = []
    for k, j in SWAP.items():
        for x in src:
            if j * 4 <= x.start < j * 4 + 4:
                c = x.pitch + S1.KEY
                c += MAJOR.get(c % 12, c % 12) - c % 12
                out.append(replace(x, start=x.start + (k - j) * 4, pitch=c + semis))
    return out


FORM = [("swap", 0, 16), ("jig A", 16, 8), ("oom-pah", 24, 12), ("stop-time", 36, 8), ("last tune", 44, 8),
        ("last orders", 52, 4)]
BARS = 56
END = 55 * 4 + 3              # the final hit: bar 55, beat 4

# the docks' grooves, played by the same drummer: the stomp for the tunes, four on the floor for the oom-pah,
# hits on the one for the stop-time, then the light feel building back
STOP = [dict(bus="kick", piece=PT.KICK, at=[0], vel=104), dict(bus="snare", piece=PT.SNARE, at=[0], vel=100)]
GROOVE = [(0, 4, S1.FULL, 4), (4, 20, S1.FULL, 4), (24, 12, S1.DANCE, 4), (36, 4, STOP, 4), (40, 2, S1.LIGHT, 4),
          (42, 2, S1.FULL, 2), (44, 8, S1.FULL, 4), (52, 3, S1.FULL, 4)]


# --- the tavern: synthesized, sparse (a crowd or dense rattle would read as rain) -----------------------------
def _pan(s, p):
    return np.stack([s * np.sqrt((1 - p) / 2), s * np.sqrt((1 + p) / 2)], axis=1)


def clinks(n, times, seed, cheers=()):
    """Pewter and glass: a mug set down or knocked here and there; at each of cheers, two tankards clash."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 2), dtype=np.float32)
    ratios, decays = [1.0, 2.76, 5.4, 8.93], [0.16, 0.08, 0.045, 0.025]
    for t0, big in [(t, False) for t in times] + [(t, True) for t in cheers]:
        for k in range(2 if big else 1):
            f0 = rng.uniform(2300, 3100) if big else rng.uniform(1700, 2600)
            i = int((t0 + k * rng.uniform(0.012, 0.025)) * SR)
            m = min(int(0.9 * SR), n - i)
            if m <= 0:
                continue
            tt = np.arange(m) / SR
            stretch = 2.2 if big else 1.0
            s = sum(np.sin(2 * np.pi * f0 * r * tt + rng.uniform(0, 6)) * np.exp(-tt / (d * stretch)) * w
                    for r, d, w in zip(ratios, decays, [1, 0.5, 0.3, 0.15]))
            s = s * (1 - np.exp(-tt / 0.0008)) * (1.0 if big else rng.uniform(0.3, 0.7))
            x[i:i + m] += _pan(s, (-0.25 + 0.5 * k) if big else rng.uniform(-0.7, 0.7))
    x = fx.highpass(x, 900, 2)
    return x * 0.6 + fx.reverb(x, seconds=1.0, damp_hz=4000, predelay_ms=12, hp=600, seed=seed) * 0.5


def thumps(n, hits, seed):
    """Tankards banged on the table: a wooden knock, and the mugs jumping. hits: (seconds, 0..1)."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 2), dtype=np.float32)
    for t0, a in hits:
        i = int(t0 * SR)
        m = min(int(0.4 * SR), n - i)
        if m <= 0:
            continue
        tt = np.arange(m) / SR
        f = 70 + 80 * np.exp(-tt / 0.03)
        body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.08)
        knock = rng.standard_normal(m) * np.exp(-tt / 0.012)
        s = body + 0.5 * fx.lowpass(knock[:, None].repeat(2, 1), 1100, 2)[:, 0]
        j = int(rng.uniform(0.008, 0.02) * SR)
        f0 = rng.uniform(1300, 1800)
        rattle = np.sin(2 * np.pi * f0 * tt) * np.exp(-tt / 0.05) + 0.4 * np.sin(2 * np.pi * f0 * 2.76 * tt) * np.exp(-tt / 0.03)
        s[j:] += 0.18 * rattle[:m - j]
        x[i:i + m] += _pan(s * a * rng.uniform(0.85, 1.0), rng.uniform(-0.3, 0.3))
    return x + fx.reverb(x, seconds=0.9, damp_hz=2500, predelay_ms=10, hp=150, seed=seed) * 0.4


def door_open(n, t0, seed):
    """The tavern door: the latch lifting and a creak as it swings open."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=np.float32)

    def add(s, at):
        i = int(at * SR)
        m = min(len(s), n - i)
        if m > 0:
            x[i:i + m] += s[:m]
    c = rng.standard_normal(int(0.02 * SR)) * np.exp(-np.arange(int(0.02 * SR)) / (0.0015 * SR))
    add(fx.highpass(c[:, None].repeat(2, 1), 2000, 2)[:, 0] * 0.25, t0)
    m = int(1.1 * SR)
    rate = 55 + 35 * np.sin(np.linspace(0, 2.6, m)) + 12 * rng.standard_normal(m).cumsum() / np.sqrt(m)
    ph = np.cumsum(np.maximum(rate, 20)) / SR
    pulses = (np.diff(np.floor(ph), prepend=0) > 0).astype(np.float32) * rng.uniform(0.6, 1.0, m)
    env = np.sin(np.linspace(0, np.pi, m)) ** 0.7
    creak = fx.bandpass(pulses[:, None].repeat(2, 1), 650, 1700, order=2)[:, 0] * env * 3.0
    add(creak, t0 + 0.12)
    s = _pan(x, 0.15)
    return s + fx.reverb(s, seconds=1.2, damp_hz=2200, predelay_ms=8, hp=80, seed=seed) * 0.5


def place(tl, bus, rec, b0, nb, gain=1.0):
    """Chop the record a song bar at a time, as the first half does."""
    for k in range(b0, b0 + nb):
        tl.put(bus, edge(rec.beats(W(k * 4), W(k * 4 + 4), tail_beats=0.12), 2, 25), W(k * 4), gain)


def jig_half() -> dict:
    """Sea Shanty 2, by the first half's band, from the downbeat after the turn (t=0) to the end."""
    tl = Timeline(60.0, W(END + 8), tail_s=8.0)
    n = tl.n

    # --- the record: Sea Shanty 2 on the docks band's instruments --------------------------------------------
    # pressed a tritone-ish up: +6 in the MIDI and the record varispeeded down a semitone, so A major lands on D
    # major with a little tape drag, like the first half's -2. The accordion's bridge (song bars 36-43) is the harmonica.
    t = {k: jig(chop(v), 6) for k, v in SRC.tracks.items() if k not in SRC.drums}
    bridge = lambda x: 36 * 4 <= x.start < 44 * 4  # noqa: E731
    t["harmonica"] = [x for x in t["accordion"] if bridge(x)]
    t["accordion"] = [x for x in t["accordion"] if not bridge(x)]
    t["flute"] = N.shift(t["piccolo"], 0, -12)
    t["flute hi"] = N.shift(t["horn"], 0, 12)
    # the tune an octave down, in the docks accordion's chest register; a stray low note folds back up
    t["horn"] = [replace(x, pitch=x.pitch - 12 if x.pitch - 12 >= 46 else x.pitch) for x in t["horn"]]
    t["ss1"] = swap_line("accordion", 1)                  # the swap's Sea Shanty lines (+1 for the record's -1)
    t["ss1 pizz"] = swap_line("pizz", 1)
    src = replace(SRC, tracks={k: warp(v) for k, v in t.items()}, drums=set())
    rec = dict(semis=-1, dst_bar=4, curve=None)
    tune = Record(src, ["horn", "ss1"], 60.0, programs={"horn": 21, "ss1": 21}, vel={"ss1": 1.2}, **rec)  # the accordion's tune
    band = Record(src, ["steel guitar", "nylon", "marimba", "accordion", "bandeon", "ss1 pizz"], 60.0,
                  programs={"steel guitar": 45, "nylon": 45, "marimba": 45, "accordion": 21, "bandeon": 21,
                            "ss1 pizz": 45},
                  vel={"steel guitar": 0.8, "nylon": 0.75, "marimba": 0.8, "accordion": 0.65, "bandeon": 0.7,
                       "ss1 pizz": 0.8}, **rec)
    harmonica = Record(src, ["harmonica"], 60.0, programs={"harmonica": 22}, **rec)
    flute = Record(src, ["flute", "flute hi"], 60.0, programs={"flute": 73, "flute hi": 73},
                   vel={"flute hi": 0.85}, **rec)

    place(tl, "sample", tune, 0, 24)                                    # the swap (speeding up), then jig A
    place(tl, "sample", band, 0, 24, 0.85)
    place(tl, "flute", flute, 16, 8)                                    # the flute on jig A
    place(tl, "sample", band, 24, 12, 0.9)                              # oom-pah: the band, the flute's tune
    place(tl, "flute", flute, 28, 8)
    place(tl, "sample", tune, 35, 1, 0.8)
    place(tl, "sample", harmonica, 36, 8)                               # stop-time: the harmonica runs on alone
    for k in range(36, 40):                                             # ... the band only hits the one
        tl.put("sample", edge(band.beats(W(k * 4), W(k * 4) + 0.35), 2, 60), W(k * 4), 0.85)
    place(tl, "sample", band, 40, 4, 0.85)
    place(tl, "sample", tune, 44, 12)                                   # the last tune and last orders
    place(tl, "sample", band, 44, 11, 0.85)
    tl.put("sample", edge(band.beats(W(55 * 4), W(55 * 4 + 4)), 2, 40), W(55 * 4), 0.85)
    tl.put("sample", edge(band.beats(W(END), W(END) + 2.5), 2, 300), W(END), 0.9)

    # --- the Rhodes and the bass: the docks' players -----------------------------------------------------------
    rhodes = PT.comp(chart(0, 24), start_bar=0, vel=54, seed=31, bpm=BPM)
    rhodes += PT.comp(chart(24, 12), start_bar=24, vel=46, style="pad", bpm=BPM)    # room for the flute
    rhodes += [x for x in PT.comp(chart(36, 4), start_bar=36, vel=60, bpm=BPM) if x.start % 4 < 0.3]
    rhodes += PT.comp(chart(40, 15), start_bar=40, vel=54, seed=32, bpm=BPM)
    rhodes = jig(rhodes)
    rhodes = [replace(x, dur=0.5) if 36 * 4 <= x.start < 40 * 4 else x for x in rhodes]   # stabs on the one
    rhodes += [N.Note(END + 0.01 * j, 5, p, 58) for j, p in enumerate(PT.VOICINGS["D69"])]
    line = N.shift(N.top_line(N.window(jig(SRC.tracks["horn"], 5), 40, 48)), 44 * 4)
    lead = PT.melody_double(line, octave=-1, vel=52)                                          # the last tune, doubled
    tl.put_full("rhodes", fluid([(warp(rhodes), 4, 0, 0)], 60.0, tail=8))
    tl.put_full("lead", fluid([(warp(lead), 4, 0, 1)], 60.0, tail=4))

    bass = PT.bass(chart(0, 55), start_bar=0, seed=33, low=PT.P["D1"], high=PT.P["C#2"])
    bass = [x if not 36 * 4 <= x.start < 40 * 4 else replace(x, dur=1.2) for x in bass
            if not (36 * 4 <= x.start < 40 * 4) or x.start % 4 < 0.3]
    bass += [N.Note(55 * 4, 1.3, PT.P["D2"], 88), N.Note(55 * 4 + 2, 0.9, PT.P["C#2"], 80),
             N.Note(END, 4, PT.P["D1"], 96)]
    bass = warp(N.humanize(jig(bass), BPM, 6, 4, seed=34, push_ms=6))
    tl.put_full("bass", fluid([(bass, 33, 0, 2)], 60.0, tail=4))
    tl.put_full("sub", PT.sub(bass, 60.0, n))

    drums = {"kick": [], "snare": [], "hats": []}
    for b0, nb, feel, fe in GROOVE:
        for piece, ns in PT.groove(nb, b0, feel, seed=60 + b0, fill_every=fe, steps=12, bpm=BPM).items():
            drums[piece] += ns
    drums["kick"] += [N.Note(END, 0.5, PT.KICK, 110)]
    drums["snare"] += [N.Note(END, 0.5, PT.SNARE, 104)]
    drums = {k: warp(v) for k, v in drums.items()}
    kicks = [x.start for x in drums["kick"]]
    for piece, ns in drums.items():
        tl.put_full(piece, fluid([(ns, S1.KIT, 128, 9)], 60.0, tail=2))

    # --- the docks' chains and room ----------------------------------------------------------------------------
    b = tl.bus
    for k in S1.LEVELS:
        b.setdefault(k, np.zeros((n, 2), dtype=np.float32))
    t_end = W(END)

    def ride(tt):       # open, a little darker for the oom-pah and the bridge, closing after the last hit
        f = np.full_like(tt, 5000.0)
        f = np.where((tt > bar_s(24)) & (tt < bar_s(40)), 4400.0, f)
        return np.where(tt > t_end + 0.6, 5000 * (400 / 5000) ** np.clip((tt - t_end - 0.6) / 3.0, 0, 1), f)
    mix, verb, echo = S1.mix_band(b, kicks, n, BPM, ride)

    # the tavern: tankards drummed on the table into the bridge, a mug here and there, the clash for the last tune
    tb = [(W(32 * 4 + k), 0.5) for k in (0, 2)] + [(W(33 * 4 + k), 0.6) for k in range(4)]
    tb += [(W(34 * 4 + k / 3), 0.65) for k in range(12) if k % 3 != 1]
    tb += [(W(35 * 4 + k / 3), 0.7 + 0.3 * k / 11) for k in range(12)]
    table = fx.level(thumps(n, tb, seed=49), -30)
    rng = np.random.default_rng(50)
    mugs = sorted(rng.uniform(bar_s(1.5), bar_s(54), 12))
    tavern = fx.level(clinks(n, mugs, seed=51, cheers=(bar_s(44) - 0.06,)), -41)
    return dict(band=mix, table=table, tavern=tavern, b=b, verb=verb, n=n, t_end=t_end)


def build(out_dir: Path, stems: bool = False) -> dict:
    first = S1.render(stomp=True)
    second = jig_half()
    t2 = S1.bar_s(49)                         # the downbeat after the turn, where the A7 resolves
    i2 = int(round(t2 * SR))
    t_end = t2 + second["t_end"]
    n = int((t_end + 11.0) * SR)
    t = np.arange(n) / SR

    # the docks: Sea Shanty's band, then the harbour falls away behind the morph (dulled, then gone)
    m0, m1 = t2, t2 + bar_s(4)
    docks = np.zeros((n, 2), dtype=np.float32)
    docks[:min(n, len(first["band"]))] += first["band"][:n]
    harbour = np.zeros((n, 2), dtype=np.float32)
    h = first["sea"] + first["gulls"]
    harbour[:min(n, len(h))] += h[:n]
    harbour = fx.sweep_lowpass(harbour, lambda tt: 18000 * (500 / 18000) ** np.clip((tt - m0) / (m1 - m0), 0, 1))
    harbour *= np.interp(t, [0, m0, m1], [1, 1, 0])[:, None]

    # the tavern half: the same band, matched to the docks' level and a touch hotter as the night goes on
    band2 = second["band"]
    rms1 = fx.gated_rms_db(first["band"][int(S1.bar_s(4) * SR):int(S1.bar_s(48) * SR)])
    rms2 = fx.gated_rms_db(band2[int(bar_s(4) * SR):int(second["t_end"] * SR)])
    g = 10 ** ((rms1 - rms2 + 0.5) / 20)
    room_in = np.interp(np.arange(len(band2)) / SR, [0, bar_s(4)], [0, 1])[:, None]
    inside = np.zeros((n, 2), dtype=np.float32)
    seg = band2 * g + second["table"] + second["tavern"] * room_in
    m = min(len(seg), n - i2)
    inside[i2:i2 + m] = seg[:m]

    # the end: a mug set down, the door opens, and the sea and a gull come back in
    opened = t_end + 3.0
    ends = door_open(n, opened, seed=56) * 0.35 + clinks(n, (t_end + 2.0,), seed=57) * 0.4
    secs = (n + 0.5) / SR
    sea2 = AMB.waves(secs, -29, seed=58)[:n]
    sea2 = fx.sweep_lowpass(sea2, lambda tt: 300 * (16000 / 300) ** np.clip((tt - opened - 0.15) / 0.8, 0, 1))
    sea2 = sea2 * np.interp(t, [0, opened, opened + 0.8, secs], [0, 0, 1, 1])[:, None]
    gull = AMB.gulls(secs, -33, seed=59, rate=0, times=(opened + 1.5,))[:n]

    vinyl = AMB.vinyl(secs, -47, seed=53, crackle_rate=3.0)[:n]       # dense crackle would read as rain
    hiss = AMB.hiss(secs, -56, seed=54)[:n]
    mix = docks + harbour + inside + ends + sea2 + gull + vinyl + hiss
    mix = fx.wow_flutter(mix, wow_ms=0.4, flutter_ms=0.03, drift_ms=0.3, seed=29)
    mix = fx.fade(mix, fin_s=1.0, fout_s=5.0)
    out = M.master(mix, -14.0, -1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{FILE}.wav"
    M.write(path, out, tags={"title": TITLE, "artist": "Gielinor Lo-Fi", "album": "Gielinor Lo-Fi",
                             "comment": "Lo-fi reinterpretation of 'Sea Shanty' and 'Sea Shanty 2' (Ian Taylor, Jagex) "
                                        "from Old School RuneScape"})
    bar1 = S1.bar_s(1)
    j = lambda b: (t2 + bar_s(b)) / bar1  # noqa: E731  jig bars in first-half bars, for the report
    secs_form = [(nm, b0, b0 + nb) for nm, b0, nb in S1.FORM if nm != "outro"] + [("turn", 48, 49)]
    secs_form += [(nm, j(b0), j(b0 + nb)) for nm, b0, nb in FORM]
    if stems:
        sd = out_dir / "stems"
        sd.mkdir(exist_ok=True)
        import soundfile as sf
        i0 = int(round(bar_s(4) * SR))                     # jig stems from its bar 4, where it reaches 88
        for k, x in {**second["b"], "verb": second["verb"]}.items():
            sf.write(str(sd / f"{k}.wav"), x[i0:], SR, subtype="PCM_16")
    rep = M.report(out, S1.BPM, secs_form)
    rep["file"] = str(path)
    rep["jig_at_speed_s"] = round(t2 + bar_s(4), 2)
    rep["band_gain_db"] = round(rms1 - rms2 + 0.5, 2)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    print(json.dumps(build(ROOT / "build" / "scallywag_stomp", stems="--stems" in sys.argv), indent=1))
