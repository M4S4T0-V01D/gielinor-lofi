"""Original lo-fi parts written against a chord chart: Rhodes comping, bass, a sine sub, boom-bap drums.

A chart is a list of bars; each bar is a list of (chord_name, beats). Chord names are looked up in VOICINGS
(written in A minor; transpose at render time).
"""
from __future__ import annotations

import random

import numpy as np

from . import notes as N
from .notes import Note
from .render import SR

# MIDI numbers; A2 = 45, C4 = 60
P = {f"{n}{o}": 12 * (o + 1) + i for o in range(0, 8) for i, n in enumerate("C C# D D# E F F# G G# A A# B".split())}
P.update({f"{n}{o}": P[f"{s}{o}"] for o in range(0, 8) for n, s in (("Db", "C#"), ("Eb", "D#"), ("Gb", "F#"),
                                                                     ("Ab", "G#"), ("Bb", "A#"))})


def v(*names):
    return [P[x] for x in names]


VOICINGS = {
    "Am9":    v("A2", "G3", "B3", "C4", "E4"),
    "Am11":   v("A2", "G3", "C4", "D4", "E4"),
    "Am7":    v("A2", "G3", "C4", "E4"),
    "Cmaj9":  v("C3", "E3", "G3", "B3", "D4"),
    "Em7":    v("E3", "G3", "B3", "D4"),
    "Em11":   v("E3", "D4", "G4", "A4"),
    "Fmaj7":  v("F2", "E3", "A3", "C4"),
    "Fmaj9":  v("F2", "E3", "G3", "A3", "C4"),
    "G69":    v("G2", "B3", "D4", "E4", "A4"),
    "G13":    v("G2", "F3", "B3", "E4"),
    "Dm9":    v("D3", "F3", "C4", "E4"),
    "E7b9":   v("E2", "G#3", "D4", "F4"),
    "Amaj9":  v("A2", "G#3", "B3", "C#4", "E4"),
    "E9":     v("E2", "G#3", "D4", "F#4"),
    "Cmaj7":  v("C3", "B3", "E4", "G4"),
    # Book of Spells (C dorian)
    "Cm9":    v("C3", "A#3", "D4", "D#4", "G4"),
    "Cm11":   v("C3", "A#3", "D#4", "F4", "G4"),
    "F13":    v("F2", "D#3", "A3", "D4"),
    "F9":     v("F2", "D#3", "G3", "A3", "C4"),
    "Ebmaj9": v("D#3", "A#3", "D4", "F4", "G4"),
    "Ebmaj7": v("D#3", "G3", "A#3", "D4"),
    "Dm7":    v("D3", "F3", "A3", "C4"),
    "Cadd9":  v("C3", "G3", "D4", "E4"),
    "G5":     v("G2", "D3", "G3", "D4"),
    "Gm9":    v("G2", "F3", "A#3", "D4", "A4"),
    # Sea Shanty (E minor)
    "Em9":    v("E2", "D3", "G3", "B3", "F#4"),
    "D69":    v("D3", "F#3", "B3", "E4", "A4"),
    "Dadd9":  v("D3", "A3", "E4", "F#4"),
    "B7":     v("B2", "A3", "D#4", "F#4"),
    "B7b13":  v("B2", "A3", "D#4", "G4"),
    # Autumn Voyage (E minor, then C minor)
    "Em/D":   v("D3", "G3", "B3", "E4", "F#4"),
    "Em/B":   v("B2", "G3", "B3", "E4", "F#4"),
    "Fm9":    v("F2", "G#3", "C4", "D#4", "G4"),
    "Fm11":   v("F2", "D#3", "G#3", "A#3", "C4"),
    "Bb69":   v("A#2", "D3", "G3", "C4", "F4"),
    "Bb13":   v("A#2", "G#3", "D4", "G4"),
    "Cm/Bb":  v("A#2", "D#3", "G3", "C4", "D4"),
    "Abmaj9": v("G#2", "G3", "C4", "D#4", "A#4"),
    "Cm/G":   v("G2", "D#3", "G3", "C4", "D4"),
    "Cmaj9#11": v("C3", "E3", "B3", "D4", "F#4"),
    # Emotion (B minor; the B section in B major)
    "Bm9":    v("B2", "A3", "C#4", "D4", "F#4"),
    "Bm11":   v("B2", "A3", "D4", "E4", "F#4"),
    "F#7":    v("F#2", "E3", "A#3", "C#4"),
    "F#7b13": v("F#2", "E3", "A#3", "D4"),
    "F#9":    v("F#2", "E3", "G#3", "A#3", "C#4"),
    "Bmaj9":  v("B2", "A#3", "C#4", "D#4", "F#4"),
    # Newbie Melody (C mixolydian: C and Gm, then a C minor turn)
    "C69":    v("C3", "E3", "A3", "D4", "G4"),
    "Gm11":   v("G2", "F3", "A#3", "C4", "D4"),
    # Arabian (B phrygian; the B section over an E pedal)
    "Bphr":   v("B2", "F#3", "A3", "C4", "E4"),
    "Cmaj7/B": v("B2", "E3", "G3", "C4"),
}


def root(chord: str) -> int:
    return VOICINGS[chord][0] % 12


def chord_at(chart, bar, beat):
    pos = 0.0
    for name, beats in chart[bar]:
        if beat < pos + beats - 1e-9:
            return name
        pos += beats
    return chart[bar][-1][0]


# --- Rhodes ------------------------------------------------------------------------------------------------------

def comp(chart, start_bar=0, vel=58, style="lazy", seed=1, roll_ms=18.0, bpm=78.0):
    """Lazy jazz comping: a long downbeat, a soft anticipation, a push into beat 4. Strummed bottom-up."""
    rng = random.Random(seed)
    out = []
    beat_s = 60.0 / bpm
    for b, bar in enumerate(chart):
        base = (start_bar + b) * 4
        pos = 0.0
        for ci, (name, beats) in enumerate(bar):
            vs = VOICINGS[name]
            if style == "pad":
                hits = [(0.0, beats - 0.1, 1.0)]
            elif beats >= 4:
                hits = [(0.0, 1.55, 1.0), (1.75, 0.6, 0.72), (3.0, 0.9, 0.85)]
                if b % 4 == 3:
                    hits = [(0.0, 2.4, 1.0), (2.75, 1.1, 0.8)]
            elif beats >= 2:
                hits = [(0.0, beats - 0.35, 0.95)] + ([(beats - 0.5, 0.45, 0.65)] if beats >= 3 else [])
            else:
                hits = [(0.0, beats - 0.1, 0.85)]
            for off, dur, k in hits:
                for j, p in enumerate(vs):
                    dt = j * roll_ms / 1000 / beat_s * rng.uniform(0.6, 1.3)
                    vv = vel * k * (0.85 if j == 0 else 1.0) + rng.randint(-5, 5)
                    out.append(Note(base + pos + off + dt, dur, p, int(max(1, min(127, vv)))))
            pos += beats
    return out


def melody_double(line, octave=0, vel=62):
    """Play a melody (e.g. the Harmony hook) on the Rhodes."""
    return [Note(n.start, max(0.2, n.dur * 0.95), n.pitch + 12 * octave, vel) for n in line]


# --- bass --------------------------------------------------------------------------------------------------------

def bass(chart, start_bar=0, vel=88, seed=2, low=P["E1"], high=P["D#2"]):
    """Root on 1 held, a ghost on the 'and' of 2, root or fifth on 3+, an approach into the next chord."""
    rng = random.Random(seed)

    def reg(pc):
        p = low + (pc - low) % 12
        return p if p <= high else p - 12

    out = []
    nb = len(chart)
    for b, bar in enumerate(chart):
        base = (start_bar + b) * 4
        pos = 0.0
        for ci, (name, beats) in enumerate(bar):
            r = reg(root(name))
            if beats >= 4:
                out.append(Note(base, 1.3, r, vel))
                out.append(Note(base + 1.75, 0.3, r, int(vel * 0.55)))
                out.append(Note(base + 2.5, 0.9, r + (7 if rng.random() < 0.4 else 0), int(vel * 0.85)))
                nxt = chart[b + 1][0][0] if b + 1 < nb else name
                nr = reg(root(nxt))
                if nr != r and rng.random() < 0.8:
                    appr = nr + (-1 if rng.random() < 0.6 else 2)
                    out.append(Note(base + 3.5, 0.45, appr, int(vel * 0.7)))
                else:
                    out.append(Note(base + 3.5, 0.45, r + 12 if rng.random() < 0.3 else r, int(vel * 0.6)))
            else:
                out.append(Note(base + pos, max(0.3, beats - 0.3), r, vel if pos == 0 else int(vel * 0.9)))
            pos += beats
    return out


def sub(notes, bpm, n, level=0.35, cutoff_semis=None):
    """A sine an octave below the bass notes' root register: the felt, not heard, part of the low end."""
    out = np.zeros(n, dtype=np.float32)
    bs = 60.0 / bpm
    for x in notes:
        f = 440.0 * 2 ** ((x.pitch - 12 - 69) / 12)
        if f < 25:
            f *= 2
        i0 = int(x.start * bs * SR)
        L = int((x.dur * bs + 0.12) * SR)
        if i0 >= n:
            continue
        L = min(L, n - i0)
        t = np.arange(L) / SR
        env = np.minimum(1, t / 0.008) * np.exp(-t / (x.dur * bs * 1.6 + 0.05))
        rel = np.clip((x.dur * bs + 0.12 - t) / 0.12, 0, 1)
        out[i0:i0 + L] += np.sin(2 * np.pi * f * t) * env * rel * (x.vel / 127) * level
    return np.stack([out, out], axis=1)


# --- drums (GM numbers on a GeneralUser kit) ---------------------------------------------------------------------

KICK, STICK, SNARE, CHH, PHH, OHH, RIDE, SHAKER = 36, 37, 38, 42, 44, 46, 51, 70
CLAP, TAMB, RIDE_BELL, HI_WOOD, LO_WOOD, BRUSH = 39, 54, 53, 76, 77, 40
LO_FLOOR, HI_FLOOR, LO_TOM = 41, 43, 45


def groove(bars, start_bar=0, lines=(), seed=3, fill_every=4, swing_amt=0.59, bpm=78.0, steps=16):
    """A drum pattern from a spec, so every track gets its own groove (boom_bap is Harmony's). `lines` are dicts:
    bus ("kick" / "snare" / "hats", or any new bus such as "toms"), piece (GM note), at (steps in the bar), and
    optionally odd / fill (steps on odd bars / on the last bar of each fill_every), vel (int or (lo, hi)), accent
    ({step: vel}), p (the chance an off-downbeat hit plays), dur. steps=12 is a triplet (12/8) grid, which is never swung. Returns {bus: notes}."""
    rng = random.Random(seed)
    buses = {"kick": [], "snare": [], "hats": []}
    for b in range(bars):
        base = (start_bar + b) * 4
        last = (b % fill_every) == fill_every - 1
        for ln in lines:
            at = ln["fill"] if last and "fill" in ln else ln["odd"] if b % 2 and "odd" in ln else ln["at"]
            for st in at:
                if st != 0 and rng.random() > ln.get("p", 1.0):
                    continue
                v = ln.get("accent", {}).get(st, ln.get("vel", 90))
                v = rng.randint(*v) if isinstance(v, tuple) else v + rng.randint(-6, 6)
                buses.setdefault(ln["bus"], []).append(
                    Note(base + st * 4 / steps, ln.get("dur", 0.2), ln["piece"], max(1, min(127, v))))
    sw = (lambda ns: N.swing(ns, swing_amt)) if steps == 16 and swing_amt > 0.5 else (lambda ns: ns)
    feel = {"kick": (6, 4, 0, 0.0), "snare": (8, 5, 1, 14.0), "hats": (9, 6, 2, 4.0)}   # ms, vel, seed, push
    out = {}
    for k, ns in buses.items():
        ms, vel, sd, push = feel.get(k, (8, 5, 3, 8.0))
        out[k] = N.humanize(sw(ns), bpm, ms, vel, seed + sd, push_ms=push)
    return out


def boom_bap(bars, start_bar=0, feel="full", seed=3, fill_every=4, swing_amt=0.59, bpm=78.0):
    """16-step boom-bap with swing, a lazy snare, and variations every few bars. Returns {piece: notes}."""
    rng = random.Random(seed)
    k, s, h = [], [], []
    for b in range(bars):
        base = (start_bar + b) * 4
        last = (b % fill_every) == fill_every - 1
        if feel == "full":
            kick_steps = [0, 7, 10] if not last else [0, 7, 10, 15]
            if b % 2 == 1 and not last:
                kick_steps = [0, 3, 10]
        elif feel == "light":
            kick_steps = [0, 10] if b % 2 == 0 else [0, 7]
        else:          # "half": one kick, snare on 3
            kick_steps = [0]
        for st in kick_steps:
            k.append(Note(base + st / 4, 0.25, KICK, 104 if st == 0 else rng.randint(78, 92)))
        if feel == "half":
            s.append(Note(base + 2, 0.25, SNARE, 92))
        else:
            for st in (4, 12):
                s.append(Note(base + st / 4, 0.25, SNARE if feel == "full" else STICK, rng.randint(94, 104)))
            for st in ([15] if last else []) + ([6] if rng.random() < 0.35 else []) + ([11] if rng.random() < 0.3 else []):
                s.append(Note(base + st / 4, 0.2, SNARE, rng.randint(26, 40)))       # ghost notes
        for st in range(0, 16, 1 if feel == "full" else 2):
            if st % 2 == 1 and rng.random() < 0.35:
                continue
            accent = 94 if st % 4 == 0 else (80 if st % 2 == 0 else 54)
            piece = OHH if (last and st == 14 and feel != "half") else CHH
            h.append(Note(base + st / 4, 0.2, piece, accent + rng.randint(-8, 8)))
    sw = lambda ns: N.swing(ns, swing_amt)
    return {"kick": N.humanize(sw(k), bpm, 6, 4, seed),
            "snare": N.humanize(sw(s), bpm, 8, 5, seed + 1, push_ms=14),
            "hats": N.humanize(sw(h), bpm, 9, 6, seed + 2, push_ms=4)}
