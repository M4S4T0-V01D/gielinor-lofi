"""Note data: load the original RuneScape MIDI, remap its meter, swing and humanize, write MIDI stems.

Times are in beats (quarter notes) throughout; the tempo only enters when audio is rendered.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path

import mido

NAMES = "C C# D D# E F F# G G# A A# B".split()


@dataclass(frozen=True)
class Note:
    start: float      # beats
    dur: float        # beats
    pitch: int
    vel: int          # 1..127

    @property
    def end(self) -> float:
        return self.start + self.dur


def name(p: int) -> str:
    return f"{NAMES[p % 12]}{p // 12 - 1}"


@dataclass
class Source:
    """An original song: notes per track, the GM program each track uses, and its meter."""
    path: Path
    tracks: dict[str, list[Note]]
    programs: dict[str, int]
    drums: set[str]
    beats_per_bar: int
    bpm: float


def load(path: str | Path) -> Source:
    m = mido.MidiFile(str(path))
    tpb = m.ticks_per_beat
    tracks, programs, drums = {}, {}, set()
    num, bpm = 4, 120.0
    for t in m.tracks:
        tick, on, out = 0, {}, []
        for x in t:
            tick += x.time
            if x.type == "time_signature":
                num = x.numerator
            elif x.type == "set_tempo":
                bpm = round(6e7 / x.tempo, 3)
            elif x.type == "program_change":
                programs.setdefault(t.name, x.program)
            elif x.type == "note_on" and x.velocity > 0:
                on[(x.channel, x.note)] = (tick, x.velocity)
                if x.channel == 9:
                    drums.add(t.name)
            elif x.type in ("note_off", "note_on") and (x.channel, x.note) in on:
                s, v = on.pop((x.channel, x.note))
                out.append(Note(s / tpb, max(1, tick - s) / tpb, x.note, v))
        if out:
            tracks[t.name] = sorted(out, key=lambda n: (n.start, n.pitch))
    return Source(Path(path), tracks, programs, drums, num, bpm)


# --- meter -------------------------------------------------------------------------------------------------------

# A 3/4 bar laid into a 4/4 bar as 3+3+2 eighths: beats 1 and 2 stretch to a dotted quarter each, beat 3 stays a
# quarter. Eighths land on the 16th grid (0.5 -> 0.75, 1.5 -> 2.25, 2.5 -> 3.5), which swings like a hip-hop loop.
WALTZ_TO_FOUR = ((0.0, 0.0), (1.0, 1.5), (2.0, 3.0), (3.0, 4.0))
STRAIGHT = None


def _map_pos(p: float, src_bar: int, dst_bar: int, curve) -> float:
    bar, r = divmod(p, src_bar)
    if curve is None:
        return bar * dst_bar + r * dst_bar / src_bar
    for (a0, b0), (a1, b1) in zip(curve, curve[1:]):
        if r <= a1 + 1e-9:
            return bar * dst_bar + b0 + (r - a0) * (b1 - b0) / (a1 - a0)
    return (bar + 1) * dst_bar


def remap(notes: list[Note], src_bar: int, dst_bar: int = 4, curve=WALTZ_TO_FOUR) -> list[Note]:
    """Lay notes from src_bar-beat bars into dst_bar-beat bars, bar for bar."""
    if src_bar == dst_bar:
        return list(notes)
    out = []
    for n in notes:
        s = _map_pos(n.start, src_bar, dst_bar, curve)
        e = _map_pos(n.end, src_bar, dst_bar, curve)
        out.append(replace(n, start=s, dur=max(0.05, e - s)))
    return out


# --- editing -----------------------------------------------------------------------------------------------------

def window(notes: list[Note], b0: float, b1: float, bar: int = 4, keep_tails: bool = False) -> list[Note]:
    """Notes starting in bars [b0, b1), shifted so bar b0 is at 0. Durations are cut at the window end."""
    s0, s1 = b0 * bar, b1 * bar
    out = []
    for n in notes:
        if s0 - 1e-6 <= n.start < s1 - 1e-6:
            d = n.dur if keep_tails else min(n.dur, s1 - n.start)
            out.append(replace(n, start=n.start - s0, dur=d))
    return out


def shift(notes, beats=0.0, semis=0):
    return [replace(n, start=n.start + beats, pitch=n.pitch + semis) for n in notes]


def scale_vel(notes, k: float, lo: int = 1, hi: int = 127):
    return [replace(n, vel=max(lo, min(hi, round(n.vel * k)))) for n in notes]


def top_line(notes: list[Note], min_gap: float = 0.05) -> list[Note]:
    """The highest note at each onset (the melody out of a voiced part)."""
    by = {}
    for n in notes:
        k = round(n.start / min_gap)
        if k not in by or n.pitch > by[k].pitch:
            by[k] = n
    return [by[k] for k in sorted(by)]


def swing(notes: list[Note], amount: float) -> list[Note]:
    """Delay notes on odd 16ths. amount 0.5 = straight, 0.6 = the off-16th sits at 60% of the eighth."""
    off = (amount - 0.5) * 0.5
    out = []
    for n in notes:
        q = n.start * 4
        if abs(q - round(q)) < 0.08 and round(q) % 2 == 1:
            out.append(replace(n, start=n.start + off, dur=max(0.05, n.dur - off)))
        else:
            out.append(n)
    return out


def humanize(notes: list[Note], bpm: float, ms: float, vel: int, seed: int, push_ms: float = 0.0) -> list[Note]:
    """Random timing (ms, gaussian) and velocity (+-vel) offsets; push_ms moves everything late (+) or early (-)."""
    rng = random.Random(seed)
    beat_ms = 60000.0 / bpm
    out = []
    for n in notes:
        dt = (rng.gauss(0, ms / 2) + push_ms) / beat_ms
        out.append(replace(n, start=max(0.0, n.start + dt),
                           vel=max(1, min(127, n.vel + rng.randint(-vel, vel)))))
    return out


# --- writing -----------------------------------------------------------------------------------------------------

def write_midi(path: str | Path, parts, bpm: float, tpb: int = 960) -> None:
    """parts: [(notes, program, bank, channel)]. Bank 128 on channel 9 means a drum kit."""
    m = mido.MidiFile(type=1, ticks_per_beat=tpb)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=round(6e7 / bpm), time=0))
    m.tracks.append(meta)
    for notes, program, bank, ch in parts:
        t = mido.MidiTrack()
        msb, lsb = (bank, 0) if bank < 128 else (0, 0)
        t.append(mido.Message("control_change", channel=ch, control=0, value=min(127, msb), time=0))
        t.append(mido.Message("control_change", channel=ch, control=32, value=lsb, time=0))
        t.append(mido.Message("program_change", channel=ch, program=program, time=0))
        ev = []
        for n in notes:
            a = round(n.start * tpb)
            b = max(a + 1, round(n.end * tpb))
            ev.append((a, 1, n.pitch, n.vel))
            ev.append((b, 0, n.pitch, 0))
        ev.sort(key=lambda e: (e[0], e[1]))   # note-offs before note-ons at the same tick
        now = 0
        for tick, kind, p, v in ev:
            t.append(mido.Message("note_on" if kind else "note_off", channel=ch, note=p, velocity=v, time=tick - now))
            now = tick
        m.tracks.append(t)
    m.save(str(path))
