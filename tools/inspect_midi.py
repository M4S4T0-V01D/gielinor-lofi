"""Print a bar-by-bar map of a MIDI file: which tracks play in each bar, and the notes of chosen tracks."""
import sys, mido
from collections import defaultdict

NAMES = "C C# D D# E F F# G G# A A# B".split()
def nn(p): return f"{NAMES[p % 12]}{p // 12 - 1}"

def notes(path):
    m = mido.MidiFile(path)
    tpb = m.ticks_per_beat
    out = []  # (track, name, start_beat, dur_beats, pitch, vel)
    num = 4
    for ti, t in enumerate(m.tracks):
        tick, on = 0, {}
        for x in t:
            tick += x.time
            if x.type == "time_signature": num = x.numerator
            if x.type == "note_on" and x.velocity > 0:
                on[(x.channel, x.note)] = (tick, x.velocity)
            elif x.type in ("note_off", "note_on") and (x.channel, x.note) in on:
                s, v = on.pop((x.channel, x.note))
                out.append((ti, t.name, s / tpb, (tick - s) / tpb, x.note, v, x.channel))
    return out, num

if __name__ == "__main__":
    path, mode = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "map"
    ns, num = notes(path)
    if mode == "map":
        names = {n[0]: n[1] for n in ns}
        bars = int(max(n[2] for n in ns) // num) + 1
        print("bar " + " ".join(f"{i:>2}" for i in sorted(names)))
        for b in range(bars):
            row = []
            for ti in sorted(names):
                c = sum(1 for n in ns if n[0] == ti and b * num <= n[2] < (b + 1) * num)
                row.append(f"{c:>2}" if c else " .")
            print(f"{b:>3} " + " ".join(row))
        for ti in sorted(names): print(ti, names[ti])
    else:  # notes <track> [bar0 bar1]
        ti = int(mode); b0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0; b1 = int(sys.argv[4]) if len(sys.argv) > 4 else 999
        cur = None
        for n in sorted((n for n in ns if n[0] == ti), key=lambda n: (n[2], n[4])):
            b = int(n[2] // num)
            if not b0 <= b < b1: continue
            if b != cur:
                print(f"\nbar {b}:", end=""); cur = b
            print(f" {nn(n[4])}@{n[2]-b*num:.2f}/{n[3]:.2f}v{n[5]}", end="")
        print()
