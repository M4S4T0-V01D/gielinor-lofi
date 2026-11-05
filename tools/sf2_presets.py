"""List the presets (bank, program, name) in a SoundFont 2 file by reading its phdr chunk."""
import struct, sys

def presets(path):
    data = open(path, "rb").read()
    i = data.find(b"phdr")
    size = struct.unpack("<I", data[i + 4:i + 8])[0]
    out = []
    for k in range(size // 38 - 1):
        rec = data[i + 8 + k * 38: i + 8 + (k + 1) * 38]
        name = rec[:20].split(b"\0")[0].decode("latin1")
        prog, bank = struct.unpack("<HH", rec[20:24])
        out.append((bank, prog, name))
    return sorted(out)

if __name__ == "__main__":
    for b, p, n in presets(sys.argv[1]):
        print(f"{b:>3} {p:>3} {n}")
