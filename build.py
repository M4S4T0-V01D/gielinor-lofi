"""Build Gielinor Lo-Fi tracks.

usage: python -I build.py [song ...] [--stems]      (no song = every song in ALBUM that has a script)
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# track order for the album; each name is songs/<name>.py with a build(out_dir, stems) function
ALBUM = ["harmony", "autumn_voyage", "dogs_of_war", "yesteryear", "book_of_spells", "sea_shanty", "castle_wars", "camelot", "emotion"]


def run(name: str, stems: bool):
    path = ROOT / "songs" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"songs.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build(ROOT / "build" / name, stems=stems)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = args or [n for n in ALBUM if (ROOT / "songs" / f"{n}.py").exists()]
    for n in names:
        print(n, json.dumps(run(n, "--stems" in sys.argv)))
