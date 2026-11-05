# Gielinor Lo-Fi

Lo-fi hip-hop reinterpretations of Old School RuneScape music. Each track samples a "record": the original
Jagex MIDI, rendered with a GM soundfont, then degraded and chopped. New Rhodes, bass and drum parts are written
against the song's own chords.

| # | Track | Source | Tempo / key | Status |
|---|-------|--------|-------------|--------|
| 01 | Harmony (Rainy Lumbridge) | `harmony.mid` (Ian Taylor), 3/4 A minor 100 bpm | 78 bpm, G minor | done |
| 02 | Autumn Voyage | `autumn voyage.mid` | 72 bpm, C# minor -> A minor | built, awaiting listen |
| 03 | Yesteryear | `yesteryear.mid` | - | planned |
| 04 | Book of Spells (Wizards' Tower) | `book of spells.mid` (Ian Taylor), 4/4 C dorian 110 bpm | 80 bpm, Bb dorian | done |
| 05 | Sea Shanty (Port Sarim) | `sea shanty.mid` (Ian Taylor), 3/4 E minor 130 bpm | 12/8 at 76 bpm, D minor | built, awaiting listen |
| 06 | Scape Main | `scape main.mid` | - | planned |
| 07 | Castle Wars | `castlewars.mid`, 4/4 E minor 180 bpm | 174 bpm jungle, D# minor | built, awaiting listen |
| - | Emotion (By the Hearth) | `emotion.mid` (Ian Taylor), 4/4 B minor 90 bpm | 84 bpm, Bb minor | built, awaiting listen |
| - | Newbie Melody (Tutorial Island) | `newbie_melody.mid` (Ian Taylor), 4/4 C mixolydian 100 bpm | 94 bpm, B mixolydian | built, awaiting listen |
| - | Arabian (Al Kharid) | `arabian.mid` (Ian Taylor), 4/4 B phrygian 150 bpm | 72 bpm half-time, Bb phrygian | built, awaiting listen |
| - | Fanfare (Falador Park) | `fanfare.mid` (Ian Taylor), 4/4 E minor 120 bpm | 116 bpm lo-fi house, Eb minor | built, awaiting listen |
| - | Gnomeball (Match Day) | `gnomeball.mid` (Ian Taylor), 4/4 C major 130 bpm | 124 bpm 90s piano house, Bb major | built, awaiting listen |
| - | Flute Salad (Italo) | `flute_salad.mid` (Ian Taylor), 4/4 G minor 100 bpm | 120 bpm Italo piano house, F# minor | built, awaiting listen |

Source MIDIs are in `~/Documents/Projects/rs-sdk/server/content/songs/`. Copy the one you need into `source/`.

## Listen

https://m4s4t0-v01d.github.io/gielinor-lofi/ (MP3s in `docs/audio/`; re-encode from `build/` after a rebuild).

Not in the repo: the source MIDIs (`source/`) and `assets/sf2/GeneralUser-GS.sf2`, which must be downloaded from
S. Christian Collins' GeneralUser GS page.

## Build

```sh
PY=~/Documents/Projects/intelligent-jungle/.venv/bin/python   # numpy, scipy, soundfile, mido, pyloudnorm
$PY -I build.py harmony --stems         # -> build/harmony/*.wav, *.mp3, report.json, stems/
$PY -I tools/check.py build/harmony/stems 78 4 12 G minor       # key/tuning check over song bars 4-12
```

You also need `fluidsynth` and `ffmpeg` on PATH. Renders are cached in `build/cache` by note content, so a
rebuild after a mix change takes seconds (about 40 s for the FX).

## Workflow for a new track

1. **Read the source.** Run `tools/inspect_midi.py source/x.mid map` for a bar-by-track activity map, and
   `tools/inspect_midi.py source/x.mid <track#> <bar0> <bar1>` for the notes. Write down the sections, the
   melody bars, and the chord per bar.
2. **Choose tempo and key.** Use 70-90 bpm. Varispeed down (`semis=-1..-3`) to taste. Varispeed lowers the
   pitch and the speed together, so the record is rendered fast and lands on the song tempo. A 3/4 source is
   remapped as 3+3+2 (`notes.WALTZ_TO_FOUR`); 4/4 sources pass straight through.
3. **Press records.** Make one `sampler.Record` per group of tracks you want to chop independently: hook,
   melody, strings, and so on.
4. **Arrange.** Use a `Timeline` of song beats. Lay bars with `place_bars`, and use `beats()`, `reverse()` and
   `edge()` for chops, stutters and reversed pickups. Keep the melody intact in at least two sections, which is
   the "recognisable" rule.
5. **Write new parts.** Make a chord chart in A-minor-style names from `parts.VOICINGS` (add voicings as
   needed). Then use `parts.comp` for the Rhodes, `parts.bass` + `parts.sub` for the low end, and write the
   track's own drum groove as a `parts.groove` spec (`boom_bap` is Harmony's). Transpose with
   `notes.shift(..., semis=KEY)`. A 3/4 tune that drags at 3+3+2 can be pressed with `dst_bar=2, curve=None`
   (each source bar = two triplet beats, so the song is in 12/8).
6. **Process.** Copy the chain from `songs/harmony.py`:
   - sample: highpass, 12-bit/26 kHz crush, saturation, filter ride, wow/flutter
   - Rhodes: chorus and saturation
   - drums: crush and tape
   - shared reverb and dotted-eighth tape echo
   - kick-keyed sidechain
   - the scene's ambience (see the rules below)
7. **Level and master.** Set stems with `fx.level` targets (`LEVELS`). `master.master` takes the mix to
   −14 LUFS and ≤ −1 dBTP.
8. **Check.** Read `report.json` (LUFS, true peak, band balance, section RMS) and the output of
   `tools/check.py` (cents offset and in-key energy of each stem).

## Album cohesion rules

- **Tempo:** 72-84 bpm (Newbie Melody runs at 94 so the tune doesn't drag; Arabian is pressed half-time, so its
  tune keeps the source's speed over a 72 bpm beat; Fanfare is a 116 bpm lo-fi house track pressed 1:1 from a
  120 bpm march; Gnomeball is 124 bpm 90s piano house, swing 0.54;
  Flute Salad is 120 bpm Italo piano house sped up from 100, swing 0.52). Swing 0.57-0.60. Snare laid back by about 14 ms.
- **Master:** the same chain on every track: −14 LUFS, −1 dBTP, mono below 140 Hz.
- **Sample treatment:** crush every sample at 12-bit/26.04 kHz. The sample filter tops out around 5 kHz, so the
  album stays dark and warm.
- **Stem levels:** use the same `LEVELS` targets everywhere. The sample is the loudest musical stem (≈ −21 dB
  gated RMS), and the drums sit under it.
- **Ambience:** tape hiss on every track, plus whatever the scene has from `engine/ambience.py`: rain only where
  it's raining (Harmony), surf and gulls at sea. Anything noisy and dense reads as rain: vinyl crackle *and* its
  surface noise, and crushed 16th hats. A dry scene gets no vinyl at all (Book of Spells), and elsewhere keep it
  sparse (`crackle_rate` ≤ 3, ≤ −46 dB).
- **Track shape:** a filtered intro without drums, then the A section with the hook intact, a contrast section
  from the source's own bridge, a turnaround, the hook again with some variation, and an outro that closes the
  filter and lets the last chord ring.
- **Every track has its own drum groove:**
  - 01 Harmony: `boom_bap`, swung 16th hats.
  - 02 Autumn Voyage: a rowing stroke on a room kit: the floor tom pulls on the last 16th into each bar, the kick
    lands on 1, the and of 2 and 3, straight 8th hats, snare on 2 and 4; the turn and C are floor tom, kick and
    one sidestick.
  - 04 Book of Spells: jazz ride (spang-a-lang) over a sidestick, a two-step kick, no hats; the B section is a
    ride bell and one sidestick.
  - 05 Sea Shanty: a 12/8 triplet grid with a stomping kick, snare on 2 and 4 with a clap on 4, a shuffled
    tambourine and a woodblock; the dance is four on the floor.
  - 07 Castle Wars (a deliberate 90s jungle departure, 174 bpm): a two-bar break programmed at ~130 bpm from
    the source's own drum kinds (tambourine, 8th hats, toms, open hat, taiko in 3+3+2) plus a GM kick and snare,
    pitched up 5 semitones and chopped into edits and rolls; half-time roller in the bridge; reese + sub bass.
  - Emotion: a brush kit (40) with a shaker in place of hats (accents on the off-8ths), the kick on 1, a+2 and 3
    (1, e+1, 3, a+3 on odd bars), the brush slap on 2 and 4 dragged 30 ms with a clap on 4, sidestick ghosts;
    the B sections are kick, sidestick and an off-8th shaker.
  - Newbie Melody: a bright kit (25), the kick on 1, a+1 and 3, snare on 2 and 4, a rim on the last 16th of odd
    bars, 8th hats with an open hat on the and of 2; B is kick and sidestick; the bells and C sections switch to
    a triplet grid (kick on 1, the last triplet of 2 and on 3, snare on 2 and 4, triplet hats); the minor turn is half-time with a
    clap on 3 and pedal hats.
  - Arabian: a power kit (16) with a darbuka layer on congas: maqsum in the A sections (doum on 1 and 3, tek on
    e+1, e+2 and 4, ka ghosts), a 3+3+2 malfuf under handclaps in B, a riq on the off-8ths auto-panned across the
    field, a finger-cymbal (triangle) every four bars.
  - Fanfare: a dance kit (26) in four on the floor, a clap on 2 and 4 with a snare under the 4, an open hat on
    every off-beat, shuffled 16th closed hats, rim ghosts; organ stabs on the off-beats pumped 7 dB by the kick.
    B and the calls add a shaker, the C breakdown drops the kick, and the source's march snare (orchestral kit)
    ticks quietly under A2.
  - Gnomeball: an Electronic kit (24) in four on the floor with a pickup kick on the a of 4 into each phrase, a
    909 snare on 2 and 4 with a clap on the 4 (flammed on odd bars), a ride on every off-beat in place of open
    hats, closed hats on the e and a, and the source's own cabasa 16ths; B and A2 add a tambourine on 2 and 4,
    the turn drops the kick. Syncopated bright-piano stabs and an off-beat synth bass, a mild 3 dB duck.
  - Flute Salad: a Standard 3 kit (2) in four on the floor (fill bars drop the 4 for a clap pickup), a clap on
    2 and 4 under a gated-reverb room with a snare on the 4 and a skip clap on the a of 2 in odd bars, straight
    16th closed hats accented on the off-beats, one open hat on the and of 4; B adds a sidestick on the a of 2
    and 4. The source's latin kit (cowbell, bongos, congas, agogo, cabasa) plays on the 808 kit, auto-panned.
    The flute tune doubles as an Italo block-chord piano hook on an electric grand; synth bass 2 plays the
    source's funky bass line.
- **No signature moves twice:** each transition trick belongs to one track. Build transitions from the song's
  own source material where you can, and only add a trick if the song needs it. Taken so far:
  - 01 Harmony: tape stop into A3, reversed pickups into A1, B and the outro, A4 stutter.
  - 02 Autumn Voyage: the source's own pedal turn and key change (C# minor -> A minor), carried over by one big
    gust of wind (`ambience.wind(gusts=...)`).
  - 04 Book of Spells: the source's timpani roll into A3, an A4 vibes beat thrown into the tape echo.
  - 05 Sea Shanty: the source's tubular bell rung as a ship's bell into B, the turn going under water
    (whole-band lowpass dip) and surfacing on a wave crash, an A4 pitch sway (`fx.sway`) like a rolling deck.
  - 07 Castle Wars: the break wound up from its programmed speed to tempo over the 8 bars before drop 2
    (`wind_up`, a rising pitch ramp).
    Its 90s jungle effects are its own too: a dub siren into the drop, breakdown and outro, 8-bit orchestra-hit
    stabs on the bass root in drop 2, an 808 boom under each drop, a pitched snare rush closing each drop, a
    resonant filter peak riding the break in, and a crunchy parallel 'desk' bus on the break.
  - Emotion: the source's own tom pickups into both B sections, and a held breath before B1: the band drops
    for a beat and a half under a granular freeze of the F# chord (`freeze()` in songs/emotion.py).
  - Newbie Melody: the intro and outro play through a tiny speaker (`fx.small_speaker`) that snaps open to full
    range when the beat drops, and the band crossfades back into it as the song ends.
  - Arabian: the break: the band drops out and the source's own drum tracks play alone under an oud taqsim on a
    held drone, with crickets coming up, until a darbuka roll into a crash throws the beat back in for B2.
    Its desert is its own too: a mirage intro and outro (`heat_haze()`: wandering random delays like air over
    hot sand, turning solid when the beat lands), koto qanun runs down the hijaz augmented second into each
    Bm11, distant camel bells (`camel_bells()`), a booming dune hum under the break (`booming_dune()`) and an
    octave-up ney shimmer fed only to the plate.
  - Fanfare: the parade pass: at the first fanfare call the house band stops and the source's own trumpets,
    horn and drums march past the listener from left to right (distance gain, air-absorption lowpass, angle pan,
    a small Doppler bend; `parade_pass()` in songs/fanfare.py) over the park fountain, then the beat slams back.
  - Gnomeball: the game: a synthesized referee's pea whistle (`pea_whistle()`) blows for kick-off into A1, to
    restart play into A2 and for full time; in the breakdown the source's woodblock becomes a dropped ball
    (`bounces()` + `thumps()`), bouncing quicker and quieter while the band returns under a rising highpass and a
    muffled kick opens up.
  - Flute Salad: the salad spinner: in the ocarina break the pads, piano and record play through a rotary
    speaker (`rotary()`: horn and drum rotors with Doppler and two-mic level sweep, the drum lagging) that turns
    slowly, then spins up from 0.8 to 7.5 Hz over the four bars into the drop. Also a dotted-quarter flute echo
    and a quarter ping-pong on the ocarina.

## Licensing notes

- The Harmony composition is © Jagex (composer Ian Taylor). These tracks are fan reinterpretations. Get
  permission before any commercial release.
- GeneralUser GS (`assets/sf2/`) permits use in music productions; see `assets/sf2/LICENSE.txt`.
- All drums come from the soundfont. All ambience is synthesized.
