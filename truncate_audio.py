#!/usr/bin/env python3

import argparse
from pathlib import Path

import soundfile as sf
import numpy as np


def process_file(infile: Path, outfile: Path, duration: float):
    audio, sr = sf.read(infile)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    target_samples = int(duration * sr)

    if len(audio) >= target_samples:
        audio = audio[:target_samples]
    else:
        audio = np.pad(audio, (0, target_samples - len(audio)))

    outfile.parent.mkdir(parents=True, exist_ok=True)
    sf.write(outfile, audio, sr, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=6.0)

    args = parser.parse_args()

    wavs = sorted(args.input_root.rglob("*.wav"))

    print(f"Found {len(wavs)} wav files")

    for wav in wavs:
        rel = wav.relative_to(args.input_root)
        out = args.output_root / rel

        process_file(
            wav,
            out,
            args.duration,
        )

    print("Done.")


if __name__ == "__main__":
    main()