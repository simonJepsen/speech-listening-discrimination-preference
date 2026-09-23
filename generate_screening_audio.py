#!/usr/bin/env python3

"""
Generate volume-calibration and headphone-screening WAV files.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def tone(sr, duration, frequency, amplitude):
    n = int(sr * duration)
    t = np.arange(n) / sr

    envelope = np.ones(n)
    fade = min(int(0.05 * sr), n // 2)
    ramp = np.linspace(0.0, 1.0, fade)
    envelope[:fade] = ramp
    envelope[-fade:] = ramp[::-1]

    return amplitude * np.sin(
        2 * np.pi * frequency * t
    ) * envelope


def stereo_in_phase(signal):
    return np.column_stack([signal, signal])


def stereo_antiphase(signal):
    return np.column_stack([signal, -signal])


def write(path, audio, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("screening")
    )
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--quiet_gain_db", type=float, default=-6.0)

    args = parser.parse_args()

    calibration = tone(
        args.sr,
        5.0,
        frequency=440.0,
        amplitude=0.035
    )

    write(
        args.output_dir / "calibration.wav",
        stereo_in_phase(calibration),
        args.sr
    )

    frequencies = [300.0, 500.0, 700.0]
    quiet_positions = [2, 3, 1]
    antiphase_positions = [3, 1, 2]

    quiet_gain = 10 ** (args.quiet_gain_db / 20.0)
    full_amplitude = 0.055

    for trial_index, (
        frequency,
        quiet_position,
        antiphase_position
    ) in enumerate(
        zip(
            frequencies,
            quiet_positions,
            antiphase_positions
        ),
        start=1
    ):
        full = tone(
            args.sr,
            args.duration,
            frequency,
            full_amplitude
        )

        quiet = tone(
            args.sr,
            args.duration,
            frequency,
            full_amplitude * quiet_gain
        )

        for position, letter in enumerate(
            ("a", "b", "c"),
            start=1
        ):
            if position == quiet_position:
                audio = stereo_in_phase(quiet)
            elif position == antiphase_position:
                audio = stereo_antiphase(full)
            else:
                audio = stereo_in_phase(full)

            write(
                args.output_dir /
                f"hp_{trial_index}_{letter}.wav",
                audio,
                args.sr
            )

    print(f"Wrote screening files to {args.output_dir}")


if __name__ == "__main__":
    main()
