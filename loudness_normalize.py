#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm


EPS = 1e-12


def db_to_gain(db: float) -> float:
    return 10 ** (db / 20.0)


def peak_dbfs(x: np.ndarray) -> float:
    peak = np.max(np.abs(x))
    return 20 * np.log10(peak + EPS)


def normalize_safe(
    audio: np.ndarray,
    sr: int,
    target_lufs: float,
    peak_limit_dbfs: float,
):
    meter = pyln.Meter(sr)

    loudness = meter.integrated_loudness(audio)

    if not np.isfinite(loudness):
        return audio, loudness, peak_dbfs(audio), peak_dbfs(audio), 0.0, True

    current_peak_db = peak_dbfs(audio)

    desired_gain_db = target_lufs - loudness
    predicted_peak_db = current_peak_db + desired_gain_db

    clipped_by_peak = False

    if predicted_peak_db > peak_limit_dbfs:
        safe_gain_db = peak_limit_dbfs - current_peak_db
        clipped_by_peak = True
    else:
        safe_gain_db = desired_gain_db

    out = audio * db_to_gain(safe_gain_db)

    final_loudness = meter.integrated_loudness(out)
    final_peak_db = peak_dbfs(out)

    return out, loudness, final_loudness, final_peak_db, safe_gain_db, clipped_by_peak


def process_file(infile: Path, outfile: Path, target_lufs: float, peak_limit_dbfs: float):
    audio, sr = sf.read(infile)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float64)

    out, in_lufs, out_lufs, out_peak_db, gain_db, peak_limited = normalize_safe(
        audio,
        sr,
        target_lufs,
        peak_limit_dbfs,
    )

    outfile.parent.mkdir(parents=True, exist_ok=True)
    sf.write(outfile, out, sr, subtype="PCM_16")

    return {
        "file": str(infile),
        "output": str(outfile),
        "input_lufs": in_lufs,
        "output_lufs": out_lufs,
        "output_peak_dbfs": out_peak_db,
        "gain_db": gain_db,
        "peak_limited": peak_limited,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--target_lufs", type=float, default=-20.0)
    parser.add_argument("--peak_limit_dbfs", type=float, default=-1.0)
    parser.add_argument("--report_csv", type=Path, default=Path("loudness_report.csv"))
    args = parser.parse_args()

    wavs = sorted(args.input_root.rglob("*.wav"))

    rows = []

    for wav in tqdm(wavs, desc="Normalizing"):
        rel = wav.relative_to(args.input_root)
        out = args.output_root / rel

        row = process_file(
            wav,
            out,
            args.target_lufs,
            args.peak_limit_dbfs,
        )

        rows.append(row)

    with args.report_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file",
            "output",
            "input_lufs",
            "output_lufs",
            "output_peak_dbfs",
            "gain_db",
            "peak_limited",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote report: {args.report_csv}")


if __name__ == "__main__":
    main()