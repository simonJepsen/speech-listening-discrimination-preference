#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
import torchaudio
import distillmos


TARGET_SR = 16000


def load_audio(path: Path):
    wav, sr = torchaudio.load(str(path))

    # Mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Distill-MOS expects 16 kHz
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(
            wav,
            sr,
            TARGET_SR,
        )

    return wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    parser.add_argument(
        "--device",
        default="cpu",
    )
    args = parser.parse_args()

    if not args.wav.exists():
        raise FileNotFoundError(args.wav)

    device = torch.device(args.device)

    model = distillmos.ConvTransformerSQAModel()
    model.eval()
    model.to(device)

    wav = load_audio(args.wav).to(device)

    with torch.no_grad():
        mos = model(wav)

    mos = float(
        mos.detach()
        .cpu()
        .reshape(-1)[0]
    )

    print(
        json.dumps({
            "mos": mos
        })
    )


if __name__ == "__main__":
    main()