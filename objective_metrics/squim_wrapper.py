#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
import torchaudio


TARGET_SR = 16000


def load_audio(path: Path):
    wav, sr = torchaudio.load(str(path))

    if wav.ndim == 2:
        wav = wav.mean(dim=0, keepdim=True)

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

    device = torch.device(args.device)

    bundle = torchaudio.pipelines.SQUIM_OBJECTIVE

    model = bundle.get_model().to(device)
    model.eval()

    wav = load_audio(args.wav).to(device)

    with torch.inference_mode():
        scores = model(wav)

    # SQUIM_OBJECTIVE returns:
    # STOI, PESQ, SI-SDR
    stoi, pesq, si_sdr = scores

    result = {
        "stoi": float(
            stoi.detach().cpu().reshape(-1)[0]
        ),
        "pesq": float(
            pesq.detach().cpu().reshape(-1)[0]
        ),
        "si_sdr": float(
            si_sdr.detach().cpu().reshape(-1)[0]
        ),
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()