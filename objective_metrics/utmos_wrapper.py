#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    args = parser.parse_args()

    if not args.wav.exists():
        raise FileNotFoundError(args.wav)

    import utmosv2

    model = utmosv2.create_model(
        pretrained=True
    )

    mos = model.predict(
        input_path=str(args.wav.resolve()),
        device="cpu",
    )

    if hasattr(mos, "detach"):
        mos = (
            mos
            .detach()
            .cpu()
            .numpy()
        )

    mos = float(
        np.asarray(mos)
        .reshape(-1)[0]
    )

    print(
        json.dumps({
            "mos": mos
        })
    )


if __name__ == "__main__":
    main()