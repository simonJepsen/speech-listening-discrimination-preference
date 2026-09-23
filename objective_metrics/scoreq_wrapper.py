#!/usr/bin/env python3
"""
SCOREQ wrapper for evaluate_all_metrics.py.

Usage:
    python scoreq_wrapper.py /path/to/audio.wav

Final stdout line:
    {"scoreq": <float>}

This uses SCOREQ for natural speech in no-reference mode, which is
appropriate for speech enhancement/restoration outputs. Higher is better.
"""

import argparse
import json
import math
import sys


def finite_or_none(value):
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    args = parser.parse_args()

    try:
        import scoreq

        model = scoreq.Scoreq(
            data_domain="natural",
            mode="nr",
        )

        score = model.predict(
            test_path=args.audio_path,
            ref_path=None,
        )

        print(json.dumps({
            "scoreq": finite_or_none(score),
        }))

    except Exception as exc:
        print(
            f"SCOREQ failed for {args.audio_path}: {exc}",
            file=sys.stderr,
        )
        print(json.dumps({"scoreq": None}))
        sys.exit(1)


if __name__ == "__main__":
    main()
