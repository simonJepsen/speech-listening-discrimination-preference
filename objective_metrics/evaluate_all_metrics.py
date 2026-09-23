#!/usr/bin/env python3

"""
Evaluate speech-enhancement systems only on utterances used in the
listening experiment.

Expected structure
------------------

audio_root/
    clean/
        fileid_0.wav
        fileid_1.wav
        ...
    noisy/
        fileid_0.wav
        fileid_1.wav
        ...
    MA_DNS/
        fileid_0.wav
        ...
    MeanFlow_DNS/
        ...
    mpsenet_DNS/
        ...
    storm_DNS/
        ...

selected_utterances.csv
-----------------------
Expected to contain at least:

dataset,utterance_id,relative_path

For example:

DNS,DNS/fileid_9,fileid_9.wav


Metrics
-------
Directly computed:
    PESQ
    ESTOI
    SI-SDR
    SI-SDRi
    CSIG
    CBAK
    COVL

Optional external metrics:
    POLQA
    DNSMOS SIG/BAK/OVRL
    UTMOS
    NISQA MOS
    NISQA Noisiness
    NISQA Coloration
    NISQA Discontinuity
    NISQA Loudness

Optional in-process learned metrics:
    TorchAudio SQUIM Objective:
        squim_stoi
        squim_pesq
        squim_si_sdr
    Distill-MOS:
        distillmos

The external wrapper scripts are expected to print one JSON object
as their final stdout line.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import soundfile as sf

from scipy.linalg import solve_toeplitz, toeplitz
from scipy.signal import resample_poly

# ============================================================
# General helpers
# ============================================================


def normalize_extension(extension: str) -> str:
    extension = extension.strip()

    if not extension.startswith("."):
        extension = "." + extension

    return extension.lower()


def finite_or_nan(value):
    try:
        value = float(value)
    except Exception:
        return np.nan

    return value if np.isfinite(value) else np.nan


def resample_to(
    audio: np.ndarray,
    source_sr: int,
    target_sr: int,
) -> np.ndarray:

    if source_sr == target_sr:
        return audio

    gcd = math.gcd(source_sr, target_sr)

    return resample_poly(
        audio,
        target_sr // gcd,
        source_sr // gcd,
    )


def match_length(*signals: np.ndarray):
    length = min(len(signal) for signal in signals)

    return [signal[:length] for signal in signals]


# ============================================================
# Audio loading
# ============================================================


def load_audio(
    path: Path,
    target_sr: Optional[int] = None,
):
    audio, sr = sf.read(
        path,
        always_2d=False,
    )

    if audio.ndim == 2:
        audio = np.mean(
            audio,
            axis=1,
        )

    audio = np.asarray(
        audio,
        dtype=np.float64,
    )

    if target_sr is not None and sr != target_sr:
        audio = resample_to(
            audio,
            sr,
            target_sr,
        )

        sr = target_sr

    return audio, sr


# ============================================================
# Selected listening-test utterances
# ============================================================


def load_selected_utterances(
    csv_path: Path,
    dataset_filter: Optional[str] = None,
) -> set[str]:

    selected = set()

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no CSV header.")

        if "relative_path" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain a " "'relative_path' column.")

        for row in reader:

            if dataset_filter is not None:
                dataset = row.get("dataset", "").strip()

                if dataset != dataset_filter:
                    continue

            relative_path = row["relative_path"].strip()

            if relative_path:
                selected.add(relative_path)

    if not selected:
        raise ValueError("No listening-test utterances were found " f"in {csv_path}.")

    return selected


# ============================================================
# File discovery
# ============================================================


def discover_relative_files(
    root: Path,
    extension: str,
) -> Dict[str, Path]:

    if not root.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    result = {}

    for path in root.rglob("*"):

        if path.is_file() and path.suffix.lower() == extension:
            relative = path.relative_to(root).as_posix()

            result[relative] = path

    return result


# ============================================================
# SI-SDR
# ============================================================


def si_sdr(
    reference: np.ndarray,
    estimate: np.ndarray,
    eps: float = 1e-8,
) -> float:

    reference, estimate = match_length(
        reference,
        estimate,
    )

    reference = reference - np.mean(reference)

    estimate = estimate - np.mean(estimate)

    denominator = np.sum(reference**2) + eps

    scale = np.sum(estimate * reference) / denominator

    target = scale * reference
    residual = estimate - target

    signal_power = np.sum(target**2) + eps

    error_power = np.sum(residual**2) + eps

    return float(10.0 * np.log10(signal_power / error_power))


# ============================================================
# PESQ
# ============================================================


def compute_pesq(
    clean: np.ndarray,
    enhanced: np.ndarray,
    sr: int,
) -> float:

    try:
        from pesq import pesq

        target_sr = 16000 if sr != 8000 else 8000

        if sr != target_sr:

            clean = resample_to(
                clean,
                sr,
                target_sr,
            )

            enhanced = resample_to(
                enhanced,
                sr,
                target_sr,
            )

            sr = target_sr

        clean, enhanced = match_length(
            clean,
            enhanced,
        )

        mode = "wb" if sr == 16000 else "nb"

        return finite_or_nan(
            pesq(
                sr,
                clean.astype(np.float32),
                enhanced.astype(np.float32),
                mode,
            )
        )

    except Exception as exc:

        print(
            f"PESQ failed: {exc}",
            file=sys.stderr,
        )

        return np.nan


# ============================================================
# ESTOI
# ============================================================


def compute_estoi(
    clean: np.ndarray,
    enhanced: np.ndarray,
    sr: int,
) -> float:

    try:
        from pystoi import stoi

        clean, enhanced = match_length(
            clean,
            enhanced,
        )

        return finite_or_nan(
            stoi(
                clean,
                enhanced,
                sr,
                extended=True,
            )
        )

    except Exception as exc:

        print(
            f"ESTOI failed: {exc}",
            file=sys.stderr,
        )

        return np.nan


# ============================================================
# Legacy composite metrics
# ============================================================


def next_power_of_two(x: int) -> int:
    return int(np.ceil(np.log2(abs(x))))


def weighted_spectral_slope(
    clean: np.ndarray,
    enhanced: np.ndarray,
    sr: int,
) -> float:

    clean, enhanced = match_length(
        clean,
        enhanced,
    )

    win_length = int(round(0.030 * sr))

    skip_rate = win_length // 4

    n_fft = 2 ** next_power_of_two(2 * win_length)

    num_crit = 25
    max_freq = sr / 2

    center_freq = np.array(
        [
            50.0000,
            120.000,
            190.000,
            260.000,
            330.000,
            400.000,
            470.000,
            540.000,
            617.372,
            703.378,
            798.717,
            904.128,
            1020.38,
            1148.30,
            1288.72,
            1442.54,
            1610.70,
            1794.16,
            1994.18,
            2212.08,
            2449.18,
            2707.18,
            2988.35,
            3294.73,
            3628.49,
        ]
    )

    bandwidth = np.array(
        [
            70.0000,
            70.0000,
            70.0000,
            70.0000,
            70.0000,
            70.0000,
            70.0000,
            77.3724,
            86.0056,
            95.3398,
            105.411,
            116.256,
            127.914,
            140.423,
            153.823,
            168.154,
            183.457,
            199.776,
            217.153,
            235.631,
            255.126,
            276.037,
            298.126,
            321.465,
            346.136,
        ]
    )

    if max_freq != 4000:
        scale = max_freq / 4000

        center_freq *= scale
        bandwidth *= scale

    bw_min = bandwidth[0]

    min_factor = np.exp(-30.0 / (2.0 * 2.303))

    crit_filter = np.zeros(
        (
            num_crit,
            n_fft,
        )
    )

    frequencies = np.arange(n_fft)

    for index in range(num_crit):

        f0 = center_freq[index] / max_freq * (n_fft / 2)

        bw = bandwidth[index] / max_freq * (n_fft / 2)

        norm_factor = np.log(bw_min) - np.log(bandwidth[index])

        filt = np.exp(-11 * ((frequencies - np.floor(f0)) / bw) ** 2 + norm_factor)

        filt *= filt > min_factor

        crit_filter[index] = filt

    num_frames = int(len(clean) / skip_rate - win_length / skip_rate)

    if num_frames <= 0:
        return np.nan

    window = np.hanning(win_length)

    distortions = []

    for frame_index in range(num_frames):

        start = frame_index * skip_rate

        stop = start + win_length

        clean_frame = clean[start:stop] * window

        enhanced_frame = enhanced[start:stop] * window

        clean_spec = (
            np.abs(
                np.fft.fft(
                    clean_frame,
                    n_fft,
                )
            )
            ** 2
        )

        enhanced_spec = (
            np.abs(
                np.fft.fft(
                    enhanced_frame,
                    n_fft,
                )
            )
            ** 2
        )

        clean_energy = np.maximum(
            crit_filter @ clean_spec,
            1e-10,
        )

        enhanced_energy = np.maximum(
            crit_filter @ enhanced_spec,
            1e-10,
        )

        clean_db = 10 * np.log10(clean_energy)

        enhanced_db = 10 * np.log10(enhanced_energy)

        clean_slope = np.diff(clean_db)

        enhanced_slope = np.diff(enhanced_db)

        slope_error = clean_slope - enhanced_slope

        distortions.append(float(np.mean(slope_error**2)))

    distortions = np.sort(distortions)

    keep = max(
        1,
        int(round(0.95 * len(distortions))),
    )

    return float(np.mean(distortions[:keep]))


def log_likelihood_ratio(
    clean: np.ndarray,
    enhanced: np.ndarray,
    sr: int,
) -> float:

    clean, enhanced = match_length(
        clean,
        enhanced,
    )

    win_length = int(round(0.030 * sr))

    skip_rate = win_length // 4

    order = 10 if sr < 10000 else 16

    num_frames = int(len(clean) / skip_rate - win_length / skip_rate)

    if num_frames <= 0:
        return np.nan

    window = np.hanning(win_length)

    scores = []

    for frame_index in range(num_frames):

        start = frame_index * skip_rate

        stop = start + win_length

        clean_frame = clean[start:stop] * window

        enhanced_frame = enhanced[start:stop] * window

        clean_corr = np.correlate(
            clean_frame,
            clean_frame,
            mode="full",
        )

        enhanced_corr = np.correlate(
            enhanced_frame,
            enhanced_frame,
            mode="full",
        )

        clean_mid = len(clean_corr) // 2

        enhanced_mid = len(enhanced_corr) // 2

        clean_corr = clean_corr[clean_mid : clean_mid + order + 1]

        enhanced_corr = enhanced_corr[enhanced_mid : enhanced_mid + order + 1]

        try:

            clean_lpc = np.r_[
                1.0,
                -solve_toeplitz(
                    (
                        clean_corr[:-1],
                        clean_corr[:-1],
                    ),
                    clean_corr[1:],
                ),
            ]

            enhanced_lpc = np.r_[
                1.0,
                -solve_toeplitz(
                    (
                        enhanced_corr[:-1],
                        enhanced_corr[:-1],
                    ),
                    enhanced_corr[1:],
                ),
            ]

            clean_matrix = toeplitz(clean_corr)

            numerator = enhanced_lpc @ clean_matrix @ enhanced_lpc.T

            denominator = clean_lpc @ clean_matrix @ clean_lpc.T

            score = np.log((numerator + 1e-10) / (denominator + 1e-10))

            if np.isfinite(score):
                scores.append(float(score))

        except Exception:
            continue

    if not scores:
        return np.nan

    scores = np.sort(scores)

    keep = max(
        1,
        int(round(0.95 * len(scores))),
    )

    return float(np.mean(scores[:keep]))


def segmental_snr(
    clean: np.ndarray,
    enhanced: np.ndarray,
    sr: int,
) -> float:

    clean, enhanced = match_length(
        clean,
        enhanced,
    )

    frame_length = int(round(0.030 * sr))

    if frame_length <= 0:
        return np.nan

    scores = []

    for start in range(
        0,
        len(clean) - frame_length + 1,
        frame_length,
    ):

        clean_frame = clean[start : start + frame_length]

        enhanced_frame = enhanced[start : start + frame_length]

        clean_power = np.sum(clean_frame**2) + 1e-10

        error_power = np.sum((clean_frame - enhanced_frame) ** 2) + 1e-10

        value = 10 * np.log10(clean_power / error_power)

        value = np.clip(
            value,
            -10,
            35,
        )

        scores.append(float(value))

    if not scores:
        return np.nan

    return float(np.mean(scores))


def compute_composite_metrics(
    clean: np.ndarray,
    enhanced: np.ndarray,
    sr: int,
    pesq_score: float,
):

    try:

        if not np.isfinite(pesq_score):
            return (
                np.nan,
                np.nan,
                np.nan,
            )

        wss_score = weighted_spectral_slope(
            clean,
            enhanced,
            sr,
        )

        llr_score = log_likelihood_ratio(
            clean,
            enhanced,
            sr,
        )

        segsnr_score = segmental_snr(
            clean,
            enhanced,
            sr,
        )

        csig = 3.093 - 1.029 * llr_score + 0.603 * pesq_score - 0.009 * wss_score

        cbak = 1.634 + 0.478 * pesq_score - 0.007 * wss_score + 0.063 * segsnr_score

        covl = 1.594 + 0.805 * pesq_score - 0.512 * llr_score - 0.007 * wss_score

        return (
            float(
                np.clip(
                    csig,
                    1,
                    5,
                )
            ),
            float(
                np.clip(
                    cbak,
                    1,
                    5,
                )
            ),
            float(
                np.clip(
                    covl,
                    1,
                    5,
                )
            ),
        )

    except Exception as exc:

        print(
            "Composite metrics failed: " f"{exc}",
            file=sys.stderr,
        )

        return (
            np.nan,
            np.nan,
            np.nan,
        )


# ============================================================
# External JSON wrappers
# ============================================================


def run_json_wrapper(
    wrapper: Path,
    audio_path: Path,
):

    command = [
        sys.executable,
        str(wrapper),
        str(audio_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    if not lines:
        raise RuntimeError("Metric wrapper produced " "no stdout.")

    return json.loads(lines[-1])


def run_pair_json_wrapper(wrapper: Path, audio_a_path: Path, audio_b_path: Path):
    command = [sys.executable, str(wrapper), str(audio_a_path), str(audio_b_path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Pairwise metric wrapper produced no stdout.")
    return json.loads(lines[-1])


def compute_scoreq(
    audio_path: Path,
    wrapper: Optional[Path],
) -> float:
    if wrapper is None:
        return np.nan

    try:
        values = run_json_wrapper(
            wrapper,
            audio_path,
        )

        score = finite_or_nan(
            values.get("scoreq")
        )

        return score

    except Exception as exc:
        print(
            f"SCOREQ failed for {audio_path}: {exc}",
            file=sys.stderr,
        )
        return np.nan


def compute_urgentpk(audio_a_path: Path, audio_b_path: Path, wrapper: Optional[Path]):
    result = {"prob_a_better": np.nan, "prob_b_better": np.nan,
              "binary_a_better": np.nan, "mos_a": np.nan, "mos_b": np.nan}
    if wrapper is None:
        return result
    try:
        values = run_pair_json_wrapper(wrapper, audio_a_path, audio_b_path)
        for key in result:
            result[key] = finite_or_nan(values.get(key))
    except Exception as exc:
        print(f"URGENT-PK failed for {audio_a_path} vs {audio_b_path}: {exc}", file=sys.stderr)
    return result


# ============================================================
# DNSMOS
# ============================================================


def compute_dnsmos(
    audio_path: Path,
    wrapper: Optional[Path],
):

    result = {
        "sig": np.nan,
        "bak": np.nan,
        "ovrl": np.nan,
    }

    if wrapper is None:
        return result

    try:

        values = run_json_wrapper(
            wrapper,
            audio_path,
        )

        result["sig"] = finite_or_nan(values.get("sig"))

        result["bak"] = finite_or_nan(values.get("bak"))

        result["ovrl"] = finite_or_nan(values.get("ovrl"))

    except Exception as exc:

        print(
            f"DNSMOS failed for " f"{audio_path}: {exc}",
            file=sys.stderr,
        )

    return result


# ============================================================
# NISQA
# ============================================================


def compute_nisqa(
    audio_path: Path,
    wrapper: Optional[Path],
):

    result = {
        "mos": np.nan,
        "noisiness": np.nan,
        "coloration": np.nan,
        "discontinuity": np.nan,
        "loudness": np.nan,
    }

    if wrapper is None:
        return result

    try:

        values = run_json_wrapper(
            wrapper,
            audio_path,
        )

        for key in result:
            result[key] = finite_or_nan(values.get(key))

    except Exception as exc:

        print(
            f"NISQA failed for " f"{audio_path}: {exc}",
            file=sys.stderr,
        )

    return result


# ============================================================
# UTMOS
# ============================================================


def compute_utmos(
    audio_path: Path,
    wrapper: Optional[Path],
) -> float:

    if wrapper is None:
        return np.nan

    try:

        values = run_json_wrapper(
            wrapper,
            audio_path,
        )

        return finite_or_nan(values.get("mos"))

    except Exception as exc:

        print(
            f"UTMOS failed for " f"{audio_path}: {exc}",
            file=sys.stderr,
        )

        return np.nan


# ============================================================
# POLQA
# ============================================================


def compute_polqa(
    reference_path: Path,
    degraded_path: Path,
    executable: Optional[Path],
) -> float:

    if executable is None:
        return np.nan

    try:

        command = [
            str(executable),
            str(reference_path),
            str(degraded_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )

        output = result.stdout + "\n" + result.stderr

        patterns = [
            r"MOS[-_\s]?LQO[^0-9+-]*" r"([-+]?\d+(?:\.\d+)?)",
            r"POLQA[^0-9+-]*" r"([-+]?\d+(?:\.\d+)?)",
            r"MOS[^0-9+-]*" r"([-+]?\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                output,
                flags=re.IGNORECASE,
            )

            if match:
                return float(match.group(1))

        raise RuntimeError("Could not parse POLQA " "MOS-LQO from output:\n" + output)

    except Exception as exc:

        print(
            f"POLQA failed for " f"{degraded_path}: {exc}",
            file=sys.stderr,
        )

        return np.nan



# ============================================================
# TorchAudio SQUIM + Distill-MOS + SCOREQ
# ============================================================

class LearnedMetricModels:
    """
    Load SQUIM Objective and Distill-MOS once and reuse the models
    across the complete evaluation run.

    SQUIM Objective outputs reference-free estimates of:
        STOI, PESQ, SI-SDR

    Distill-MOS expects mono 16-kHz waveforms of shape:
        [batch_size, signal_length]
    """

    def __init__(
        self,
        device: str = "cpu",
        enable_squim: bool = False,
        enable_distillmos: bool = False,
    ):
        self.device_name = device
        self.device = None

        self.torch = None
        self.torchaudio = None

        self.squim_model = None
        self.squim_sample_rate = 16000

        self.distillmos_model = None
        self.scoreq_model = None
        self.scoreq_sample_rate = 16000

        if not (enable_squim or enable_distillmos):
            return

        try:
            import torch
            import torchaudio

            self.torch = torch
            self.torchaudio = torchaudio
            self.device = torch.device(device)

        except Exception as exc:
            print(
                f"Could not import torch/torchaudio: {exc}",
                file=sys.stderr,
            )
            return

        if enable_squim:
            self._load_squim()

        if enable_distillmos:
            self._load_distillmos()

        


    def _load_squim(self):
        try:
            bundle = self.torchaudio.pipelines.SQUIM_OBJECTIVE

            self.squim_sample_rate = int(
                getattr(bundle, "sample_rate", 16000)
            )

            self.squim_model = (
                bundle
                .get_model()
                .to(self.device)
                .eval()
            )

            print(
                f"Loaded TorchAudio SQUIM Objective "
                f"on {self.device}."
            )

        except Exception as exc:
            print(
                f"Could not load TorchAudio SQUIM Objective: {exc}",
                file=sys.stderr,
            )
            self.squim_model = None


    def _load_distillmos(self):
        try:
            import distillmos

            self.distillmos_model = (
                distillmos
                .ConvTransformerSQAModel()
                .to(self.device)
                .eval()
            )

            print(
                f"Loaded Distill-MOS on {self.device}."
            )

        except Exception as exc:
            print(
                f"Could not load Distill-MOS: {exc}",
                file=sys.stderr,
            )
            self.distillmos_model = None


    def _waveform_tensor(
        self,
        audio: np.ndarray,
        sr: int,
        target_sr: int,
    ):
        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if sr != target_sr:
            audio = resample_to(
                audio,
                sr,
                target_sr,
            ).astype(np.float32)

        return (
            self.torch
            .from_numpy(audio)
            .unsqueeze(0)
            .to(self.device)
        )


    def compute_squim(
        self,
        audio: np.ndarray,
        sr: int,
    ):
        result = {
            "stoi": np.nan,
            "pesq": np.nan,
            "si_sdr": np.nan,
        }

        if self.squim_model is None:
            return result

        try:
            wav = self._waveform_tensor(
                audio,
                sr,
                self.squim_sample_rate,
            )

            with self.torch.inference_mode():
                scores = self.squim_model(wav)

            # SQUIM_OBJECTIVE returns:
            # STOI, PESQ, SI-SDR
            stoi_score, pesq_score, si_sdr_score = scores

            result["stoi"] = finite_or_nan(
                stoi_score
                .detach()
                .cpu()
                .reshape(-1)[0]
                .item()
            )

            result["pesq"] = finite_or_nan(
                pesq_score
                .detach()
                .cpu()
                .reshape(-1)[0]
                .item()
            )

            result["si_sdr"] = finite_or_nan(
                si_sdr_score
                .detach()
                .cpu()
                .reshape(-1)[0]
                .item()
            )

        except Exception as exc:
            print(
                f"SQUIM inference failed: {exc}",
                file=sys.stderr,
            )

        return result


    def compute_distillmos(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> float:

        if self.distillmos_model is None:
            return np.nan

        try:
            wav = self._waveform_tensor(
                audio,
                sr,
                16000,
            )

            with self.torch.inference_mode():
                mos = self.distillmos_model(wav)

            return finite_or_nan(
                mos
                .detach()
                .cpu()
                .reshape(-1)[0]
                .item()
            )

        except Exception as exc:
            print(
                f"Distill-MOS inference failed: {exc}",
                file=sys.stderr,
            )

            return np.nan



    def _load_scoreq(self):
        """Load SCOREQ once and reuse it for the complete evaluation."""
        try:
            from scoreq_pytorch import SCOREQScoreTorch

            self.scoreq_model = SCOREQScoreTorch(
                data_domain="natural",
                mode="nr",
                device=self.device_name,
            )

            print(
                f"Loaded SCOREQ (natural, NR) on {self.device}."
            )

        except Exception as exc:
            print(
                f"Could not load SCOREQ: {exc}",
                file=sys.stderr,
            )
            self.scoreq_model = None


    def compute_scoreq(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> float:
        """
        Compute natural-speech no-reference SCOREQ.

        SCOREQ-PyTorch expects mono 16-kHz audio. Higher is better
        in natural-speech NR mode.
        """
        if self.scoreq_model is None:
            return np.nan

        try:
            wav = self._waveform_tensor(
                audio,
                sr,
                self.scoreq_sample_rate,
            )

            with self.torch.inference_mode():
                score = self.scoreq_model.score(
                    wav,
                    None,
                )

            return finite_or_nan(
                score.detach().cpu().reshape(-1)[0].item()
            )

        except Exception as exc:
            print(
                f"SCOREQ inference failed: {exc}",
                file=sys.stderr,
            )
            return np.nan


# ============================================================
# CSV writing
# ============================================================


def write_results(
    path: Path,
    rows: Iterable[dict],
):

    rows = list(rows)

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Argument parser
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio_root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--selected_utterances_csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=("Optional dataset filter for " "selected_utterances.csv, " "e.g. DNS."),
    )

    parser.add_argument(
        "--clean_dir",
        type=str,
        default="clean",
    )

    parser.add_argument(
        "--noisy_dir",
        type=str,
        default="noisy",
    )

    parser.add_argument(
        "--systems",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--include_noisy",
        action="store_true",
        help=("Also evaluate the noisy input " "as a condition."),
    )

    parser.add_argument(
        "--include_clean",
        action="store_true",
        help=(
            "Also evaluate clean as a condition. "
            "Intrusive metrics versus itself may "
            "be degenerate."
        ),
    )

    parser.add_argument(
        "--extension",
        type=str,
        default=".wav",
    )

    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("metrics_selected_utterances.csv"),
    )

    parser.add_argument(
        "--dnsmos_wrapper",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--nisqa_wrapper",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--utmos_wrapper",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--polqa_bin",
        type=Path,
        default=None,
    )

    parser.add_argument("--scoreq_wrapper", type=Path, default=None)
    parser.add_argument("--urgentpk_wrapper", type=Path, default=None)
    parser.add_argument(
        "--pair_output_csv", type=Path, default=None,
        help="Optional CSV with one row per utterance/system pair, metric deltas, and URGENT-PK."
    )

    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help=(
            "Skip selected utterances that are "
            "missing from a condition instead of "
            "raising an error."
        ),
    )

    parser.add_argument(
        "--squim",
        action="store_true",
        help="Compute TorchAudio SQUIM Objective metrics.",
    )

    parser.add_argument(
        "--distillmos",
        action="store_true",
        help="Compute Distill-MOS.",
    )

    parser.add_argument(
        "--scoreq",
        action="store_true",
        help="Compute SCOREQ (natural-speech, no-reference mode).",
    )

    parser.add_argument(
        "--learned_metric_device",
        type=str,
        default="cpu",
        help=(
            "Device for SQUIM and Distill-MOS, "
            "for example cpu, cuda, or mps."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================


def main():

    args = parse_args()

    extension = normalize_extension(args.extension)

    selected = load_selected_utterances(
        args.selected_utterances_csv,
        dataset_filter=args.dataset,
    )

    print("\nListening-test utterances:" f" {len(selected)}")

    clean_root = args.audio_root / args.clean_dir

    noisy_root = args.audio_root / args.noisy_dir

    clean_files = discover_relative_files(
        clean_root,
        extension,
    )

    noisy_files = discover_relative_files(
        noisy_root,
        extension,
    )

    missing_clean = selected - set(clean_files)

    missing_noisy = selected - set(noisy_files)

    if missing_clean:
        raise RuntimeError(
            f"Clean directory is missing "
            f"{len(missing_clean)} "
            "listening-test utterances."
        )

    if missing_noisy:
        raise RuntimeError(
            f"Noisy directory is missing "
            f"{len(missing_noisy)} "
            "listening-test utterances."
        )

    conditions = list(args.systems)

    if args.include_noisy:
        conditions = [
            args.noisy_dir,
            *conditions,
        ]

    if args.include_clean:
        conditions = [
            args.clean_dir,
            *conditions,
        ]

    learned_metrics = LearnedMetricModels(
        device=args.learned_metric_device,
        enable_squim=args.squim,
        enable_distillmos=args.distillmos,
    )

    rows = []

    for condition in conditions:

        condition_root = args.audio_root / condition

        condition_files = discover_relative_files(
            condition_root,
            extension,
        )

        missing = selected - set(condition_files)

        if missing:

            print(f"\n{condition} is missing " f"{len(missing)} selected files.")

            for item in sorted(missing):
                print(f"  MISSING: {item}")

            if not args.allow_missing:
                raise RuntimeError(
                    f"{condition} does not contain " "all listening-test utterances."
                )

        available = sorted(selected & set(condition_files))

        print(f"\n================================")

        print(f"Condition: {condition}")

        print(f"Selected files available: " f"{len(available)} / " f"{len(selected)}")

        print(f"================================")

        for file_index, relative_path in enumerate(
            available,
            start=1,
        ):

            print(
                f"[{condition}] "
                f"{file_index:4d}/"
                f"{len(available):4d} "
                f"{relative_path}"
            )

            clean_path = clean_files[relative_path]

            noisy_path = noisy_files[relative_path]

            condition_path = condition_files[relative_path]

            clean, sr = load_audio(clean_path)

            noisy, _ = load_audio(
                noisy_path,
                target_sr=sr,
            )

            enhanced, _ = load_audio(
                condition_path,
                target_sr=sr,
            )

            clean_eval, noisy_eval, enhanced_eval = match_length(
                clean,
                noisy,
                enhanced,
            )

            # ------------------------------------
            # Reference-based metrics
            # ------------------------------------

            pesq_score = compute_pesq(
                clean_eval,
                enhanced_eval,
                sr,
            )

            estoi_score = compute_estoi(
                clean_eval,
                enhanced_eval,
                sr,
            )

            enhanced_si_sdr = si_sdr(
                clean_eval,
                enhanced_eval,
            )

            noisy_si_sdr = si_sdr(
                clean_eval,
                noisy_eval,
            )

            si_sdri = enhanced_si_sdr - noisy_si_sdr

            csig, cbak, covl = compute_composite_metrics(
                clean_eval,
                enhanced_eval,
                sr,
                pesq_score,
            )

            # ------------------------------------
            # Non-intrusive metrics
            # ------------------------------------

            dnsmos = compute_dnsmos(
                condition_path,
                args.dnsmos_wrapper,
            )

            nisqa = compute_nisqa(
                condition_path,
                args.nisqa_wrapper,
            )

            utmos = compute_utmos(
                condition_path,
                args.utmos_wrapper,
            )

            scoreq_score = compute_scoreq(
                condition_path,
                args.scoreq_wrapper,
            )

            # ------------------------------------
            # POLQA
            # ------------------------------------

            polqa = compute_polqa(
                clean_path,
                condition_path,
                args.polqa_bin,
            )

            # ------------------------------------
            # TorchAudio SQUIM + Distill-MOS + SCOREQ
            # ------------------------------------

            squim = learned_metrics.compute_squim(
                enhanced_eval,
                sr,
            )

            distillmos_score = (
                learned_metrics.compute_distillmos(
                    enhanced_eval,
                    sr,
                )
            )

            # ------------------------------------
            # Save row
            # ------------------------------------

            utterance_id = Path(relative_path).with_suffix("").as_posix()

            row = {
                "dataset": args.dataset or "",
                "utterance_id": utterance_id,
                "relative_path": relative_path,
                "condition": condition,
                "clean_path": str(clean_path),
                "noisy_path": str(noisy_path),
                "condition_path": str(condition_path),
                "sample_rate": sr,
                "pesq": pesq_score,
                "estoi": estoi_score,
                "si_sdr": enhanced_si_sdr,
                "noisy_si_sdr": noisy_si_sdr,
                "si_sdri": si_sdri,
                "csig": csig,
                "cbak": cbak,
                "covl": covl,
                "polqa": polqa,
                "dnsmos_sig": dnsmos["sig"],
                "dnsmos_bak": dnsmos["bak"],
                "dnsmos_ovrl": dnsmos["ovrl"],
                "utmos": utmos,
                "scoreq": scoreq_score,
                "distillmos": distillmos_score,
                "squim_stoi": squim["stoi"],
                "squim_pesq": squim["pesq"],
                "squim_si_sdr": squim["si_sdr"],
                "nisqa_mos": nisqa["mos"],
                "nisqa_noisiness": nisqa["noisiness"],
                "nisqa_coloration": nisqa["coloration"],
                "nisqa_discontinuity": nisqa["discontinuity"],
                "nisqa_loudness": nisqa["loudness"],
            }

            rows.append(row)

            # Write after every file so a long job
            # can be interrupted without losing all
            # completed evaluations.
            write_results(
                args.output_csv,
                rows,
            )

    # ========================================================
    # Pairwise metric table
    # ========================================================
    if args.pair_output_csv is not None:
        from itertools import combinations

        pair_metrics = [
            "pesq", "estoi", "si_sdr", "si_sdri", "csig", "cbak", "covl",
            "polqa", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "utmos",
            "scoreq", "distillmos", "squim_stoi", "squim_pesq", "squim_si_sdr",
            "nisqa_mos", "nisqa_noisiness", "nisqa_coloration",
            "nisqa_discontinuity", "nisqa_loudness",
        ]

        by_utterance = {}
        for result_row in rows:
            by_utterance.setdefault(result_row["utterance_id"], {})[result_row["condition"]] = result_row

        pair_rows = []
        for utterance_id in sorted(by_utterance):
            condition_rows = by_utterance[utterance_id]
            available_conditions = [c for c in conditions if c in condition_rows]

            for system_a, system_b in combinations(available_conditions, 2):
                row_a = condition_rows[system_a]
                row_b = condition_rows[system_b]
                pair_row = {
                    "dataset": args.dataset or "",
                    "utterance_id": utterance_id,
                    "relative_path": row_a["relative_path"],
                    "system_a": system_a,
                    "system_b": system_b,
                    "path_a": row_a["condition_path"],
                    "path_b": row_b["condition_path"],
                }

                for metric in pair_metrics:
                    value_a = finite_or_nan(row_a.get(metric))
                    value_b = finite_or_nan(row_b.get(metric))
                    pair_row[f"{metric}_a"] = value_a
                    pair_row[f"{metric}_b"] = value_b
                    if np.isfinite(value_a) and np.isfinite(value_b):
                        delta = value_a - value_b
                        pair_row[f"{metric}_delta"] = delta
                        pair_row[f"{metric}_abs_delta"] = abs(delta)
                    else:
                        pair_row[f"{metric}_delta"] = np.nan
                        pair_row[f"{metric}_abs_delta"] = np.nan

                urgentpk = compute_urgentpk(
                    Path(row_a["condition_path"]),
                    Path(row_b["condition_path"]),
                    args.urgentpk_wrapper,
                )
                pair_row["urgentpk_prob_a_better"] = urgentpk["prob_a_better"]
                pair_row["urgentpk_prob_b_better"] = urgentpk["prob_b_better"]
                pair_row["urgentpk_binary_a_better"] = urgentpk["binary_a_better"]
                pair_row["urgentpk_mos_a"] = urgentpk["mos_a"]
                pair_row["urgentpk_mos_b"] = urgentpk["mos_b"]
                pair_rows.append(pair_row)
                write_results(args.pair_output_csv, pair_rows)

        print(f"\nPairwise rows written: {len(pair_rows)}")
        print(f"Pairwise output: {args.pair_output_csv.resolve()}")

    print("\n================================")

    print("Evaluation complete.")

    print(f"Rows written: " f"{len(rows)}")

    print(f"Output: " f"{args.output_csv.resolve()}")


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )

        sys.exit(1)
