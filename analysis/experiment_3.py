#!/usr/bin/env python3
"""
Experiment 3: objective metric differences versus perceptual responses.

Inputs
------
1) Triangle utterance-pair cells from Experiment 1.
2) Preference utterance-pair cells from Experiment 2.
3) Objective metric system-pair CSV.

Analyses
--------
A. Objective metric separation versus discrimination
       D^(m) = |Delta m|
       Spearman SRCC with utterance-cluster bootstrap CI
       P_D(D) = 1/3 + 2/3 * sigmoid(alpha + beta D)
       Binomial deviance goodness-of-fit
       Likelihood-ratio test versus intercept-only model

B. Discrimination versus preference decisiveness
       Spearman SRCC with utterance-cluster bootstrap CI

C. Objective metric separation versus preference decisiveness
       Spearman SRCC with utterance-cluster bootstrap CI
       P_dec(D) = sigmoid(alpha + beta D)

D. Objective metric direction versus signed preference direction
       Spearman SRCC with utterance-cluster bootstrap CI
       E[Y_pref | Delta] = 2*sigmoid(alpha + beta Delta) - 1

Reference-based metrics exclude any comparison containing Clean from all
metric-specific SRCCs, fits, and figures. Non-intrusive metrics retain all
listening-test pairs.

Plot formatting intentionally matches Experiment 1:
    * ICASSP full two-column width: 6.89 in
    * 3.55 in figure height
    * 10 pt Times-like LaTeX text (newtxtext/newtxmath)
    * strict black/white plotting
    * full rectangular axes box
    * subtle horizontal grid
    * open empirical markers and solid black fitted curves
    * PDF plus 600-dpi PNG
    * no bbox_inches='tight'
    * pdf.fonttype = ps.fonttype = 42

Example
-------
python experiment_3_formatted.py \
    results/triangle/discrimination_utterance_pair.csv \
    results/preference/preference_utterance_pair.csv \
    objective_metrics_results/metrics_system_pairs.csv \
    --outdir experiment3
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize, least_squares
from scipy.special import expit
from scipy.stats import spearmanr, chi2


# ============================================================
# Constants
# ============================================================

CHANCE_LEVEL = 1.0 / 3.0
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 1234

# Same final width and height convention as Experiment 1.
DOUBLE_COLUMN_WIDTH = 3.35
FIGURE_HEIGHT = 1.6

# Marker/line sizes chosen to match Experiment 1 at final paper size.
POINT_SIZE = 10
FIT_LINEWIDTH = 1.35
REFERENCE_LINEWIDTH = 0.9

DISPLAY_NAMES = {
    "clean": "Clean",
    "MA_DNS": "MA",
    "MeanFlow_DNS": "MeanFlow",
    "mpsenet_DNS": "MP-SENet",
    "storm_DNS": "StoRM",
}

METRIC_DISPLAY_NAMES = {
    "pesq": "PESQ",
    "estoi": "ESTOI",
    "si_sdr": r"SI-SDR",
    "si_sdri": r"SI-SDRi",
    "csig": "CSIG",
    "cbak": "CBAK",
    "covl": "COVL",
    "polqa": "POLQA",
    "dnsmos": "DNSMOS",
    "utmos": "UTMOS",
    "nisqa": "NISQA",
    "squim_objective": "SQUIM",
    "squim": "SQUIM",
    "distill_mos": "Distill-MOS",
    "distillmos": "Distill-MOS",
    "scoreq": "SCOREQ",
    "urgentpk_mos": "URGENT-PK MOS",
}

REFERENCE_BASED_METRICS = {
    "pesq",
    "estoi",
    "si_sdr",
    "si_sdri",
    "csig",
    "cbak",
    "covl",
    "polqa",
}

METADATA_BASES = {
    "path",
    "relative_path",
    "file",
    "filename",
    "audio",
    "wav",
    "utterance",
    "utterance_id",
    "utterance_key",
    "dataset",
    "system",
    "system_a",
    "system_b",
    "system_A",
    "system_B",
    "pair",
}


# ============================================================
# Plot configuration -- same style as Experiment 1
# ============================================================

def configure_plotting():
    """Configure the same publication style used in Experiment 1."""

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",

            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,

            "axes.linewidth": 0.8,

            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,

            # Avoid Matplotlib Type 3 fonts.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,

            # Times-like LaTeX text/math.
            "text.latex.preamble": (
                r"\usepackage{newtxtext}"
                r"\usepackage{newtxmath}"
            ),
        }
    )


def make_full_box(ax):
    """Draw all four spines, matching Experiment 1."""

    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("black")

    ax.tick_params(
        axis="both",
        direction="out",
        color="black",
        labelcolor="black",
    )


def finish_axis(ax):
    """Apply common Experiment-1 axis styling."""

    ax.grid(
        axis="y",
        color="black",
        linewidth=0.4,
        alpha=0.12,
        zorder=0,
    )
    make_full_box(ax)


def save_figure(fig, path_without_suffix: Path):
    """Save PDF and 600-dpi PNG without bbox_inches='tight'."""

    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(path_without_suffix.with_suffix(".pdf"))
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=600)
    plt.close(fig)


# ============================================================
# General helpers
# ============================================================

def display_name(system):
    system = str(system)
    return DISPLAY_NAMES.get(system, system.replace("_DNS", ""))


def normalize_utterance_id(value):
    """Normalize e.g. DNS/fileid_59 and fileid_59 to the same key."""

    value = str(value).strip().replace("\\", "/")
    return value.split("/")[-1]


def canonical_pair(a, b):
    a = str(a)
    b = str(b)
    if a <= b:
        return a, b, 1.0
    return b, a, -1.0


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def latex_escape_text(text):
    """Escape plain text before using it in a LaTeX-rendered Matplotlib label."""

    text = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def pretty_metric(metric):
    metric = str(metric)
    if metric in METRIC_DISPLAY_NAMES:
        return METRIC_DISPLAY_NAMES[metric]
    return metric.replace("_", " ").upper()


def slugify(text):
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def first_existing(columns, candidates):
    for col in candidates:
        if col in columns:
            return col
    return None


def ensure_columns(df, required, label):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def extract_system_columns(df):
    """Return the two system columns from a table."""

    candidates = [
        ("system_A", "system_B"),
        ("system_a", "system_b"),
        ("system_x", "system_y"),
    ]
    for a, b in candidates:
        if a in df.columns and b in df.columns:
            return a, b
    raise ValueError(
        "Could not find system-pair columns. Expected one of: "
        "system_A/system_B, system_a/system_b, or system_x/system_y."
    )


# ============================================================
# Load and normalize Experiment 1 discrimination cells
# ============================================================

def prepare_discrimination(path: Path):
    df = pd.read_csv(path)

    utt_col = first_existing(
        df.columns,
        ["utterance_id", "utterance_key"],
    )
    if utt_col is None:
        raise ValueError("Discrimination CSV needs utterance_id or utterance_key.")

    sys_a_col, sys_b_col = extract_system_columns(df)

    p_col = first_existing(
        df.columns,
        ["p_hat_D", "p_hat_d", "accuracy", "discrimination_rate"],
    )
    if p_col is None:
        # Reconstruct from count columns when available.
        k_col = first_existing(df.columns, ["k", "n_correct", "correct"])
        n_col = first_existing(df.columns, ["n", "n_responses", "total"])
        if k_col is None or n_col is None:
            raise ValueError(
                "Discrimination CSV needs p_hat_D (or equivalent), or count columns k/n."
            )
        df["p_hat_D"] = safe_numeric(df[k_col]) / safe_numeric(df[n_col])
        p_col = "p_hat_D"

    k_col = first_existing(df.columns, ["k", "n_correct"])
    n_col = first_existing(df.columns, ["n", "n_responses"])

    rows = []
    for _, r in df.iterrows():
        a, b, _ = canonical_pair(r[sys_a_col], r[sys_b_col])
        row = {
            "utterance_id": normalize_utterance_id(r[utt_col]),
            "system_A": a,
            "system_B": b,
            "p_hat_D": pd.to_numeric(r[p_col], errors="coerce"),
        }
        if k_col is not None:
            row["k_D"] = pd.to_numeric(r[k_col], errors="coerce")
        if n_col is not None:
            row["n_D"] = pd.to_numeric(r[n_col], errors="coerce")
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["p_hat_D"])

    # If count columns are absent, infer no likelihood counts; SRCC still works.
    if "k_D" not in out.columns:
        out["k_D"] = np.nan
    if "n_D" not in out.columns:
        out["n_D"] = np.nan

    return out


# ============================================================
# Load and normalize Experiment 2 preference cells
# ============================================================

def prepare_preference(path: Path):
    df = pd.read_csv(path)

    utt_col = first_existing(df.columns, ["utterance_id", "utterance_key"])
    if utt_col is None:
        raise ValueError("Preference CSV needs utterance_id or utterance_key.")

    sys_a_col, sys_b_col = extract_system_columns(df)

    dec_col = first_existing(
        df.columns,
        ["p_hat_dec", "p_dec", "decisiveness", "decisiveness_rate"],
    )
    pref_col = first_existing(
        df.columns,
        ["S_hat_pref", "s_hat_pref", "S_pref", "preference_direction"],
    )

    n_a_col = first_existing(df.columns, ["N_A", "n_A", "n_a"])
    n_b_col = first_existing(df.columns, ["N_B", "n_B", "n_b"])
    n_n_col = first_existing(df.columns, ["N_N", "n_N", "n_n", "n_no_preference"])
    n_col = first_existing(df.columns, ["n", "n_responses", "total"])

    if dec_col is None:
        if n_a_col is None or n_b_col is None:
            raise ValueError(
                "Preference CSV needs p_hat_dec or count columns N_A and N_B."
            )
        if n_col is None:
            if n_n_col is None:
                raise ValueError(
                    "Need n or N_N in order to reconstruct decisiveness."
                )
            df["__n_total"] = (
                safe_numeric(df[n_a_col])
                + safe_numeric(df[n_b_col])
                + safe_numeric(df[n_n_col])
            )
            n_col = "__n_total"
        df["__p_hat_dec"] = (
            safe_numeric(df[n_a_col]) + safe_numeric(df[n_b_col])
        ) / safe_numeric(df[n_col])
        dec_col = "__p_hat_dec"

    if pref_col is None:
        if n_a_col is None or n_b_col is None:
            raise ValueError(
                "Preference CSV needs S_hat_pref or count columns N_A and N_B."
            )
        if n_col is None:
            if n_n_col is None:
                raise ValueError(
                    "Need n or N_N in order to reconstruct preference direction."
                )
            df["__n_total2"] = (
                safe_numeric(df[n_a_col])
                + safe_numeric(df[n_b_col])
                + safe_numeric(df[n_n_col])
            )
            n_col = "__n_total2"
        df["__S_hat_pref"] = (
            safe_numeric(df[n_a_col]) - safe_numeric(df[n_b_col])
        ) / safe_numeric(df[n_col])
        pref_col = "__S_hat_pref"

    rows = []
    for _, r in df.iterrows():
        raw_a = str(r[sys_a_col])
        raw_b = str(r[sys_b_col])
        a, b, orientation = canonical_pair(raw_a, raw_b)

        p_dec = pd.to_numeric(r[dec_col], errors="coerce")
        s_pref_raw = pd.to_numeric(r[pref_col], errors="coerce")

        # Signed preference direction must follow the canonical A/B orientation.
        s_pref = s_pref_raw * orientation

        row = {
            "utterance_id": normalize_utterance_id(r[utt_col]),
            "system_A": a,
            "system_B": b,
            "p_hat_dec": p_dec,
            "S_hat_pref": s_pref,
        }

        # Decisiveness likelihood counts.
        if n_a_col is not None and n_b_col is not None:
            n_a = pd.to_numeric(r[n_a_col], errors="coerce")
            n_b = pd.to_numeric(r[n_b_col], errors="coerce")
            row["k_dec"] = n_a + n_b
        else:
            row["k_dec"] = np.nan

        if n_col is not None:
            row["n_pref"] = pd.to_numeric(r[n_col], errors="coerce")
        elif n_a_col is not None and n_b_col is not None and n_n_col is not None:
            row["n_pref"] = (
                pd.to_numeric(r[n_a_col], errors="coerce")
                + pd.to_numeric(r[n_b_col], errors="coerce")
                + pd.to_numeric(r[n_n_col], errors="coerce")
            )
        else:
            row["n_pref"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    return out


# ============================================================
# Objective metric table
# ============================================================

def discover_metric_bases(df):
    """
    Find scalar metric bases from columns such as
        pesq_delta, pesq_abs_delta, pesq_a, pesq_b.

    Metadata/path-like columns are explicitly excluded.
    """

    bases = set()

    for col in df.columns:
        lower = col.lower()

        # IMPORTANT: ``*_abs_delta`` also ends with ``*_delta``.
        # Match the longest suffix first and STOP after the first match.
        # Without the break, e.g. ``pesq_abs_delta`` creates both
        # ``pesq`` and the spurious metric ``pesq_abs``.
        for suffix in ["_abs_delta", "_delta"]:
            if lower.endswith(suffix):
                base = col[: -len(suffix)]
                if base.lower() not in METADATA_BASES:
                    bases.add(base)
                break

    # Fall back to a/b metric columns only when a matching pair exists.
    cols = set(df.columns)
    for col in df.columns:
        if col.endswith("_a"):
            base = col[:-2]
            if base.lower() in METADATA_BASES:
                continue
            if f"{base}_b" in cols:
                bases.add(base)

    return sorted(bases)


def prepare_metrics(path: Path):
    raw = pd.read_csv(path)

    utt_col = first_existing(raw.columns, ["utterance_id", "utterance_key"])
    if utt_col is None:
        raise ValueError("Metric CSV needs utterance_id or utterance_key.")

    sys_a_col, sys_b_col = extract_system_columns(raw)
    metrics = discover_metric_bases(raw)

    # ``*_abs`` is never a separate metric.  The absolute quantity used for
    # discrimination/decisiveness is ``<metric>_abs_delta`` for the SAME metric.
    bad_metrics = [m for m in metrics if m.lower().endswith("_abs")]
    if bad_metrics:
        raise RuntimeError(
            "Spurious *_abs metric bases were discovered: "
            + ", ".join(bad_metrics)
        )

    if not metrics:
        raise ValueError(
            "No metric columns were discovered. Expected columns such as "
            "pesq_delta / pesq_abs_delta or pesq_a / pesq_b."
        )

    rows = []

    for _, r in raw.iterrows():
        raw_a = str(r[sys_a_col])
        raw_b = str(r[sys_b_col])
        a, b, orientation = canonical_pair(raw_a, raw_b)

        row = {
            "utterance_id": normalize_utterance_id(r[utt_col]),
            "system_A": a,
            "system_B": b,
        }

        for metric in metrics:
            a_col = f"{metric}_a"
            b_col = f"{metric}_b"
            delta_col = f"{metric}_delta"
            abs_col = f"{metric}_abs_delta"

            delta_raw = np.nan

            # Prefer explicit A/B values because orientation is unambiguous.
            if a_col in raw.columns and b_col in raw.columns:
                va = pd.to_numeric(r[a_col], errors="coerce")
                vb = pd.to_numeric(r[b_col], errors="coerce")
                if np.isfinite(va) and np.isfinite(vb):
                    delta_raw = va - vb
            elif delta_col in raw.columns:
                delta_raw = pd.to_numeric(r[delta_col], errors="coerce")

            if np.isfinite(delta_raw):
                delta_canonical = float(delta_raw) * orientation
                abs_delta = abs(delta_canonical)
            else:
                delta_canonical = np.nan
                if abs_col in raw.columns:
                    abs_delta = abs(pd.to_numeric(r[abs_col], errors="coerce"))
                else:
                    abs_delta = np.nan

            row[f"{metric}_delta"] = delta_canonical
            row[f"{metric}_abs_delta"] = abs_delta

        rows.append(row)

    out = pd.DataFrame(rows)

    # Remove duplicate rows that can otherwise multiply listening cells on merge.
    key = ["utterance_id", "system_A", "system_B"]
    duplicate_count = int(out.duplicated(key).sum())
    if duplicate_count:
        print(
            f"WARNING: objective metric table contains {duplicate_count} duplicate "
            "utterance/pair rows; keeping the first occurrence."
        )
        out = out.drop_duplicates(key, keep="first")

    return out, metrics


# ============================================================
# Merge all three sources
# ============================================================

def merge_inputs(discrimination, preference, metrics):
    key = ["utterance_id", "system_A", "system_B"]

    perceptual = discrimination.merge(
        preference,
        on=key,
        how="inner",
        validate="one_to_one",
    )

    merged = perceptual.merge(
        metrics,
        on=key,
        how="inner",
        validate="one_to_one",
    )

    print("\nJoin diagnostics")
    print("----------------")
    print(f"Discrimination cells : {len(discrimination)}")
    print(f"Preference cells     : {len(preference)}")
    print(f"Perceptual overlap   : {len(perceptual)}")
    print(f"Metric rows          : {len(metrics)}")
    print(f"Final merged cells   : {len(merged)}")
    print(f"Unique utterances    : {merged['utterance_id'].nunique()}")
    print()

    return merged


# ============================================================
# Metric filtering
# ============================================================

def metric_is_reference_based(metric):
    return metric.lower() in REFERENCE_BASED_METRICS


def subset_for_metric(df, metric):
    """Return the exact cell subset used for SRCC, fit, and plotting.

    Intrusive/reference-based metrics are undefined as fair pairwise
    comparisons when one of the candidate systems is the clean reference.
    Therefore every Clean-containing pair is removed *before* any SRCC,
    bootstrap, response-function fit, or figure is computed.
    """

    out = df.copy()

    if metric_is_reference_based(metric):
        a = out["system_A"].astype(str).str.strip().str.lower()
        b = out["system_B"].astype(str).str.strip().str.lower()
        out = out.loc[(a != "clean") & (b != "clean")].copy()

        # Defensive check: a reference-based analysis must never contain Clean.
        a_check = out["system_A"].astype(str).str.strip().str.lower()
        b_check = out["system_B"].astype(str).str.strip().str.lower()
        if ((a_check == "clean") | (b_check == "clean")).any():
            raise RuntimeError(
                f"Internal error: Clean comparison survived filtering for {metric}."
            )

    return out


# ============================================================
# Spearman SRCC + utterance-cluster bootstrap CI
# ============================================================

def spearman_complete(df, x_col, y_col):
    x = safe_numeric(df[x_col]).to_numpy(dtype=float)
    y = safe_numeric(df[y_col]).to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan, len(x)

    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue), len(x)


def bootstrap_spearman_by_utterance(
    df,
    x_col,
    y_col,
    n_boot=BOOTSTRAP_SAMPLES,
    seed=BOOTSTRAP_SEED,
):
    """
    Cluster bootstrap over utterances.

    Each sampled utterance contributes all of its system-pair cells. If an
    utterance is sampled more than once, all of its rows are duplicated as a
    separate bootstrap cluster copy rather than collapsed.
    """

    work = df[["utterance_id", x_col, y_col]].copy()
    work[x_col] = safe_numeric(work[x_col])
    work[y_col] = safe_numeric(work[y_col])
    work = work.dropna(subset=[x_col, y_col])

    utterances = work["utterance_id"].drop_duplicates().to_numpy()
    if len(utterances) < 2:
        return np.nan, np.nan, 0

    groups = {
        u: work.loc[work["utterance_id"] == u, [x_col, y_col]].copy()
        for u in utterances
    }

    rng = np.random.default_rng(seed)
    rhos = []

    for _ in range(int(n_boot)):
        sampled = rng.choice(utterances, size=len(utterances), replace=True)

        xs = []
        ys = []
        for u in sampled:
            g = groups[u]
            xs.append(g[x_col].to_numpy(dtype=float))
            ys.append(g[y_col].to_numpy(dtype=float))

        x = np.concatenate(xs)
        y = np.concatenate(ys)

        if np.unique(x).size < 2 or np.unique(y).size < 2:
            continue

        rho = spearmanr(x, y).statistic
        if np.isfinite(rho):
            rhos.append(float(rho))

    if not rhos:
        return np.nan, np.nan, 0

    lo, hi = np.quantile(rhos, [0.025, 0.975])
    return float(lo), float(hi), len(rhos)


# ============================================================
# Likelihood / response-function fitting
# ============================================================

def bernoulli_binomial_loglik(k, n, p):
    """Binomial log-likelihood ignoring the constant combinatorial term."""

    eps = 1e-10
    p = np.clip(p, eps, 1.0 - eps)
    return np.sum(k * np.log(p) + (n - k) * np.log(1.0 - p))


def fit_discrimination_psychometric(df, x_col):
    """
    Maximum-likelihood fit of the chance-constrained triangle response model

        P_D(D) = 1/3 + 2/3 * sigmoid(alpha + beta D).

    In addition to alpha/beta, this function reports diagnostics for the
    psychometric model:

      * loglik: fitted-model binomial log-likelihood (constant omitted)
      * loglik_null: intercept-only chance-constrained model
      * loglik_saturated: saturated grouped-binomial model
      * deviance: 2 * (loglik_saturated - loglik)
      * deviance_df: number of cells minus number of fitted parameters
      * deviance_p: asymptotic chi-square goodness-of-fit p-value
      * null_deviance: 2 * (loglik_saturated - loglik_null)
      * lrt_stat: 2 * (loglik - loglik_null)
      * lrt_df: 1 (the added metric-separation slope)
      * lrt_p: chi-square likelihood-ratio p-value
      * AIC / AIC_null
      * pseudo_R2_deviance: proportional reduction in deviance vs null

    The deviance GOF p-value is an asymptotic diagnostic. With sparse grouped
    cells, the plotted empirical proportions and bootstrap uncertainty should
    still be inspected rather than treating this p-value as the sole GOF test.
    """

    work = df[[x_col, "k_D", "n_D"]].copy()
    work[x_col] = safe_numeric(work[x_col])
    work["k_D"] = safe_numeric(work["k_D"])
    work["n_D"] = safe_numeric(work["n_D"])
    work = work.dropna()
    work = work.loc[
        (work["n_D"] > 0)
        & (work["k_D"] >= 0)
        & (work["k_D"] <= work["n_D"])
    ]

    if len(work) < 3:
        return None

    x = work[x_col].to_numpy(dtype=float)
    k = work["k_D"].to_numpy(dtype=float)
    n = work["n_D"].to_numpy(dtype=float)

    def model_probability(alpha, beta):
        return CHANCE_LEVEL + (1.0 - CHANCE_LEVEL) * expit(alpha + beta * x)

    def nll(theta):
        alpha, beta = theta
        return -bernoulli_binomial_loglik(k, n, model_probability(alpha, beta))

    # Full metric model.
    fit = minimize(nll, x0=np.array([0.0, 1.0]), method="BFGS")
    if not np.all(np.isfinite(fit.x)):
        return None

    alpha, beta = map(float, fit.x)
    p_fit = model_probability(alpha, beta)
    ll_fit = float(bernoulli_binomial_loglik(k, n, p_fit))

    # Intercept-only null model. It retains the triangle-test lower asymptote
    # but contains no information from objective metric separation.
    def null_nll(theta):
        alpha0 = float(np.atleast_1d(theta)[0])
        p0 = CHANCE_LEVEL + (1.0 - CHANCE_LEVEL) * expit(alpha0)
        p = np.full_like(x, p0, dtype=float)
        return -bernoulli_binomial_loglik(k, n, p)

    null_fit = minimize(null_nll, x0=np.array([0.0]), method="BFGS")
    ll_null = float(-null_fit.fun) if np.isfinite(null_fit.fun) else np.nan
    alpha_null = float(null_fit.x[0]) if np.all(np.isfinite(null_fit.x)) else np.nan

    # Saturated grouped-binomial model. For k=0 or k=n the conventional
    # 0*log(0)=0 limit is used by the clipped log-likelihood helper.
    p_sat = np.divide(k, n, out=np.zeros_like(k, dtype=float), where=n > 0)
    ll_sat = float(bernoulli_binomial_loglik(k, n, p_sat))

    n_cells = int(len(work))
    n_parameters = 2
    deviance_df = max(n_cells - n_parameters, 0)
    deviance = max(0.0, 2.0 * (ll_sat - ll_fit))
    deviance_p = (
        float(chi2.sf(deviance, deviance_df))
        if deviance_df > 0 and np.isfinite(deviance)
        else np.nan
    )

    null_deviance = (
        max(0.0, 2.0 * (ll_sat - ll_null)) if np.isfinite(ll_null) else np.nan
    )

    # Full vs intercept-only LRT: exactly one additional parameter, beta.
    lrt_stat = (
        max(0.0, 2.0 * (ll_fit - ll_null)) if np.isfinite(ll_null) else np.nan
    )
    lrt_df = 1
    lrt_p = float(chi2.sf(lrt_stat, lrt_df)) if np.isfinite(lrt_stat) else np.nan

    aic = 2 * n_parameters - 2 * ll_fit
    aic_null = 2 * 1 - 2 * ll_null if np.isfinite(ll_null) else np.nan
    pseudo_r2 = (
        1.0 - deviance / null_deviance
        if np.isfinite(null_deviance) and null_deviance > 0
        else np.nan
    )

    return {
        "alpha": alpha,
        "beta": beta,
        "loglik": ll_fit,
        "success": bool(fit.success),
        "n_cells_fit": n_cells,
        "n_responses_fit": int(np.sum(n)),
        "alpha_null": alpha_null,
        "loglik_null": ll_null,
        "loglik_saturated": ll_sat,
        "deviance": float(deviance),
        "deviance_df": int(deviance_df),
        "deviance_p": deviance_p,
        "null_deviance": float(null_deviance) if np.isfinite(null_deviance) else np.nan,
        "lrt_stat": float(lrt_stat) if np.isfinite(lrt_stat) else np.nan,
        "lrt_df": int(lrt_df),
        "lrt_p": lrt_p,
        "aic": float(aic),
        "aic_null": float(aic_null) if np.isfinite(aic_null) else np.nan,
        "delta_aic_vs_null": (
            float(aic_null - aic) if np.isfinite(aic_null) else np.nan
        ),
        "pseudo_R2_deviance": float(pseudo_r2) if np.isfinite(pseudo_r2) else np.nan,
    }

def fit_decisiveness_logistic(df, x_col):
    """
    MLE for
        P_dec = sigmoid(alpha + beta D).
    """

    work = df[[x_col, "k_dec", "n_pref"]].copy()
    work[x_col] = safe_numeric(work[x_col])
    work["k_dec"] = safe_numeric(work["k_dec"])
    work["n_pref"] = safe_numeric(work["n_pref"])
    work = work.dropna()
    work = work.loc[work["n_pref"] > 0]

    if len(work) < 3:
        return None

    x = work[x_col].to_numpy(dtype=float)
    k = work["k_dec"].to_numpy(dtype=float)
    n = work["n_pref"].to_numpy(dtype=float)

    def nll(theta):
        alpha, beta = theta
        p = expit(alpha + beta * x)
        return -bernoulli_binomial_loglik(k, n, p)

    fit = minimize(
        nll,
        x0=np.array([0.0, 1.0]),
        method="BFGS",
    )

    if not np.all(np.isfinite(fit.x)):
        return None

    return {
        "alpha": float(fit.x[0]),
        "beta": float(fit.x[1]),
        "loglik": float(-fit.fun),
        "success": bool(fit.success),
        "n_cells_fit": int(len(work)),
        "n_responses_fit": int(np.sum(n)),
    }


def fit_preference_bounded_response(df, x_col):
    """
    Weighted nonlinear least squares for
        E[Y_pref | Delta] = 2*sigmoid(alpha + beta Delta) - 1.

    Cell weights are sqrt(n_pref), so cells supported by more responses have
    greater influence while the response remains the empirical signed mean.
    """

    work = df[[x_col, "S_hat_pref", "n_pref"]].copy()
    work[x_col] = safe_numeric(work[x_col])
    work["S_hat_pref"] = safe_numeric(work["S_hat_pref"])
    work["n_pref"] = safe_numeric(work["n_pref"])
    work = work.dropna(subset=[x_col, "S_hat_pref"])

    if len(work) < 3:
        return None

    x = work[x_col].to_numpy(dtype=float)
    y = work["S_hat_pref"].to_numpy(dtype=float)
    n = work["n_pref"].fillna(1.0).clip(lower=1.0).to_numpy(dtype=float)
    w = np.sqrt(n)

    def residual(theta):
        alpha, beta = theta
        pred = 2.0 * expit(alpha + beta * x) - 1.0
        return w * (pred - y)

    fit = least_squares(
        residual,
        x0=np.array([0.0, 1.0]),
    )

    if not np.all(np.isfinite(fit.x)):
        return None

    return {
        "alpha": float(fit.x[0]),
        "beta": float(fit.x[1]),
        "weighted_sse": float(np.sum(fit.fun ** 2)),
        "success": bool(fit.success),
        "n_cells_fit": int(len(work)),
        "n_responses_fit": int(np.nansum(n)),
    }


# ============================================================
# Plot helpers
# ============================================================

def scatter_open(ax, x, y):
    ax.scatter(
        x,
        y,
        s=POINT_SIZE,
        facecolors="none",
        edgecolors="black",
        linewidths=0.7,
        alpha=0.42,
        zorder=2,
    )


def plot_fit_line(ax, x_grid, y_grid, label="Fit"):
    ax.plot(
        x_grid,
        y_grid,
        color="black",
        linewidth=FIT_LINEWIDTH,
        linestyle="-",
        label=label,
        zorder=3,
    )


def finite_xy(df, x_col, y_col):
    x = safe_numeric(df[x_col]).to_numpy(dtype=float)
    y = safe_numeric(df[y_col]).to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def padded_range(x, symmetric=False):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return -1.0, 1.0

    lo = float(np.min(x))
    hi = float(np.max(x))

    if symmetric:
        lim = max(abs(lo), abs(hi))
        if lim == 0:
            lim = 1.0
        return -1.05 * lim, 1.05 * lim

    span = hi - lo
    if span == 0:
        pad = max(abs(lo) * 0.05, 0.05)
    else:
        pad = 0.04 * span
    return lo - pad, hi + pad


def srcc_annotation(rho, lo, hi):
    if not (np.isfinite(rho) and np.isfinite(lo) and np.isfinite(hi)):
        return None
    return (
        r"$\rho_S="
        + f"{rho:.2f}"
        + r"$"
        + "\n"
        + r"$95\%\ \mathrm{CI}=["
        + f"{lo:.2f}, {hi:.2f}"
        + r"]$"
    )


def add_srcc_annotation(ax, rho, lo, hi):
    text = srcc_annotation(rho, lo, hi)
    if text is None:
        return
    ax.text(
        0.03,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="black",
    )


# ============================================================
# Metric-specific plots -- Experiment 1 formatting
# ============================================================

def plot_metric_discrimination(metric_df, metric, stats_row, fit, outdir):
    x_col = f"{metric}_abs_delta"
    x, y = finite_xy(metric_df, x_col, "p_hat_D")
    if len(x) == 0:
        return

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, FIGURE_HEIGHT))
    scatter_open(ax, x, y)

    if fit is not None:
        x_lo, x_hi = padded_range(x)
        x_grid = np.linspace(max(0.0, x_lo), x_hi, 400)
        y_grid = CHANCE_LEVEL + (1.0 - CHANCE_LEVEL) * expit(
            fit["alpha"] + fit["beta"] * x_grid
        )
        plot_fit_line(ax, x_grid, y_grid)

    ax.axhline(
        CHANCE_LEVEL,
        color="black",
        linestyle="--",
        linewidth=REFERENCE_LINEWIDTH,
        label=r"Chance ($p_0=\frac{1}{3}$)",
        zorder=1,
    )

    metric_label = latex_escape_text(pretty_metric(metric))
    ax.set_xlabel(r"$D^{(m)}_{u,A,B}=|\Delta m_{u,A,B}|$" + "  " + f"({metric_label})")
    ax.set_ylabel(r"Discrimination rate $\hat{p}^{D}_{u,A,B}$")
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))

    add_srcc_annotation(
        ax,
        stats_row["rho"],
        stats_row["ci95_lo"],
        stats_row["ci95_hi"],
    )

    finish_axis(ax)
    ax.legend(frameon=False, loc="lower right")

    save_figure(
        fig,
        outdir / "discrimination" / f"{slugify(metric)}_discrimination",
    )


# ============================================================
# Combined manuscript figure configuration
# ============================================================
# Edit ONLY this list to add/remove/reorder columns.
# Tuple: (metric key in CSV, displayed column title)
COMBINED_PLOT_METRICS = [
    ("pesq", "PESQ"),
    ("squim_stoi", r"$\mathrm{SQUIM}_{\mathrm{STOI}}$"),
]

# Final ICASSP single-column dimensions.
COMBINED_PLOT_WIDTH = 3.35
COMBINED_PLOT_HEIGHT = 2.30

# Fine layout controls. These are deliberately exposed here so the figure can
# be tuned without touching the plotting code.
COMBINED_LEFT = 0.175
COMBINED_RIGHT = 0.995
COMBINED_BOTTOM = 0.175
COMBINED_TOP = 0.925
COMBINED_WSPACE = 0.20
COMBINED_HSPACE = 0.38
COMBINED_RHO_FONTSIZE = 7.0
COMBINED_TICK_FONTSIZE = 7.0
COMBINED_TITLE_FONTSIZE = 8.0
COMBINED_LABEL_FONTSIZE = 8.0


def _rho_only_annotation(ax, df, x_col, y_col):
    """Small SRCC annotation suitable for a 3-column single-column figure."""
    rho, _, _ = spearman_complete(df, x_col, y_col)
    if np.isfinite(rho):
        ax.text(
            0.05, 0.94,
            rf"$\rho_S={rho:.2f}$",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=COMBINED_RHO_FONTSIZE,
            color="black",
            zorder=5,
        )


def plot_discrimination_preference_grid(merged, outdir):
    """Single-column 2 x N manuscript figure.

    Top row:    |metric difference| -> triangle discrimination.
    Bottom row: signed metric difference -> signed preference direction.

    Metric names appear only as column titles. The x-axis labels are generic
    |Delta m| and Delta m so the panels remain readable at 3.35 inches.
    """
    panels = COMBINED_PLOT_METRICS
    ncols = len(panels)
    if ncols == 0:
        return

    with plt.rc_context({
        "font.size": 9,
        "axes.labelsize": COMBINED_LABEL_FONTSIZE,
        "axes.titlesize": COMBINED_TITLE_FONTSIZE,
        "xtick.labelsize": COMBINED_TICK_FONTSIZE,
        "ytick.labelsize": COMBINED_TICK_FONTSIZE,
    }):
        fig, axes = plt.subplots(
            2, ncols,
            figsize=(COMBINED_PLOT_WIDTH, COMBINED_PLOT_HEIGHT),
            squeeze=False,
            sharey="row",
        )

        for j, (metric, title) in enumerate(panels):
            df = subset_for_metric(merged, metric)
            abs_col = f"{metric}_abs_delta"
            delta_col = f"{metric}_delta"

            if abs_col not in df.columns:
                raise KeyError(f"Missing required column: {abs_col}")
            if delta_col not in df.columns:
                raise KeyError(f"Missing required column: {delta_col}")

            # ---------- Top row: discrimination ----------
            ax = axes[0, j]
            x, y = finite_xy(df, abs_col, "p_hat_D")
            if len(x) == 0:
                raise ValueError(f"No finite discrimination observations for {metric}")

            ax.scatter(
                x, y, s=9, facecolors="none", edgecolors="black",
                linewidths=0.55, alpha=0.42, zorder=2,
            )

            fit = fit_discrimination_psychometric(df, abs_col)
            if fit is not None:
                lo, hi = padded_range(x)
                grid = np.linspace(max(0.0, lo), hi, 400)
                curve = CHANCE_LEVEL + (1.0 - CHANCE_LEVEL) * expit(
                    fit["alpha"] + fit["beta"] * grid
                )
                ax.plot(grid, curve, color="black", linewidth=1.0, zorder=3)

            ax.axhline(
                CHANCE_LEVEL, color="black", linestyle="--",
                linewidth=0.7, zorder=1,
            )
            ax.set_title(title, pad=2.5)
            ax.set_xlabel(r"$|\Delta m|$", labelpad=1.5)
            ax.set_ylim(0.25, 1.02)
            ax.set_yticks([1.0/3.0, 0.6, 0.8, 1.0])
            if j == 0:
                ax.set_yticklabels([r"$1/3$", r"$0.6$", r"$0.8$", r"$1.0$"])
            _rho_only_annotation(ax, df, abs_col, "p_hat_D")
            finish_axis(ax)
            ax.tick_params(axis="both", which="major", pad=1.0)

            # ---------- Bottom row: preference direction ----------
            ax = axes[1, j]
            x, y = finite_xy(df, delta_col, "S_hat_pref")
            if len(x) == 0:
                raise ValueError(f"No finite preference observations for {metric}")

            ax.scatter(
                x, y, s=9, facecolors="none", edgecolors="black",
                linewidths=0.55, alpha=0.42, zorder=2,
            )

            fit = fit_preference_bounded_response(df, delta_col)
            if fit is not None:
                lo, hi = padded_range(x, symmetric=True)
                grid = np.linspace(lo, hi, 400)
                curve = 2.0 * expit(fit["alpha"] + fit["beta"] * grid) - 1.0
                ax.plot(grid, curve, color="black", linewidth=1.0, zorder=3)

            ax.axhline(0.0, color="black", linestyle="--", linewidth=0.7, zorder=1)
            ax.axvline(0.0, color="black", linestyle=":", linewidth=0.7, zorder=1)
            lo, hi = padded_range(x, symmetric=True)
            ax.set_xlim(lo, hi)
            ax.set_ylim(-1.02, 1.02)
            ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax.set_xlabel(r"$\Delta m$", labelpad=1.5)
            _rho_only_annotation(ax, df, delta_col, "S_hat_pref")
            finish_axis(ax)
            ax.tick_params(axis="both", which="major", pad=1.0)

        # One y label per row, only on the first column.
        axes[0, 0].set_ylabel(r"Discrimination $\hat{p}^{D}$", labelpad=2.5)
        axes[1, 0].set_ylabel(r"Preference $\hat{S}$", labelpad=2.5)

        # Suppress repeated y tick labels without changing the shared ticks.
        for j in range(1, ncols):
            axes[0, j].tick_params(axis="y", labelleft=False)
            axes[1, j].tick_params(axis="y", labelleft=False)

        fig.subplots_adjust(
            left=COMBINED_LEFT,
            right=COMBINED_RIGHT,
            bottom=COMBINED_BOTTOM,
            top=COMBINED_TOP,
            wspace=COMBINED_WSPACE,
            hspace=COMBINED_HSPACE,
        )

        base = outdir / "combined" / "discrimination_preference_2x3"
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(base.with_suffix(".pdf"))
        fig.savefig(base.with_suffix(".png"), dpi=600)
        plt.close(fig)


def plot_metric_decisiveness(metric_df, metric, stats_row, fit, outdir):
    x_col = f"{metric}_abs_delta"
    x, y = finite_xy(metric_df, x_col, "p_hat_dec")
    if len(x) == 0:
        return

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, FIGURE_HEIGHT))
    scatter_open(ax, x, y)

    if fit is not None:
        x_lo, x_hi = padded_range(x)
        x_grid = np.linspace(max(0.0, x_lo), x_hi, 400)
        y_grid = expit(fit["alpha"] + fit["beta"] * x_grid)
        plot_fit_line(ax, x_grid, y_grid)

    metric_label = latex_escape_text(pretty_metric(metric))
    ax.set_xlabel(r"$D^{(m)}_{u,A,B}=|\Delta m_{u,A,B}|$" + "  " + f"({metric_label})")
    ax.set_ylabel(r"Decisiveness $\hat{p}^{\mathrm{dec}}_{u,A,B}$")
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))

    add_srcc_annotation(
        ax,
        stats_row["rho"],
        stats_row["ci95_lo"],
        stats_row["ci95_hi"],
    )

    finish_axis(ax)
    if fit is not None:
        ax.legend(frameon=False, loc="lower right")

    save_figure(
        fig,
        outdir / "decisiveness" / f"{slugify(metric)}_decisiveness",
    )


def plot_metric_preference(metric_df, metric, stats_row, fit, outdir):
    x_col = f"{metric}_delta"
    x, y = finite_xy(metric_df, x_col, "S_hat_pref")
    if len(x) == 0:
        return

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, FIGURE_HEIGHT))
    scatter_open(ax, x, y)

    if fit is not None:
        x_lo, x_hi = padded_range(x, symmetric=True)
        x_grid = np.linspace(x_lo, x_hi, 400)
        y_grid = 2.0 * expit(fit["alpha"] + fit["beta"] * x_grid) - 1.0
        plot_fit_line(ax, x_grid, y_grid)

    ax.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=REFERENCE_LINEWIDTH,
        zorder=1,
    )
    ax.axvline(
        0.0,
        color="black",
        linestyle=":",
        linewidth=REFERENCE_LINEWIDTH,
        zorder=1,
    )

    metric_label = latex_escape_text(pretty_metric(metric))
    ax.set_xlabel(r"Signed metric difference $\Delta m_{u,A,B}$" + "  " + f"({metric_label})")
    ax.set_ylabel(r"Preference direction $\hat{S}_{u,A,B}$")
    ax.set_ylim(-1.02, 1.02)
    ax.set_yticks(np.arange(-1.0, 1.01, 0.5))

    x_lo, x_hi = padded_range(x, symmetric=True)
    ax.set_xlim(x_lo, x_hi)

    add_srcc_annotation(
        ax,
        stats_row["rho"],
        stats_row["ci95_lo"],
        stats_row["ci95_hi"],
    )

    finish_axis(ax)
    if fit is not None:
        ax.legend(frameon=False, loc="lower right")

    save_figure(
        fig,
        outdir / "preference_direction" / f"{slugify(metric)}_preference_direction",
    )


def plot_discrimination_vs_decisiveness(df, stats_row, outdir):
    x, y = finite_xy(df, "p_hat_D", "p_hat_dec")
    if len(x) == 0:
        return

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, FIGURE_HEIGHT))
    scatter_open(ax, x, y)

    ax.axvline(
        CHANCE_LEVEL,
        color="black",
        linestyle="--",
        linewidth=REFERENCE_LINEWIDTH,
        label=r"Chance ($p_0=\frac{1}{3}$)",
        zorder=1,
    )

    ax.set_xlabel(r"Discrimination rate $\hat{p}^{D}_{u,A,B}$")
    ax.set_ylabel(r"Decisiveness $\hat{p}^{\mathrm{dec}}_{u,A,B}$")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))

    add_srcc_annotation(
        ax,
        stats_row["rho"],
        stats_row["ci95_lo"],
        stats_row["ci95_hi"],
    )

    finish_axis(ax)
    ax.legend(frameon=False, loc="lower right")

    save_figure(
        fig,
        outdir / "discrimination_vs_decisiveness",
    )


# ============================================================
# Analysis tables
# ============================================================

def srcc_row(
    df,
    x_col,
    y_col,
    analysis,
    metric=None,
    n_boot=BOOTSTRAP_SAMPLES,
    seed=BOOTSTRAP_SEED,
):
    rho, p_value, n_cells = spearman_complete(df, x_col, y_col)
    lo, hi, n_valid_boot = bootstrap_spearman_by_utterance(
        df,
        x_col,
        y_col,
        n_boot=n_boot,
        seed=seed,
    )

    return {
        "analysis": analysis,
        "metric": metric if metric is not None else "",
        "n_cells": n_cells,
        "n_utterances": int(df["utterance_id"].nunique()),
        "rho": rho,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "p_scipy_naive": p_value,
        "n_valid_bootstrap": n_valid_boot,
    }


def fit_to_columns(prefix, fit):
    """Flatten fit results into CSV columns without dropping diagnostics."""
    standard_keys = [
        "alpha", "beta", "success", "n_cells_fit", "n_responses_fit",
        "loglik", "weighted_sse", "alpha_null", "loglik_null",
        "loglik_saturated", "deviance", "deviance_df", "deviance_p",
        "null_deviance", "lrt_stat", "lrt_df", "lrt_p", "aic",
        "aic_null", "delta_aic_vs_null", "pseudo_R2_deviance",
    ]

    if fit is None:
        return {
            f"{prefix}_{key}": (False if key == "success" else np.nan)
            for key in standard_keys
        }

    result = {}
    for key in standard_keys:
        if key == "success":
            default = False
        else:
            default = np.nan
        result[f"{prefix}_{key}"] = fit.get(key, default)
    return result


def format_rho_ci(row):
    if not np.isfinite(row["rho"]):
        return "--"
    if np.isfinite(row["ci95_lo"]) and np.isfinite(row["ci95_hi"]):
        return (
            f"{row['rho']:.2f} "
            + r"["
            + f"{row['ci95_lo']:.2f}, {row['ci95_hi']:.2f}"
            + r"]"
        )
    return f"{row['rho']:.2f}"


def write_srcc_latex(df, path: Path, caption_label=None):
    """Write a compact two-column-paper-ready SRCC tabular."""

    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"Metric & $N$ & $\rho_S$ [95\% CI] \\",
        r"\hline",
    ]

    for _, row in df.iterrows():
        metric = pretty_metric(row["metric"]) if str(row["metric"]) else "--"
        metric = metric.replace("&", r"\&")
        lines.append(
            f"{metric} & {int(row['n_cells'])} & {format_rho_ci(row)} \\\\" 
        )

    lines.extend([r"\hline", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


# ============================================================
# Main Experiment 3 analysis
# ============================================================

def run_analysis(
    discrimination_path: Path,
    preference_path: Path,
    metric_path: Path,
    outdir: Path,
    n_boot: int,
    seed: int,
):
    outdir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    discrimination = prepare_discrimination(discrimination_path)
    preference = prepare_preference(preference_path)
    metrics_df, metrics = prepare_metrics(metric_path)

    metrics_df.to_csv(
        outdir / "pairwise_metrics_canonical.csv",
        index=False,
    )

    merged = merge_inputs(discrimination, preference, metrics_df)
    merged.to_csv(
        outdir / "experiment3_merged_cells.csv",
        index=False,
    )

    print("Metrics discovered:")
    for metric in metrics:
        tag = "reference-based" if metric_is_reference_based(metric) else "non-intrusive/pairwise"
        print(f"  {metric:24s} [{tag}]")
    print()

    discrimination_rows = []
    decisiveness_rows = []
    preference_rows = []
    combined_rows = []

    # --------------------------------------------------------
    # Direct discrimination versus decisiveness
    # --------------------------------------------------------
    direct_row = srcc_row(
        merged,
        "p_hat_D",
        "p_hat_dec",
        analysis="discrimination_vs_decisiveness",
        metric="",
        n_boot=n_boot,
        seed=seed,
    )
    direct_df = pd.DataFrame([direct_row])
    direct_df.to_csv(
        outdir / "discrimination_vs_decisiveness.csv",
        index=False,
    )
    plot_discrimination_vs_decisiveness(
        merged,
        direct_row,
        outdir,
    )

    # --------------------------------------------------------
    # Per-metric analyses
    # --------------------------------------------------------
    for metric_index, metric in enumerate(metrics):
        metric_df = subset_for_metric(merged, metric)

        abs_col = f"{metric}_abs_delta"
        delta_col = f"{metric}_delta"

        if abs_col not in metric_df.columns or delta_col not in metric_df.columns:
            continue

        # Use a deterministic but metric-specific bootstrap seed.
        metric_seed = seed + metric_index * 101

        # 1) Objective separation -> discrimination
        row_d = srcc_row(
            metric_df,
            abs_col,
            "p_hat_D",
            analysis="metric_separation_vs_discrimination",
            metric=metric,
            n_boot=n_boot,
            seed=metric_seed,
        )
        fit_d = fit_discrimination_psychometric(metric_df, abs_col)
        row_d.update(fit_to_columns("fit", fit_d))
        row_d["reference_based"] = metric_is_reference_based(metric)
        discrimination_rows.append(row_d)

        plot_metric_discrimination(
            metric_df,
            metric,
            row_d,
            fit_d,
            outdir,
        )

        # 2) Objective separation -> decisiveness
        row_dec = srcc_row(
            metric_df,
            abs_col,
            "p_hat_dec",
            analysis="metric_separation_vs_decisiveness",
            metric=metric,
            n_boot=n_boot,
            seed=metric_seed + 1,
        )
        fit_dec = fit_decisiveness_logistic(metric_df, abs_col)
        row_dec.update(fit_to_columns("fit", fit_dec))
        row_dec["reference_based"] = metric_is_reference_based(metric)
        decisiveness_rows.append(row_dec)

        plot_metric_decisiveness(
            metric_df,
            metric,
            row_dec,
            fit_dec,
            outdir,
        )

        # 3) Signed metric difference -> signed preference direction
        row_pref = srcc_row(
            metric_df,
            delta_col,
            "S_hat_pref",
            analysis="metric_direction_vs_preference_direction",
            metric=metric,
            n_boot=n_boot,
            seed=metric_seed + 2,
        )
        fit_pref = fit_preference_bounded_response(metric_df, delta_col)
        row_pref.update(fit_to_columns("fit", fit_pref))
        row_pref["reference_based"] = metric_is_reference_based(metric)
        preference_rows.append(row_pref)

        plot_metric_preference(
            metric_df,
            metric,
            row_pref,
            fit_pref,
            outdir,
        )

        combined_rows.extend([row_d, row_dec, row_pref])

    # Combined 2 x N manuscript figure: discrimination + preference direction.
    plot_discrimination_preference_grid(merged, outdir)

    discrimination_table = pd.DataFrame(discrimination_rows)
    decisiveness_table = pd.DataFrame(decisiveness_rows)
    preference_table = pd.DataFrame(preference_rows)
    combined_table = pd.DataFrame(combined_rows)

    # Sort by SRCC from highest to lowest for compact reporting.
    for table in [discrimination_table, decisiveness_table, preference_table]:
        if not table.empty:
            table.sort_values("rho", ascending=False, inplace=True, na_position="last")
            table.reset_index(drop=True, inplace=True)

    discrimination_table.to_csv(
        outdir / "srcc_discrimination.csv",
        index=False,
    )
    decisiveness_table.to_csv(
        outdir / "srcc_decisiveness.csv",
        index=False,
    )
    preference_table.to_csv(
        outdir / "srcc_preference_direction.csv",
        index=False,
    )
    combined_table.to_csv(
        outdir / "experiment3_metric_results.csv",
        index=False,
    )

    # Sorted aliases retained for convenience.
    discrimination_table.to_csv(
        outdir / "srcc_discrimination_sorted.csv",
        index=False,
    )
    decisiveness_table.to_csv(
        outdir / "srcc_decisiveness_sorted.csv",
        index=False,
    )
    preference_table.to_csv(
        outdir / "srcc_preference_direction_sorted.csv",
        index=False,
    )

    # LaTeX tables.
    if not discrimination_table.empty:
        write_srcc_latex(
            discrimination_table,
            outdir / "srcc_discrimination.tex",
        )
    if not decisiveness_table.empty:
        write_srcc_latex(
            decisiveness_table,
            outdir / "srcc_decisiveness.tex",
        )
    if not preference_table.empty:
        write_srcc_latex(
            preference_table,
            outdir / "srcc_preference_direction.tex",
        )

    direct_for_tex = direct_df.copy()
    direct_for_tex["metric"] = "Discrimination--decisiveness"
    write_srcc_latex(
        direct_for_tex,
        outdir / "srcc_discrimination_vs_decisiveness.tex",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------
    print("Discrimination vs decisiveness")
    print("------------------------------")
    print(
        f"rho = {direct_row['rho']:.3f}, "
        f"95% cluster-bootstrap CI "
        f"[{direct_row['ci95_lo']:.3f}, {direct_row['ci95_hi']:.3f}], "
        f"N = {direct_row['n_cells']}"
    )
    print()

    def print_metric_table(title, table):
        print(title)
        print("-" * len(title))
        if table.empty:
            print("No results.\n")
            return
        for _, r in table.iterrows():
            print(
                f"{pretty_metric(r['metric']):18s} "
                f"rho={r['rho']: .3f}  "
                f"CI=[{r['ci95_lo']: .3f}, {r['ci95_hi']: .3f}]  "
                f"N={int(r['n_cells'])}"
            )
        print()

    print_metric_table(
        "Metric separation vs discrimination",
        discrimination_table,
    )
    print_metric_table(
        "Metric separation vs decisiveness",
        decisiveness_table,
    )
    print_metric_table(
        "Metric direction vs preference direction",
        preference_table,
    )

    print(f"Results written to: {outdir.resolve()}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 3: relate utterance-wise objective metric differences "
            "to discrimination, decisiveness, and preference direction."
        )
    )

    parser.add_argument(
        "discrimination_csv",
        type=Path,
        help="Experiment 1 utterance-by-system-pair discrimination CSV.",
    )
    parser.add_argument(
        "preference_csv",
        type=Path,
        help="Experiment 2 utterance-by-system-pair preference CSV.",
    )
    parser.add_argument(
        "metrics_csv",
        type=Path,
        help="Objective metric system-pair CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("experiment3"),
        help="Output directory (default: experiment3).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=BOOTSTRAP_SAMPLES,
        help=f"Utterance-cluster bootstrap replicates (default: {BOOTSTRAP_SAMPLES}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BOOTSTRAP_SEED,
        help=f"Random seed (default: {BOOTSTRAP_SEED}).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_analysis(
        discrimination_path=args.discrimination_csv,
        preference_path=args.preference_csv,
        metric_path=args.metrics_csv,
        outdir=args.outdir,
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
