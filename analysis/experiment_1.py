#!/usr/bin/env python3
"""
Triangle discrimination analysis.

For utterance u and system pair (A, B),

    p_hat^D_{u,A,B} = k_{u,A,B} / n_{u,A,B}

Figure 1
--------
x-axis:
    system pair

open circles:
    utterance-level discrimination rates

filled circle:
    unweighted mean across tested utterances

error bars:
    95% percentile bootstrap CI across utterances

Figure 2
--------
x-axis:
    utterance

open circles:
    discrimination rates for the available system pairs

filled circle:
    unweighted mean across available system pairs

Important plotting choice
-------------------------
There is NO horizontal jitter in either figure. All observations belonging
to one x-axis category are plotted on exactly the same vertical line.

Both figures are sorted from highest to lowest mean discrimination.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import binomtest, norm


# ============================================================
# Constants
# ============================================================

CHANCE_LEVEL = 1.0 / 3.0

BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 1234

# ICASSP full two-column width: approximately 175 mm.
# The figure is designed directly at its final paper width.
SINGLE_COLUMN_WIDTH = 3.35

# Slightly taller than before for improved readability,
# while preserving the same final paper width.
PAIR_FIGURE_HEIGHT = 1.6
UTTERANCE_FIGURE_HEIGHT = 1.6

# Marker sizes tuned for final printed size.
PAIR_POINT_SIZE = 10
UTTERANCE_POINT_SIZE = 8
PAIR_MEAN_MARKER_SIZE = 3.8
UTTERANCE_MEAN_MARKER_SIZE = 3.4


DISPLAY_NAMES = {
    "clean": "C",
    "MA_DNS": "MA",
    "MeanFlow_DNS": "MF",
    "mpsenet_DNS": "MP",
    "storm_DNS": "Sto",
}


# ============================================================
# Plot configuration
# ============================================================

def configure_plotting():
    """
    Configure publication-style figures.

    Notes
    -----
    * text.usetex=True gives genuine LaTeX-rendered text/math.
    * newtxtext/newtxmath provide Times-like vector fonts.
    * pdf.fonttype/ps.fonttype=42 prevent Matplotlib from producing
      Type 3 fonts for its own text output.
    """

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",

            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,

            "axes.linewidth": 0.8,

            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,

            # Vector font embedding.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,

            # Times-like LaTeX fonts.
            "text.latex.preamble": (
                r"\usepackage{newtxtext}"
                r"\usepackage{newtxmath}"
            ),
        }
    )


# ============================================================
# Basic helpers
# ============================================================

def display_name(system):
    system = str(system)

    if system in DISPLAY_NAMES:
        return DISPLAY_NAMES[system]

    return system.replace("_DNS", "")


def canonical_pair(a, b):
    a = str(a)
    b = str(b)

    if a <= b:
        return a, b

    return b, a


def pair_label(a, b):
    return f"{display_name(a)} vs {display_name(b)}"


def parse_bool(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(value, (int, np.integer)):
        return int(value != 0)

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return np.nan
        return int(value != 0)

    s = str(value).strip().lower()

    if s in {
        "true",
        "1",
        "yes",
        "y",
        "correct",
    }:
        return 1

    if s in {
        "false",
        "0",
        "no",
        "n",
        "incorrect",
    }:
        return 0

    raise ValueError(
        f"Could not interpret is_correct={value!r}"
    )


def short_utterance_label(utterance_id):
    s = str(utterance_id)

    s = s.split("/")[-1]

    if s.startswith("fileid_"):
        s = s[len("fileid_"):]

    return s


# ============================================================
# Wilson confidence interval
# ============================================================

def wilson_ci(k, n, alpha=0.05):
    if n <= 0:
        return np.nan, np.nan

    p = k / n
    z = norm.ppf(1 - alpha / 2)

    denominator = 1 + z**2 / n

    centre = (
        p + z**2 / (2 * n)
    ) / denominator

    half_width = (
        z
        * np.sqrt(
            p * (1 - p) / n
            + z**2 / (4 * n**2)
        )
        / denominator
    )

    return (
        max(0.0, centre - half_width),
        min(1.0, centre + half_width),
    )


# ============================================================
# Binomial summary
# ============================================================

def summarize_binomial(group):
    n = len(group)

    k = int(
        group["correct"].sum()
    )

    p_hat = k / n

    ci_lo, ci_hi = wilson_ci(
        k,
        n,
    )

    p_value = binomtest(
        k,
        n,
        p=CHANCE_LEVEL,
        alternative="greater",
    ).pvalue

    return pd.Series(
        {
            "n": n,
            "k": k,
            "p_hat_D": p_hat,
            "wilson95_lo": ci_lo,
            "wilson95_hi": ci_hi,
            "p_exact_one_sided": p_value,
        }
    )


# ============================================================
# Bootstrap CI across utterances
# ============================================================

def bootstrap_mean_ci(
    values,
    n_boot=BOOTSTRAP_SAMPLES,
    alpha=0.05,
    seed=BOOTSTRAP_SEED,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    estimate = float(
        np.mean(values)
    )

    if len(values) == 1:
        return estimate, estimate, estimate

    rng = np.random.default_rng(seed)

    boot = np.empty(
        n_boot,
        dtype=float,
    )

    for b in range(n_boot):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        boot[b] = np.mean(sample)

    lo = np.quantile(
        boot,
        alpha / 2,
    )

    hi = np.quantile(
        boot,
        1 - alpha / 2,
    )

    return estimate, float(lo), float(hi)


# ============================================================
# System-pair bootstrap summary
# ============================================================

def build_pair_bootstrap_summary(cells):
    rows = []

    for (
        system_a,
        system_b,
    ), group in cells.groupby(
        ["system_A", "system_B"],
        sort=False,
    ):

        values = group[
            "p_hat_D"
        ].to_numpy(
            dtype=float
        )

        mean_p, lo, hi = bootstrap_mean_ci(
            values
        )

        rows.append(
            {
                "system_A": system_a,
                "system_B": system_b,
                "n_utterances": len(values),
                "mean_p_hat_D": mean_p,
                "bootstrap95_lo": lo,
                "bootstrap95_hi": hi,
            }
        )

    summary = pd.DataFrame(
        rows
    )

    # Highest discrimination first.
    summary = (
        summary
        .sort_values(
            "mean_p_hat_D",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return summary


# ============================================================
# Full rectangular axes box
# ============================================================

def make_full_box(ax):
    for side in [
        "left",
        "right",
        "top",
        "bottom",
    ]:
        ax.spines[
            side
        ].set_visible(
            True
        )

        ax.spines[
            side
        ].set_linewidth(
            0.8
        )

        ax.spines[
            side
        ].set_color(
            "black"
        )

    ax.tick_params(
        axis="both",
        direction="out",
        color="black",
        labelcolor="black",
    )


# ============================================================
# Figure 1
# System pair vs discrimination
# ============================================================

def plot_system_pair_discrimination(
    cells,
    pair_summary,
    outdir,
):
    cells = cells.copy()
    pair_summary = pair_summary.copy()

    cells["pair_label"] = [
        pair_label(a, b)
        for a, b in zip(
            cells["system_A"],
            cells["system_B"],
        )
    ]

    pair_summary["pair_label"] = [
        pair_label(a, b)
        for a, b in zip(
            pair_summary["system_A"],
            pair_summary["system_B"],
        )
    ]

    # Sort by mean discrimination.
    pair_summary = (
        pair_summary
        .sort_values(
            "mean_p_hat_D",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    pair_order = pair_summary[
        "pair_label"
    ].tolist()

    fig, ax = plt.subplots(
        figsize=(
            SINGLE_COLUMN_WIDTH,
            PAIR_FIGURE_HEIGHT,
        )
    )

    for x, label in enumerate(
        pair_order
    ):
        values = (
            cells.loc[
                cells["pair_label"] == label,
                "p_hat_D",
            ]
            .to_numpy(
                dtype=float
            )
        )

        # ----------------------------------------------------
        # Individual utterance values
        #
        # IMPORTANT:
        # No horizontal jitter.
        # Every observation belonging to this pair lies on
        # exactly the same vertical line x = constant.
        # ----------------------------------------------------

        ax.scatter(
            np.full(
                len(values),
                x,
                dtype=float,
            ),
            values,

            s=PAIR_POINT_SIZE,

            facecolors="none",
            edgecolors="black",

            linewidths=0.55,
            alpha=0.42,

            zorder=20,
        )

        row = pair_summary.loc[
            pair_summary["pair_label"] == label
        ].iloc[0]

        mean_p = float(
            row["mean_p_hat_D"]
        )

        lo = float(
            row["bootstrap95_lo"]
        )

        hi = float(
            row["bootstrap95_hi"]
        )

        yerr = np.array(
            [
                [mean_p - lo],
                [hi - mean_p],
            ]
        )

        # Mean + 95% utterance-bootstrap CI.
        ax.errorbar(
            x,
            mean_p,

            yerr=yerr,

            fmt="o",

            color="black",
            ecolor="black",

            markerfacecolor="black",
            markeredgecolor="black",

            markersize=PAIR_MEAN_MARKER_SIZE,

            elinewidth=0.85,
            capsize=2.0,
            capthick=0.85,

            zorder=4,
        )

    # Chance level.
    ax.axhline(
        CHANCE_LEVEL,

        color="black",

        linestyle="--",
        linewidth=0.8,

        label=(
            r"Chance "
            r"($p_0=\frac{1}{3}$)"
        ),

        zorder=1,
    )

    ax.set_xticks(
        np.arange(
            len(pair_order)
        )
    )

    ax.set_xticklabels(
        pair_order,
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )

    ax.set_ylabel(
        r"\textrm{Discrimination}"
        r"$\hat{p}^{D}_{u,A,B}$"
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.set_yticks(
        np.arange(
            0,
            1.01,
            0.2,
        )
    )

    # Very light horizontal guides.
    ax.grid(
        axis="y",
        color="black",
        linewidth=0.4,
        alpha=0.12,
        zorder=0,
    )

    ax.margins(
        x=0.025
    )

    make_full_box(
        ax
    )

    ax.legend(
        frameon=False,
        loc="lower left",
    )

    fig.tight_layout(
        pad=0.12
    )

    fig.savefig(
        outdir
        / "triangle_discrimination_system_pairs.pdf"
    )

    fig.savefig(
        outdir
        / "triangle_discrimination_system_pairs.png",
        dpi=600,
    )

    plt.close(
        fig
    )


# ============================================================
# Figure 2
# Utterance vs discrimination
# ============================================================

def plot_utterance_discrimination(
    cells,
    outdir,
):
    cells = cells.copy()

    # Mean discrimination for every utterance.
    utterance_summary = (
        cells
        .groupby(
            "utterance_id"
        )["p_hat_D"]
        .agg(
            [
                "mean",
                "size",
            ]
        )
        .reset_index()
        .rename(
            columns={
                "mean": "mean_p_hat_D",
                "size": "n_pairs",
            }
        )
    )

    # Highest discrimination first.
    utterance_summary = (
        utterance_summary
        .sort_values(
            "mean_p_hat_D",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    utterance_order = utterance_summary[
        "utterance_id"
    ].tolist()

    fig, ax = plt.subplots(
        figsize=(
            SINGLE_COLUMN_WIDTH,
            UTTERANCE_FIGURE_HEIGHT,
        )
    )

    for x, utterance_id in enumerate(
        utterance_order
    ):
        values = (
            cells.loc[
                cells["utterance_id"]
                == utterance_id,
                "p_hat_D",
            ]
            .to_numpy(
                dtype=float
            )
        )

        # ----------------------------------------------------
        # Pair-wise values
        #
        # IMPORTANT:
        # No horizontal jitter.
        # Every pair-level observation belonging to this
        # utterance lies on exactly the same vertical line.
        # ----------------------------------------------------

        ax.scatter(
            np.full(
                len(values),
                x,
                dtype=float,
            ),
            values,

            s=UTTERANCE_POINT_SIZE,

            facecolors="none",
            edgecolors="black",

            linewidths=0.55,
            alpha=0.50,

            zorder=2,
        )

        # Mean across available system pairs.
        mean_value = float(
            np.mean(values)
        )

        ax.plot(
            x,
            mean_value,

            marker="o",
            linestyle="none",

            color="black",

            markerfacecolor="black",
            markeredgecolor="black",

            markersize=UTTERANCE_MEAN_MARKER_SIZE,

            zorder=4,
        )

    # Chance level.
    ax.axhline(
        CHANCE_LEVEL,

        color="black",

        linestyle="--",
        linewidth=0.8,

        label=(
            r"Chance "
            r"($p_0=\frac{1}{3}$)"
        ),

        zorder=1,
    )

    labels = [
        short_utterance_label(u)
        for u in utterance_order
    ]

    ax.set_xticks(
        np.arange(
            len(utterance_order)
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=90,
        ha="center",
        fontsize=5.0,
    )

    ax.set_xlabel(
        r"\textrm{Utterance}"
    )

    ax.set_ylabel(
        r"\textrm{Discrimination}"
        r"$\hat{p}^{D}_{u,A,B}$"
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.set_yticks(
        np.arange(
            0,
            1.01,
            0.2,
        )
    )

    ax.grid(
        axis="y",
        color="black",
        linewidth=0.4,
        alpha=0.12,
        zorder=0,
    )

    ax.margins(
        x=0.01
    )

    make_full_box(
        ax
    )

    ax.legend(
        frameon=False,
        loc="lower left",
    )

    fig.tight_layout(
        pad=0.18
    )

    fig.savefig(
        outdir
        / "triangle_discrimination_utterances.pdf"
    )

    fig.savefig(
        outdir
        / "triangle_discrimination_utterances.png",
        dpi=600,
    )

    plt.close(
        fig
    )

    return utterance_summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Triangle discrimination analysis."
        )
    )

    parser.add_argument(
        "triangle_csv",
        help="Triangle response CSV.",
    )

    parser.add_argument(
        "--outdir",
        default="triangle",
    )

    args = parser.parse_args()

    configure_plotting()

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------

    df = pd.read_csv(
        args.triangle_csv
    )

    required = [
        "participant_id",
        "utterance_id",
        "system_x",
        "system_y",
        "is_correct",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: "
            + ", ".join(
                missing
            )
        )

    # --------------------------------------------------------
    # Triangle trials only
    # --------------------------------------------------------

    if "task" in df.columns:
        df = df.loc[
            df["task"]
            .astype(str)
            .str.lower()
            .eq("triangle")
        ].copy()

    # --------------------------------------------------------
    # Valid rows
    # --------------------------------------------------------

    df = (
        df.dropna(
            subset=required
        )
        .copy()
    )

    # --------------------------------------------------------
    # Parse outcome
    # --------------------------------------------------------

    df["correct"] = (
        df["is_correct"]
        .map(
            parse_bool
        )
        .astype(
            int
        )
    )

    # --------------------------------------------------------
    # Canonical system pairs
    # --------------------------------------------------------

    pairs = [
        canonical_pair(a, b)
        for a, b in zip(
            df["system_x"],
            df["system_y"],
        )
    ]

    df["system_A"] = [
        p[0]
        for p in pairs
    ]

    df["system_B"] = [
        p[1]
        for p in pairs
    ]

    # --------------------------------------------------------
    # Utterance x pair cells
    # --------------------------------------------------------

    cells = (
        df.groupby(
            [
                "utterance_id",
                "system_A",
                "system_B",
            ],
            sort=False,
        )
        .apply(
            summarize_binomial
        )
        .reset_index()
    )

    cells.to_csv(
        outdir
        / "discrimination_utterance_pair.csv",
        index=False,
    )

    # --------------------------------------------------------
    # System-pair bootstrap summary
    # --------------------------------------------------------

    pair_summary = build_pair_bootstrap_summary(
        cells
    )

    pair_summary.to_csv(
        outdir
        / "discrimination_system_pair_bootstrap.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Formal pooled system-pair binomial tests
    # --------------------------------------------------------

    pair_binomial = (
        df.groupby(
            [
                "system_A",
                "system_B",
            ],
            sort=False,
        )
        .apply(
            summarize_binomial
        )
        .reset_index()
        .sort_values(
            "p_hat_D",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    pair_binomial.to_csv(
        outdir
        / "discrimination_system_pair_binomial.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    overall = (
        summarize_binomial(
            df
        )
        .to_frame()
        .T
    )

    overall.to_csv(
        outdir
        / "discrimination_overall.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Figure 1
    # --------------------------------------------------------

    plot_system_pair_discrimination(
        cells,
        pair_summary,
        outdir,
    )

    # --------------------------------------------------------
    # Figure 2
    # --------------------------------------------------------

    utterance_summary = (
        plot_utterance_discrimination(
            cells,
            outdir,
        )
    )

    utterance_summary.to_csv(
        outdir
        / "discrimination_utterance_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print(
        "\nSystem pairs sorted by performance:"
    )

    print(
        pair_summary[
            [
                "system_A",
                "system_B",
                "mean_p_hat_D",
                "bootstrap95_lo",
                "bootstrap95_hi",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nUtterances sorted by performance:"
    )

    print(
        utterance_summary[
            [
                "utterance_id",
                "mean_p_hat_D",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nCreated:"
    )

    print(
        outdir
        / "triangle_discrimination_system_pairs.pdf"
    )

    print(
        outdir
        / "triangle_discrimination_utterances.pdf"
    )


if __name__ == "__main__":
    main()
