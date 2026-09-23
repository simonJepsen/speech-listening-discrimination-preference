#!/usr/bin/env python3

"""
Experiment 2: Preference analysis.

Expected input columns
----------------------
id
created_at
participant_id
session_id
experiment_version
schedule_id
task
block_position
trial_id
dataset
utterance_id
system_x
system_y
system_a
system_b
presentation_order
selected_option
preferred_system
no_preference
response_time_ms
play_count_a
play_count_b
trial_index

Preference response
-------------------

For each trial,

    R in {A, B, N}

where N denotes "No preference".

Define

    Y_dec = I{R in {A,B}}

and

    Y_pref =
        +1, if R = A
         0, if R = N
        -1, if R = B

after canonicalizing the system-pair orientation.

For utterance u and system pair (A,B),

    p_hat_dec =
        (N_A + N_B) /
        (N_A + N_B + N_N)

and

    S_hat_pref =
        (N_A - N_B) /
        (N_A + N_B + N_N).

Figures
-------
1. System pair vs decisiveness
   - small points: utterance-wise decisiveness
   - large point: mean across utterances
   - error bar: 95% bootstrap CI across utterances
   - sorted from highest to lowest performance

2. Utterance vs decisiveness
   - small points: system-pair decisiveness values
   - large point: mean across available pairs
   - sorted from highest to lowest performance

3. System pair vs mean preference direction
   - small points: utterance-wise signed preference values
   - large point: mean across utterances
   - error bar: 95% bootstrap CI across utterances
   - sorted from highest to lowest signed preference

4. Utterance vs mean preference direction
   - small points: pair-wise preference-direction values
   - large point: mean across available pairs
   - sorted from highest to lowest signed preference

Plot styling
------------
- monochrome only
- full rectangular axes box
- true LaTeX rendering
- Times-style font
- 12 pt
- ICASSP full two-column width
- no horizontal jitter
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# ============================================================
# Constants
# ============================================================

DOUBLE_COLUMN_WIDTH = 3.35

PAIR_FIGURE_HEIGHT = 1.6
UTTERANCE_FIGURE_HEIGHT = 1.6

PAIR_POINT_SIZE = 10
UTTERANCE_POINT_SIZE = 8
PAIR_MEAN_MARKER_SIZE = 3.8
UTTERANCE_MEAN_MARKER_SIZE = 3.4

BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 1234


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

    plt.rcParams.update(
        {
            # ------------------------------------------------
            # True LaTeX
            # ------------------------------------------------
            "text.usetex": True,

            "font.family": "serif",

            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,

            # ------------------------------------------------
            # Axes
            # ------------------------------------------------
            "axes.linewidth": 0.8,

            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,

            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,

            # ------------------------------------------------
            # Vector text
            # ------------------------------------------------
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
# Utility functions
# ============================================================

def display_name(system):

    system = str(system)

    if system in DISPLAY_NAMES:
        return DISPLAY_NAMES[system]

    return system.replace(
        "_DNS",
        "",
    )


def canonical_pair(a, b):
    """
    Canonical system-pair orientation.

    Preference direction is defined relative to this orientation.
    """

    a = str(a)
    b = str(b)

    if a <= b:
        return a, b

    return b, a


def pair_label(a, b):

    return (
        f"{display_name(a)} vs "
        f"{display_name(b)}"
    )


def short_utterance_label(utterance_id):

    value = str(
        utterance_id
    )

    value = (
        value
        .split("/")[-1]
    )

    if value.startswith("fileid_"):

        value = value[
            len("fileid_"):
        ]

    return value


def parse_bool(value):

    if pd.isna(value):
        return np.nan

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    if isinstance(
        value,
        (int, np.integer),
    ):
        return bool(value)

    if isinstance(
        value,
        (float, np.floating),
    ):

        if np.isnan(value):
            return np.nan

        return bool(value)

    value = (
        str(value)
        .strip()
        .lower()
    )

    if value in {
        "true",
        "1",
        "yes",
        "y",
    }:
        return True

    if value in {
        "false",
        "0",
        "no",
        "n",
    }:
        return False

    raise ValueError(
        f"Could not interpret boolean value {value!r}"
    )


# ============================================================
# Parse one preference trial
# ============================================================

def parse_preference_row(row):
    """
    Convert raw row into canonical response coordinates.

    Uses:
        system_a
        system_b
        selected_option
        no_preference

    Returns
    -------
    system_A
    system_B
    response_canonical

    where response_canonical is one of

        A
        B
        N
    """

    raw_a = str(
        row["system_a"]
    )

    raw_b = str(
        row["system_b"]
    )

    no_preference = parse_bool(
        row["no_preference"]
    )

    # --------------------------------------------------------
    # Raw response in presented A/B orientation
    # --------------------------------------------------------

    if no_preference:

        raw_response = "N"

    else:

        selected = (
            str(
                row["selected_option"]
            )
            .strip()
            .upper()
        )

        if selected == "A":

            raw_response = "A"

        elif selected == "B":

            raw_response = "B"

        else:

            raise ValueError(
                "Expected selected_option to be A or B when "
                f"no_preference=False, got {selected!r}"
            )

    # --------------------------------------------------------
    # Canonical pair orientation
    # --------------------------------------------------------

    system_A, system_B = (
        canonical_pair(
            raw_a,
            raw_b,
        )
    )

    same_orientation = (
        raw_a == system_A
        and
        raw_b == system_B
    )

    # --------------------------------------------------------
    # Convert response to canonical orientation
    # --------------------------------------------------------

    if raw_response == "N":

        canonical_response = "N"

    elif same_orientation:

        canonical_response = raw_response

    else:

        if raw_response == "A":

            canonical_response = "B"

        else:

            canonical_response = "A"

    return (
        system_A,
        system_B,
        canonical_response,
    )


# ============================================================
# Consistency check
# ============================================================

def preferred_system_consistent(row):
    """
    Check that selected_option and preferred_system agree.

    No-preference rows are considered consistent without requiring
    preferred_system to contain a particular value.
    """

    no_preference = parse_bool(
        row["no_preference"]
    )

    if no_preference:
        return True

    selected = (
        str(
            row["selected_option"]
        )
        .strip()
        .upper()
    )

    if selected == "A":

        expected = str(
            row["system_a"]
        )

    elif selected == "B":

        expected = str(
            row["system_b"]
        )

    else:

        return False

    actual = str(
        row["preferred_system"]
    )

    return (
        expected == actual
    )


# ============================================================
# Bootstrap CI
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

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    estimate = float(
        np.mean(values)
    )

    if len(values) == 1:

        return (
            estimate,
            estimate,
            estimate,
        )

    rng = np.random.default_rng(
        seed
    )

    boot_means = np.empty(
        n_boot,
        dtype=float,
    )

    for b in range(
        n_boot
    ):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        boot_means[b] = np.mean(
            sample
        )

    lower = np.quantile(
        boot_means,
        alpha / 2
    )

    upper = np.quantile(
        boot_means,
        1 - alpha / 2
    )

    return (
        estimate,
        lower,
        upper,
    )


# ============================================================
# Preference summaries
# ============================================================

def summarize_preference_cell(group):
    """
    Summarize one utterance × system-pair cell.
    """

    n_a = int(
        (
            group[
                "response_canonical"
            ]
            == "A"
        ).sum()
    )

    n_b = int(
        (
            group[
                "response_canonical"
            ]
            == "B"
        ).sum()
    )

    n_n = int(
        (
            group[
                "response_canonical"
            ]
            == "N"
        ).sum()
    )

    n = (
        n_a
        + n_b
        + n_n
    )

    if n == 0:

        p_dec = np.nan
        s_pref = np.nan

    else:

        p_dec = (
            n_a + n_b
        ) / n

        s_pref = (
            n_a - n_b
        ) / n

    return pd.Series(
        {
            "n": n,

            "N_A": n_a,
            "N_B": n_b,
            "N_N": n_n,

            "p_hat_dec": p_dec,
            "S_hat_pref": s_pref,
        }
    )


# ============================================================
# Pair-level bootstrap summary
# ============================================================

def build_pair_summary(cells):

    rows = []

    for (
        system_a,
        system_b,
    ), group in cells.groupby(
        [
            "system_A",
            "system_B",
        ],
        sort=False,
    ):

        decisiveness = (
            group[
                "p_hat_dec"
            ]
            .to_numpy(
                dtype=float
            )
        )

        preference = (
            group[
                "S_hat_pref"
            ]
            .to_numpy(
                dtype=float
            )
        )

        (
            mean_dec,
            dec_lo,
            dec_hi,
        ) = bootstrap_mean_ci(
            decisiveness
        )

        (
            mean_pref,
            pref_lo,
            pref_hi,
        ) = bootstrap_mean_ci(
            preference
        )

        rows.append(
            {
                "system_A": system_a,
                "system_B": system_b,

                "n_utterances": len(group),

                "mean_p_hat_dec": mean_dec,

                "dec_bootstrap95_lo": dec_lo,
                "dec_bootstrap95_hi": dec_hi,

                "mean_S_hat_pref": mean_pref,

                "pref_bootstrap95_lo": pref_lo,
                "pref_bootstrap95_hi": pref_hi,
            }
        )

    return pd.DataFrame(
        rows
    )


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
        colors="black",
    )


# ============================================================
# System-pair plot
# ============================================================

def plot_system_pairs(
    cells,
    pair_summary,
    value_col,
    mean_col,
    ci_lo_col,
    ci_hi_col,
    ylabel,
    filename,
    outdir,
    y_min,
    y_max,
    reference_line=None,
    reference_label=None,
):
    """
    Small open circles:
        utterance-wise observations

    Filled circles:
        condition means

    Error bars:
        95% utterance-bootstrap CIs

    Sorted highest -> lowest.
    """

    cells = cells.copy()
    summary = pair_summary.copy()

    cells[
        "pair_label"
    ] = [
        pair_label(
            a,
            b,
        )
        for a, b in zip(
            cells[
                "system_A"
            ],
            cells[
                "system_B"
            ],
        )
    ]

    summary[
        "pair_label"
    ] = [
        pair_label(
            a,
            b,
        )
        for a, b in zip(
            summary[
                "system_A"
            ],
            summary[
                "system_B"
            ],
        )
    ]

    # ========================================================
    # Sort by performance
    # ========================================================

    summary = (
        summary
        .sort_values(
            mean_col,
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    order = (
        summary[
            "pair_label"
        ]
        .tolist()
    )

    fig, ax = plt.subplots(
        figsize=(
            DOUBLE_COLUMN_WIDTH,
            PAIR_FIGURE_HEIGHT,
        )
    )

    for x, label in enumerate(
        order
    ):

        values = (
            cells.loc[
                cells[
                    "pair_label"
                ]
                == label,
                value_col,
            ]
            .to_numpy(
                dtype=float
            )
        )

        # ----------------------------------------------------
        # Individual utterances
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

            linewidths=0.7,

            alpha=0.42,

            zorder=2,
        )

        row = (
            summary.loc[
                summary[
                    "pair_label"
                ]
                == label
            ]
            .iloc[0]
        )

        mean_value = float(
            row[
                mean_col
            ]
        )

        lo = float(
            row[
                ci_lo_col
            ]
        )

        hi = float(
            row[
                ci_hi_col
            ]
        )

        yerr = np.array(
            [
                [
                    mean_value - lo
                ],
                [
                    hi - mean_value
                ],
            ]
        )

        # ----------------------------------------------------
        # Mean + bootstrap CI
        # ----------------------------------------------------

        ax.errorbar(
            x,
            mean_value,

            yerr=yerr,

            fmt="o",

            color="black",
            ecolor="black",

            markerfacecolor="black",
            markeredgecolor="black",

            markersize=PAIR_MEAN_MARKER_SIZE,

            elinewidth=1.15,
            capsize=3.0,
            capthick=1.15,

            zorder=4,
        )

    # ========================================================
    # Reference line
    # ========================================================

    if reference_line is not None:

        ax.axhline(
            reference_line,

            color="black",

            linestyle="--",
            linewidth=1.1,

            label=reference_label,

            zorder=1,
        )

    # ========================================================
    # Axes
    # ========================================================

    ax.set_xticks(
        np.arange(
            len(order)
        )
    )

    ax.set_xticklabels(
        order,
        rotation=28,
        ha="right",
        rotation_mode="anchor",
    )


    ax.set_ylabel(
        ylabel
    )

    ax.set_ylim(
        y_min,
        y_max,
    )

    if y_min >= 0.0 and y_max <= 1.02:
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
        x=0.025
    )

    if y_min >= 0.0 and y_max <= 1.02:
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

    if reference_line is not None:

        ax.legend(
            frameon=False,
            loc="best",
        )

    fig.tight_layout(
        pad=0.4
    )

    fig.savefig(
        outdir
        / f"{filename}.pdf"
    )

    fig.savefig(
        outdir
        / f"{filename}.png",
        dpi=600,
    )

    plt.close(
        fig
    )


# ============================================================
# Utterance plot
# ============================================================

def plot_utterances(
    cells,
    value_col,
    mean_name,
    ylabel,
    filename,
    outdir,
    y_min,
    y_max,
    reference_line=None,
    reference_label=None,
):
    """
    Small open circles:
        system-pair observations for an utterance

    Filled circle:
        mean across available system pairs

    Sorted highest -> lowest.
    """

    summary = (
        cells
        .groupby(
            "utterance_id"
        )[
            value_col
        ]
        .agg(
            [
                "mean",
                "size",
            ]
        )
        .reset_index()
        .rename(
            columns={
                "mean": mean_name,
                "size": "n_system_pairs",
            }
        )
    )

    # ========================================================
    # Sort by performance
    # ========================================================

    summary = (
        summary
        .sort_values(
            mean_name,
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    order = (
        summary[
            "utterance_id"
        ]
        .tolist()
    )

    fig, ax = plt.subplots(
        figsize=(
            DOUBLE_COLUMN_WIDTH,
            UTTERANCE_FIGURE_HEIGHT,
        )
    )

    for x, utterance_id in enumerate(
        order
    ):

        values = (
            cells.loc[
                cells[
                    "utterance_id"
                ]
                == utterance_id,
                value_col,
            ]
            .to_numpy(
                dtype=float
            )
        )

        # ----------------------------------------------------
        # Pair-level values
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

            linewidths=0.75,

            alpha=0.50,

            zorder=2,
        )

        # ----------------------------------------------------
        # Utterance mean
        # ----------------------------------------------------

        ax.plot(
            x,
            float(
                np.mean(
                    values
                )
            ),

            marker="o",
            linestyle="none",

            color="black",

            markerfacecolor="black",
            markeredgecolor="black",

            markersize=UTTERANCE_MEAN_MARKER_SIZE,

            zorder=4,
        )

    # ========================================================
    # Reference line
    # ========================================================

    if reference_line is not None:

        ax.axhline(
            reference_line,

            color="black",

            linestyle="--",
            linewidth=1.1,

            label=reference_label,

            zorder=1,
        )

    labels = [
        short_utterance_label(
            u
        )
        for u in order
    ]

    ax.set_xticks(
        np.arange(
            len(order)
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=90,
        ha="center",
        fontsize=7.5,
    )

    ax.set_xlabel(
        r"\textrm{Utterance}"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_ylim(
        y_min,
        y_max,
    )

    make_full_box(
        ax
    )

    if reference_line is not None:

        ax.legend(
            frameon=False,
            loc="best",
        )

    fig.tight_layout(
        pad=0.4
    )

    fig.savefig(
        outdir
        / f"{filename}.pdf"
    )

    fig.savefig(
        outdir
        / f"{filename}.png",
        dpi=600,
    )

    plt.close(
        fig
    )

    return summary


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Experiment 2: preference analysis."
        )
    )

    parser.add_argument(
        "preference_csv",
        help=(
            "CSV containing preference responses."
        ),
    )

    parser.add_argument(
        "--outdir",
        default="preference",
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

    # ========================================================
    # Load CSV
    # ========================================================

    df = pd.read_csv(
        args.preference_csv
    )

    required_columns = [
        "participant_id",
        "utterance_id",
        "system_a",
        "system_b",
        "selected_option",
        "no_preference",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                missing_columns
            )
        )

    # ========================================================
    # Preference trials only
    # ========================================================

    if "task" in df.columns:

        df = (
            df.loc[
                df[
                    "task"
                ]
                .astype(str)
                .str.lower()
                .eq(
                    "preference"
                )
            ]
            .copy()
        )

    # ========================================================
    # Remove incomplete core rows
    # ========================================================

    df = (
        df.dropna(
            subset=[
                "participant_id",
                "utterance_id",
                "system_a",
                "system_b",
                "no_preference",
            ]
        )
        .copy()
    )

    # ========================================================
    # Parse no_preference first
    # ========================================================

    df[
        "no_preference_bool"
    ] = (
        df[
            "no_preference"
        ]
        .map(
            parse_bool
        )
    )

    # ========================================================
    # selected_option may legitimately be empty for N trials.
    #
    # Only require it when no_preference=False.
    # ========================================================

    missing_selection = (
        (~df["no_preference_bool"])
        &
        (
            df["selected_option"].isna()
            |
            df["selected_option"]
            .astype(str)
            .str.strip()
            .eq("")
        )
    )

    if missing_selection.any():

        bad_rows = df.loc[
            missing_selection,
            [
                "participant_id",
                "trial_id"
                if "trial_id" in df.columns
                else "utterance_id",
                "system_a",
                "system_b",
                "selected_option",
                "no_preference",
            ],
        ]

        raise ValueError(
            "Found decisive preference trials with no "
            "selected_option:\n"
            + bad_rows.to_string(
                index=False
            )
        )

    # ========================================================
    # Optional internal consistency check
    # ========================================================

    if "preferred_system" in df.columns:

        check_mask = (
            ~df[
                "no_preference_bool"
            ]
            &
            df[
                "preferred_system"
            ].notna()
        )

        checked = df.loc[
            check_mask
        ].copy()

        if len(checked) > 0:

            consistency = (
                checked.apply(
                    preferred_system_consistent,
                    axis=1,
                )
            )

            n_bad = int(
                (~consistency).sum()
            )

            if n_bad > 0:

                print(
                    f"WARNING: {n_bad} rows have inconsistent "
                    "selected_option and preferred_system."
                )

            else:

                print(
                    "Preference consistency check passed."
                )

    # ========================================================
    # Canonicalize pair and response
    # ========================================================

    canonical = [
        parse_preference_row(
            row
        )
        for _, row in df.iterrows()
    ]

    df[
        "system_A"
    ] = [
        item[0]
        for item in canonical
    ]

    df[
        "system_B"
    ] = [
        item[1]
        for item in canonical
    ]

    df[
        "response_canonical"
    ] = [
        item[2]
        for item in canonical
    ]

    # ========================================================
    # Trial-level analysis variables
    # ========================================================

    df[
        "Y_dec"
    ] = (
        df[
            "response_canonical"
        ]
        .isin(
            [
                "A",
                "B",
            ]
        )
        .astype(
            int
        )
    )

    df[
        "Y_pref"
    ] = (
        df[
            "response_canonical"
        ]
        .map(
            {
                "A": +1,
                "N": 0,
                "B": -1,
            }
        )
        .astype(
            int
        )
    )

    # ========================================================
    # Save parsed trial-level responses
    # ========================================================

    df.to_csv(
        outdir
        / "preference_trial_level_parsed.csv",
        index=False,
    )

    # ========================================================
    # Utterance × system-pair cells
    # ========================================================

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
            summarize_preference_cell
        )
        .reset_index()
    )

    cells.to_csv(
        outdir
        / "preference_utterance_pair.csv",
        index=False,
    )

    # ========================================================
    # Pair-level utterance bootstrap
    # ========================================================

    pair_summary = (
        build_pair_summary(
            cells
        )
    )

    pair_summary.to_csv(
        outdir
        / "preference_system_pair_bootstrap.csv",
        index=False,
    )

    # ========================================================
    # Pooled pair-level descriptive summary
    # ========================================================

    pair_pooled = (
        df.groupby(
            [
                "system_A",
                "system_B",
            ],
            sort=False,
        )
        .apply(
            summarize_preference_cell
        )
        .reset_index()
    )

    pair_pooled.to_csv(
        outdir
        / "preference_system_pair_pooled.csv",
        index=False,
    )

    # ========================================================
    # Overall preference summary
    # ========================================================

    n_a = int(
        (
            df[
                "response_canonical"
            ]
            == "A"
        ).sum()
    )

    n_b = int(
        (
            df[
                "response_canonical"
            ]
            == "B"
        ).sum()
    )

    n_n = int(
        (
            df[
                "response_canonical"
            ]
            == "N"
        ).sum()
    )

    n_total = (
        n_a
        + n_b
        + n_n
    )

    overall = pd.DataFrame(
        [
            {
                "n": n_total,

                "N_A": n_a,
                "N_B": n_b,
                "N_N": n_n,

                "p_hat_dec":
                    (
                        n_a
                        + n_b
                    )
                    / n_total,

                "S_hat_pref":
                    (
                        n_a
                        - n_b
                    )
                    / n_total,
            }
        ]
    )

    overall.to_csv(
        outdir
        / "preference_overall.csv",
        index=False,
    )

    # ========================================================
    # FIGURE 1:
    # System pair vs decisiveness
    # ========================================================

    plot_system_pairs(
        cells=cells,

        pair_summary=pair_summary,

        value_col="p_hat_dec",
        mean_col="mean_p_hat_dec",

        ci_lo_col="dec_bootstrap95_lo",
        ci_hi_col="dec_bootstrap95_hi",

        ylabel=(
            r"\textrm{Decisiveness }"
            r"$\hat{p}^{\mathrm{dec}}_{u,A,B}$"
        ),

        filename=(
            "preference_decisiveness_system_pairs"
        ),

        outdir=outdir,

        y_min=0.0,
        y_max=1.02,

        reference_line=None,
    )

    # ========================================================
    # FIGURE 2:
    # Utterance vs decisiveness
    # ========================================================

    utterance_dec_summary = (
        plot_utterances(
            cells=cells,

            value_col="p_hat_dec",

            mean_name="mean_p_hat_dec",

            ylabel=(
                r"\textrm{Decisiveness }"
                r"$\hat{p}^{\mathrm{dec}}_{u,A,B}$"
            ),

            filename=(
                "preference_decisiveness_utterances"
            ),

            outdir=outdir,

            y_min=0.0,
            y_max=1.02,

            reference_line=None,
        )
    )

    utterance_dec_summary.to_csv(
        outdir
        / "preference_utterance_decisiveness_summary.csv",
        index=False,
    )

    # ========================================================
    # FIGURE 3:
    # System pair vs preference direction
    # ========================================================

    plot_system_pairs(
        cells=cells,

        pair_summary=pair_summary,

        value_col="S_hat_pref",
        mean_col="mean_S_hat_pref",

        ci_lo_col="pref_bootstrap95_lo",
        ci_hi_col="pref_bootstrap95_hi",

        ylabel=(
            r"\textrm{Preference}"
            r"$\hat{S}_{u,A,B}$"
        ),

        filename=(
            "preference_direction_system_pairs"
        ),

        outdir=outdir,

        y_min=-1.02,
        y_max=1.02,

        reference_line=0.0,

        reference_label=(
            r"\textrm{No directional preference }"
            r"($S=0$)"
        ),
    )

    # ========================================================
    # FIGURE 4:
    # Utterance vs preference direction
    # ========================================================

    utterance_pref_summary = (
        plot_utterances(
            cells=cells,

            value_col="S_hat_pref",

            mean_name="mean_S_hat_pref",

            ylabel=(
                r"\textrm{Mean preference direction }"
                r"$\hat{S}_{u,A,B}$"
            ),

            filename=(
                "preference_direction_utterances"
            ),

            outdir=outdir,

            y_min=-1.02,
            y_max=1.02,

            reference_line=0.0,

            reference_label=(
                r"\textrm{No directional preference }"
                r"($S=0$)"
            ),
        )
    )

    utterance_pref_summary.to_csv(
        outdir
        / "preference_utterance_direction_summary.csv",
        index=False,
    )

    # ========================================================
    # Console summaries
    # ========================================================

    print(
        "\nOverall preference"
    )

    print(
        overall.to_string(
            index=False
        )
    )

    print(
        "\nSystem pairs sorted by decisiveness"
    )

    print(
        pair_summary[
            [
                "system_A",
                "system_B",
                "mean_p_hat_dec",
                "dec_bootstrap95_lo",
                "dec_bootstrap95_hi",
            ]
        ]
        .sort_values(
            "mean_p_hat_dec",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nSystem pairs sorted by preference direction"
    )

    print(
        pair_summary[
            [
                "system_A",
                "system_B",
                "mean_S_hat_pref",
                "pref_bootstrap95_lo",
                "pref_bootstrap95_hi",
            ]
        ]
        .sort_values(
            "mean_S_hat_pref",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nDataset"
    )

    print(
        f"Participants: "
        f"{df['participant_id'].nunique()}"
    )

    print(
        f"Utterances:   "
        f"{df['utterance_id'].nunique()}"
    )

    print(
        f"System pairs: "
        f"{pair_summary.shape[0]}"
    )

    print(
        f"Valid trials: "
        f"{len(df)}"
    )

    print(
        f"Decisive:     "
        f"{int(df['Y_dec'].sum())}"
    )

    print(
        f"No preference: "
        f"{int((df['Y_dec'] == 0).sum())}"
    )

    print(
        "\nFigures:"
    )

    for name in [
        "preference_decisiveness_system_pairs.pdf",
        "preference_decisiveness_utterances.pdf",
        "preference_direction_system_pairs.pdf",
        "preference_direction_utterances.pdf",
    ]:

        print(
            outdir / name
        )


if __name__ == "__main__":
    main()