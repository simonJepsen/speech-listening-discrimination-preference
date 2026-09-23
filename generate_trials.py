#!/usr/bin/env python3
"""
Generate balanced participant schedules for a combined:

1. Triangle discrimination experiment
2. A/B preference experiment

The script:

- finds utterances available for every requested system;
- randomly selects --num_utterances utterances;
- assigns a subset of system pairs to each utterance;
- balances system-pair occurrence counts;
- repeats each utterance-pair item across independent listeners;
- creates participant-specific schedules;
- prevents an utterance from appearing twice within one block;
- prevents the same utterance-pair item from appearing in both blocks
  for the same participant when feasible;
- balances triangle presentation sequences;
- balances A/B presentation order;
- writes a schedule manifest for the web frontend.

Expected directory structure:

audio/
  DNS/
    clean/
      file_001.wav
      file_002.wav
    MA_DNS/
      file_001.wav
      file_002.wav
    MeanFlow_DNS/
      file_001.wav
      file_002.wav
    ...

Nested speaker directories are supported, provided the same relative
file structure exists for every system.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


TRIANGLE_SEQUENCES = (
    "AAB",
    "ABA",
    "BAA",
    "BBA",
    "BAB",
    "ABB",
)

PREFERENCE_ORDERS = (
    "AB",
    "BA",
)


@dataclass(frozen=True)
class Utterance:
    dataset: str
    relative_path: str

    @property
    def utterance_id(self) -> str:
        path = Path(self.relative_path)
        without_suffix = path.with_suffix("").as_posix()
        return f"{self.dataset}/{without_suffix}"

    @property
    def key(self) -> str:
        return f"{self.dataset}::{self.relative_path}"


@dataclass(frozen=True)
class TrialItem:
    item_id: str
    dataset: str
    relative_path: str
    utterance_id: str
    system_a: str
    system_b: str

    @property
    def utterance_key(self) -> str:
        return f"{self.dataset}::{self.relative_path}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate balanced participant-specific triangle and "
            "preference listening-test schedules."
        )
    )

    parser.add_argument(
        "--audio_root",
        type=Path,
        required=True,
        help="Root directory containing dataset/system audio directories.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset directory names beneath --audio_root.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        required=True,
        help="System directory names beneath each dataset directory.",
    )
    parser.add_argument(
        "--num_utterances",
        type=int,
        default=50,
        help=(
            "Number of common utterances to sample randomly across the "
            "requested datasets. Default: 50."
        ),
    )
    parser.add_argument(
        "--pairs_per_utterance",
        type=int,
        default=3,
        help=(
            "Number of distinct system pairs assigned to each selected "
            "utterance. Default: 3."
        ),
    )
    parser.add_argument(
        "--repetitions_per_item",
        type=int,
        default=10,
        help=(
            "Number of independent listener judgments required for every "
            "utterance-system-pair item in each block. Default: 10."
        ),
    )
    parser.add_argument(
        "--trials_per_block",
        type=int,
        default=30,
        help="Number of trials per participant per block. Default: 30.",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default=".wav",
        help="Audio extension to discover. Default: .wav.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("generated_trials"),
        help="Output directory. Default: generated_trials.",
    )
    parser.add_argument(
        "--allow_partial_last_schedule",
        action="store_true",
        help=(
            "Allow the final participant schedule to contain fewer than "
            "--trials_per_block trials."
        ),
    )
    parser.add_argument(
        "--max_assignment_attempts",
        type=int,
        default=1000,
        help="Maximum randomized schedule-assignment attempts. Default: 1000.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate --output_dir if it already exists.",
    )

    return parser.parse_args()


def normalize_extension(extension: str) -> str:
    extension = extension.strip()
    if not extension:
        raise ValueError("The audio extension cannot be empty.")

    if not extension.startswith("."):
        extension = f".{extension}"

    return extension.lower()


def validate_arguments(args: argparse.Namespace) -> None:
    if not args.audio_root.is_dir():
        raise FileNotFoundError(
            f"Audio root does not exist or is not a directory: "
            f"{args.audio_root}"
        )

    if len(args.systems) < 2:
        raise ValueError("At least two systems are required.")

    if len(set(args.systems)) != len(args.systems):
        raise ValueError("System names must be unique.")

    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Dataset names must be unique.")

    if args.num_utterances <= 0:
        raise ValueError("--num_utterances must be positive.")

    if args.pairs_per_utterance <= 0:
        raise ValueError("--pairs_per_utterance must be positive.")

    if args.repetitions_per_item <= 0:
        raise ValueError("--repetitions_per_item must be positive.")

    if args.trials_per_block <= 0:
        raise ValueError("--trials_per_block must be positive.")

    if args.max_assignment_attempts <= 0:
        raise ValueError("--max_assignment_attempts must be positive.")

    possible_pairs = math.comb(len(args.systems), 2)

    if args.pairs_per_utterance > possible_pairs:
        raise ValueError(
            f"Requested {args.pairs_per_utterance} pairs per utterance, "
            f"but {len(args.systems)} systems provide only "
            f"{possible_pairs} unique pairs."
        )

    # This restriction guarantees that an utterance can appear at most once
    # within a participant block.
    if args.trials_per_block > args.num_utterances:
        raise ValueError(
            "--trials_per_block cannot exceed --num_utterances because "
            "the generator prevents the same utterance from appearing "
            "twice in one participant block."
        )


def discover_relative_audio_paths(
    directory: Path,
    extension: str,
) -> set[str]:
    files = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() == extension
    }
    return files


def discover_common_utterances(
    audio_root: Path,
    datasets: Sequence[str],
    systems: Sequence[str],
    extension: str,
) -> list[Utterance]:
    all_common: list[Utterance] = []

    for dataset in datasets:
        system_file_sets: dict[str, set[str]] = {}

        for system in systems:
            system_dir = audio_root / dataset / system

            if not system_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing system directory: {system_dir}"
                )

            relative_paths = discover_relative_audio_paths(
                system_dir,
                extension,
            )

            if not relative_paths:
                raise RuntimeError(
                    f"No {extension} files found in: {system_dir}"
                )

            system_file_sets[system] = relative_paths

            print(
                f"Found {len(relative_paths):5d} files: "
                f"{dataset}/{system}"
            )

        common_paths = set.intersection(
            *(system_file_sets[system] for system in systems)
        )

        print(
            f"Common utterances for {dataset}: {len(common_paths)}"
        )

        if not common_paths:
            raise RuntimeError(
                f"No common utterances exist across all systems for "
                f"dataset '{dataset}'."
            )

        for relative_path in sorted(common_paths):
            all_common.append(
                Utterance(
                    dataset=dataset,
                    relative_path=relative_path,
                )
            )

    return all_common


def sample_utterances(
    utterances: Sequence[Utterance],
    num_utterances: int,
    rng: random.Random,
) -> list[Utterance]:
    if len(utterances) < num_utterances:
        raise ValueError(
            f"Requested {num_utterances} utterances, but only "
            f"{len(utterances)} common utterances were found."
        )

    selected = rng.sample(list(utterances), num_utterances)
    selected.sort(key=lambda item: (item.dataset, item.relative_path))
    return selected


def choose_balanced_pairs(
    utterances: Sequence[Utterance],
    systems: Sequence[str],
    pairs_per_utterance: int,
    rng: random.Random,
) -> list[TrialItem]:
    all_pairs = [
        tuple(sorted(pair))
        for pair in itertools.combinations(systems, 2)
    ]

    pair_counts: Counter[tuple[str, str]] = Counter()
    system_counts: Counter[str] = Counter()
    items: list[TrialItem] = []

    utterance_order = list(utterances)
    rng.shuffle(utterance_order)

    for utterance in utterance_order:
        # Add random values only to break ties reproducibly.
        pair_tiebreakers = {
            pair: rng.random()
            for pair in all_pairs
        }

        ranked_pairs = sorted(
            all_pairs,
            key=lambda pair: (
                pair_counts[pair],
                system_counts[pair[0]] + system_counts[pair[1]],
                pair_tiebreakers[pair],
            ),
        )

        selected_pairs = ranked_pairs[:pairs_per_utterance]

        for system_a, system_b in selected_pairs:
            pair_counts[(system_a, system_b)] += 1
            system_counts[system_a] += 1
            system_counts[system_b] += 1

            item_index = len(items) + 1
            item_id = f"item_{item_index:05d}"

            items.append(
                TrialItem(
                    item_id=item_id,
                    dataset=utterance.dataset,
                    relative_path=utterance.relative_path,
                    utterance_id=utterance.utterance_id,
                    system_a=system_a,
                    system_b=system_b,
                )
            )

    items.sort(key=lambda item: item.item_id)

    print("\nSelected unique system-pair counts:")

    for pair in sorted(all_pairs):
        print(
            f"  {pair[0]:20s} vs {pair[1]:20s}: "
            f"{pair_counts[pair]:4d}"
        )

    return items


def calculate_schedule_count(
    num_items: int,
    repetitions_per_item: int,
    trials_per_block: int,
    allow_partial: bool,
) -> int:
    total_assignments = num_items * repetitions_per_item

    if not allow_partial and total_assignments % trials_per_block != 0:
        raise ValueError(
            "\nThe requested design does not divide into complete schedules.\n"
            f"\nUnique items:             {num_items}"
            f"\nRepetitions per item:     {repetitions_per_item}"
            f"\nTotal trials per block:   {total_assignments}"
            f"\nTrials per participant:   {trials_per_block}"
            f"\nRemainder:                "
            f"{total_assignments % trials_per_block}\n"
            "\nChange one of the design parameters or pass "
            "--allow_partial_last_schedule."
        )

    return math.ceil(total_assignments / trials_per_block)


def build_assignment_copies(
    items: Sequence[TrialItem],
    repetitions_per_item: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    copies: list[dict[str, Any]] = []

    # Construct one randomized pass through all items per repetition.
    # This naturally separates repeated copies of the same item.
    for repetition in range(1, repetitions_per_item + 1):
        repetition_items = list(items)
        rng.shuffle(repetition_items)

        for item in repetition_items:
            copies.append(
                {
                    "item": item,
                    "repetition": repetition,
                }
            )

    return copies


def assign_items_to_schedules(
    items: Sequence[TrialItem],
    repetitions_per_item: int,
    trials_per_block: int,
    num_schedules: int,
    rng: random.Random,
    max_attempts: int,
    forbidden_item_ids_by_schedule: dict[int, set[str]] | None = None,
) -> list[list[dict[str, Any]]]:
    """
    Assign repeated items to participant schedules.

    Constraints:

    - no schedule exceeds trials_per_block;
    - no item appears twice in one schedule;
    - no utterance appears twice in one schedule;
    - optionally, an item can be forbidden from the corresponding schedule,
      e.g. because that participant already heard it in another block.
    """

    forbidden_item_ids_by_schedule = (
        forbidden_item_ids_by_schedule or {}
    )

    for attempt in range(1, max_attempts + 1):
        schedules: list[list[dict[str, Any]]] = [
            [] for _ in range(num_schedules)
        ]
        schedule_utterances: list[set[str]] = [
            set() for _ in range(num_schedules)
        ]
        schedule_items: list[set[str]] = [
            set() for _ in range(num_schedules)
        ]
        schedule_pairs: list[Counter[tuple[str, str]]] = [
            Counter() for _ in range(num_schedules)
        ]

        copies = build_assignment_copies(
            items,
            repetitions_per_item,
            rng,
        )

        success = True

        for copy in copies:
            item: TrialItem = copy["item"]
            pair = (item.system_a, item.system_b)

            candidates: list[int] = []

            for schedule_index in range(num_schedules):
                if len(schedules[schedule_index]) >= trials_per_block:
                    continue

                if item.item_id in schedule_items[schedule_index]:
                    continue

                if item.utterance_key in schedule_utterances[schedule_index]:
                    continue

                if item.item_id in forbidden_item_ids_by_schedule.get(
                    schedule_index,
                    set(),
                ):
                    continue

                candidates.append(schedule_index)

            if not candidates:
                success = False
                break

            # Prefer schedules with:
            # 1. fewer total trials;
            # 2. fewer occurrences of this pair;
            # 3. random tie-breaking.
            candidate_tiebreakers = {
                index: rng.random()
                for index in candidates
            }

            chosen = min(
                candidates,
                key=lambda index: (
                    len(schedules[index]),
                    schedule_pairs[index][pair],
                    candidate_tiebreakers[index],
                ),
            )

            schedules[chosen].append(copy)
            schedule_items[chosen].add(item.item_id)
            schedule_utterances[chosen].add(item.utterance_key)
            schedule_pairs[chosen][pair] += 1

        if success:
            expected_total = len(items) * repetitions_per_item
            assigned_total = sum(len(schedule) for schedule in schedules)

            if assigned_total != expected_total:
                continue

            print(
                f"Schedule assignment succeeded on attempt {attempt}."
            )
            return schedules

    raise RuntimeError(
        f"Could not construct valid participant schedules after "
        f"{max_attempts} attempts.\n"
        "Possible remedies:\n"
        "  - increase --num_utterances;\n"
        "  - reduce --trials_per_block;\n"
        "  - reduce --repetitions_per_item;\n"
        "  - allow overlap between the two blocks;\n"
        "  - increase --max_assignment_attempts."
    )


def make_audio_path(
    audio_root: Path,
    item: TrialItem,
    system: str,
) -> str:
    return (
        audio_root
        / item.dataset
        / system
        / item.relative_path
    ).as_posix()


def triangle_sequence_details(
    sequence: str,
    item: TrialItem,
    audio_root: Path,
) -> dict[str, Any]:
    if sequence not in TRIANGLE_SEQUENCES:
        raise ValueError(f"Invalid triangle sequence: {sequence}")

    paths = {
        "A": make_audio_path(audio_root, item, item.system_a),
        "B": make_audio_path(audio_root, item, item.system_b),
    }

    counts = Counter(sequence)
    odd_letter = min(counts, key=counts.get)
    repeated_letter = max(counts, key=counts.get)
    correct_position = sequence.index(odd_letter) + 1

    letter_to_system = {
        "A": item.system_a,
        "B": item.system_b,
    }

    return {
        "audio_1": paths[sequence[0]],
        "audio_2": paths[sequence[1]],
        "audio_3": paths[sequence[2]],
        "correct_position": correct_position,
        "odd_system": letter_to_system[odd_letter],
        "repeated_system": letter_to_system[repeated_letter],
    }


def create_triangle_rows(
    schedules: Sequence[Sequence[dict[str, Any]]],
    audio_root: Path,
    rng: random.Random,
) -> list[list[dict[str, Any]]]:
    all_rows: list[list[dict[str, Any]]] = []
    sequence_counter = 0

    # A shuffled cycle keeps the six triangle sequences balanced globally.
    sequence_cycle = list(TRIANGLE_SEQUENCES)
    rng.shuffle(sequence_cycle)

    for schedule_index, schedule in enumerate(schedules, start=1):
        rows: list[dict[str, Any]] = []

        schedule_entries = list(schedule)
        rng.shuffle(schedule_entries)

        for trial_number, assignment in enumerate(
            schedule_entries,
            start=1,
        ):
            item: TrialItem = assignment["item"]

            if sequence_counter > 0 and (
                sequence_counter % len(TRIANGLE_SEQUENCES) == 0
            ):
                rng.shuffle(sequence_cycle)

            sequence = sequence_cycle[
                sequence_counter % len(TRIANGLE_SEQUENCES)
            ]
            sequence_counter += 1

            details = triangle_sequence_details(
                sequence,
                item,
                audio_root,
            )

            rows.append(
                {
                    "trial_id": (
                        f"triangle_set_{schedule_index:03d}_"
                        f"trial_{trial_number:03d}"
                    ),
                    "schedule_id": f"set_{schedule_index:03d}",
                    "schedule_number": schedule_index,
                    "trial_number": trial_number,
                    "block": "triangle",
                    "item_id": item.item_id,
                    "repetition": assignment["repetition"],
                    "dataset": item.dataset,
                    "utterance_id": item.utterance_id,
                    "relative_path": item.relative_path,
                    "system_a": item.system_a,
                    "system_b": item.system_b,
                    "sequence": sequence,
                    "audio_1": details["audio_1"],
                    "audio_2": details["audio_2"],
                    "audio_3": details["audio_3"],
                    "correct_position": details["correct_position"],
                    "odd_system": details["odd_system"],
                    "repeated_system": details["repeated_system"],
                }
            )

        all_rows.append(rows)

    return all_rows


def create_preference_rows(
    schedules: Sequence[Sequence[dict[str, Any]]],
    audio_root: Path,
    rng: random.Random,
) -> list[list[dict[str, Any]]]:
    all_rows: list[list[dict[str, Any]]] = []
    order_counter = 0

    order_cycle = list(PREFERENCE_ORDERS)
    rng.shuffle(order_cycle)

    for schedule_index, schedule in enumerate(schedules, start=1):
        rows: list[dict[str, Any]] = []

        schedule_entries = list(schedule)
        rng.shuffle(schedule_entries)

        for trial_number, assignment in enumerate(
            schedule_entries,
            start=1,
        ):
            item: TrialItem = assignment["item"]

            if order_counter > 0 and (
                order_counter % len(PREFERENCE_ORDERS) == 0
            ):
                rng.shuffle(order_cycle)

            order = order_cycle[
                order_counter % len(PREFERENCE_ORDERS)
            ]
            order_counter += 1

            audio_a = make_audio_path(
                audio_root,
                item,
                item.system_a,
            )
            audio_b = make_audio_path(
                audio_root,
                item,
                item.system_b,
            )

            if order == "AB":
                audio_1 = audio_a
                audio_2 = audio_b
                left_system = item.system_a
                right_system = item.system_b
            else:
                audio_1 = audio_b
                audio_2 = audio_a
                left_system = item.system_b
                right_system = item.system_a

            rows.append(
                {
                    "trial_id": (
                        f"preference_set_{schedule_index:03d}_"
                        f"trial_{trial_number:03d}"
                    ),
                    "schedule_id": f"set_{schedule_index:03d}",
                    "schedule_number": schedule_index,
                    "trial_number": trial_number,
                    "block": "preference",
                    "item_id": item.item_id,
                    "repetition": assignment["repetition"],
                    "dataset": item.dataset,
                    "utterance_id": item.utterance_id,
                    "relative_path": item.relative_path,
                    "system_a": item.system_a,
                    "system_b": item.system_b,
                    "presentation_order": order,
                    "audio_a": audio_a,
                    "audio_b": audio_b,
                    "audio_1": audio_1,
                    "audio_2": audio_2,
                    "left_system": left_system,
                    "right_system": right_system,
                }
            )

        all_rows.append(rows)

    return all_rows


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)


def flatten(
    nested_rows: Sequence[Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        row
        for schedule_rows in nested_rows
        for row in schedule_rows
    ]


def prepare_output_directory(
    output_dir: Path,
    overwrite: bool,
) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Use --overwrite to replace it."
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)


def write_selected_utterances(
    output_dir: Path,
    utterances: Sequence[Utterance],
) -> None:
    rows = [
        {
            "dataset": utterance.dataset,
            "utterance_id": utterance.utterance_id,
            "relative_path": utterance.relative_path,
        }
        for utterance in utterances
    ]

    write_csv(
        output_dir / "selected_utterances.csv",
        rows,
        fieldnames=(
            "dataset",
            "utterance_id",
            "relative_path",
        ),
    )


def write_item_pool(
    output_dir: Path,
    items: Sequence[TrialItem],
) -> None:
    rows = [
        {
            "item_id": item.item_id,
            "dataset": item.dataset,
            "utterance_id": item.utterance_id,
            "relative_path": item.relative_path,
            "system_a": item.system_a,
            "system_b": item.system_b,
        }
        for item in items
    ]

    write_csv(
        output_dir / "selected_item_pool.csv",
        rows,
        fieldnames=(
            "item_id",
            "dataset",
            "utterance_id",
            "relative_path",
            "system_a",
            "system_b",
        ),
    )


def write_participant_schedules(
    output_dir: Path,
    triangle_rows: Sequence[Sequence[dict[str, Any]]],
    preference_rows: Sequence[Sequence[dict[str, Any]]],
) -> None:
    triangle_fieldnames = (
        "trial_id",
        "schedule_id",
        "schedule_number",
        "trial_number",
        "block",
        "item_id",
        "repetition",
        "dataset",
        "utterance_id",
        "relative_path",
        "system_a",
        "system_b",
        "sequence",
        "audio_1",
        "audio_2",
        "audio_3",
        "correct_position",
        "odd_system",
        "repeated_system",
    )

    preference_fieldnames = (
        "trial_id",
        "schedule_id",
        "schedule_number",
        "trial_number",
        "block",
        "item_id",
        "repetition",
        "dataset",
        "utterance_id",
        "relative_path",
        "system_a",
        "system_b",
        "presentation_order",
        "audio_a",
        "audio_b",
        "audio_1",
        "audio_2",
        "left_system",
        "right_system",
    )

    schedules_root = (
        output_dir
        / "participant_schedules"
    )

    for index, (triangle, preference) in enumerate(
        zip(triangle_rows, preference_rows),
        start=1,
    ):
        schedule_dir = schedules_root / f"set_{index:03d}"

        write_csv(
            schedule_dir / "triangle_trials.csv",
            triangle,
            triangle_fieldnames,
        )
        write_csv(
            schedule_dir / "preference_trials.csv",
            preference,
            preference_fieldnames,
        )

    write_csv(
        output_dir / "triangle_assignments_master.csv",
        flatten(triangle_rows),
        triangle_fieldnames,
    )
    write_csv(
        output_dir / "preference_assignments_master.csv",
        flatten(preference_rows),
        preference_fieldnames,
    )


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    num_common_utterances: int,
    num_selected_utterances: int,
    num_items: int,
    num_schedules: int,
    triangle_rows: Sequence[Sequence[dict[str, Any]]],
    preference_rows: Sequence[Sequence[dict[str, Any]]],
) -> None:
    manifest = {
        "experiment_version": "triangle-preference-v1",
        "seed": args.seed,
        "audio_root": args.audio_root.as_posix(),
        "datasets": list(args.datasets),
        "systems": list(args.systems),
        "extension": args.extension,
        "num_common_utterances_available": num_common_utterances,
        "num_selected_utterances": num_selected_utterances,
        "pairs_per_utterance": args.pairs_per_utterance,
        "num_unique_items": num_items,
        "repetitions_per_item_per_block": args.repetitions_per_item,
        "trials_per_block": args.trials_per_block,
        "trials_per_participant": (
            args.trials_per_block * 2
        ),
        "num_schedules": num_schedules,
        "allow_partial_last_schedule": (
            args.allow_partial_last_schedule
        ),
        "schedule_ids": [
            f"set_{index:03d}"
            for index in range(1, num_schedules + 1)
        ],
        "schedule_path_template": (
            "generated_trials/participant_schedules/"
            "set_{schedule_number:03d}"
        ),
        "triangle_schedule_sizes": [
            len(rows)
            for rows in triangle_rows
        ],
        "preference_schedule_sizes": [
            len(rows)
            for rows in preference_rows
        ],
    }

    with (
        output_dir / "schedule_manifest.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def verify_generated_design(
    items: Sequence[TrialItem],
    triangle_schedules: Sequence[Sequence[dict[str, Any]]],
    preference_schedules: Sequence[Sequence[dict[str, Any]]],
    repetitions_per_item: int,
    trials_per_block: int,
    allow_partial: bool,
) -> None:
    expected_item_ids = {item.item_id for item in items}

    for block_name, schedules in (
        ("triangle", triangle_schedules),
        ("preference", preference_schedules),
    ):
        item_counts: Counter[str] = Counter()

        for schedule_index, schedule in enumerate(schedules, start=1):
            if (
                not allow_partial
                and len(schedule) != trials_per_block
            ):
                raise AssertionError(
                    f"{block_name} schedule {schedule_index} contains "
                    f"{len(schedule)} trials instead of "
                    f"{trials_per_block}."
                )

            utterance_keys = [
                assignment["item"].utterance_key
                for assignment in schedule
            ]
            item_ids = [
                assignment["item"].item_id
                for assignment in schedule
            ]

            if len(utterance_keys) != len(set(utterance_keys)):
                raise AssertionError(
                    f"{block_name} schedule {schedule_index} repeats "
                    "an utterance."
                )

            if len(item_ids) != len(set(item_ids)):
                raise AssertionError(
                    f"{block_name} schedule {schedule_index} repeats "
                    "an utterance-pair item."
                )

            item_counts.update(item_ids)

        if set(item_counts) != expected_item_ids:
            raise AssertionError(
                f"{block_name} does not contain the expected item set."
            )

        invalid_counts = {
            item_id: count
            for item_id, count in item_counts.items()
            if count != repetitions_per_item
        }

        if invalid_counts:
            raise AssertionError(
                f"{block_name} has incorrect repetition counts: "
                f"{invalid_counts}"
            )

    # Ensure the exact same utterance-pair item is not presented in both
    # blocks to the same participant.
    for schedule_index, (
        triangle_schedule,
        preference_schedule,
    ) in enumerate(
        zip(triangle_schedules, preference_schedules),
        start=1,
    ):
        triangle_item_ids = {
            assignment["item"].item_id
            for assignment in triangle_schedule
        }
        preference_item_ids = {
            assignment["item"].item_id
            for assignment in preference_schedule
        }

        overlap = triangle_item_ids & preference_item_ids

        if overlap:
            raise AssertionError(
                f"Schedule {schedule_index} contains the same item in "
                f"both blocks: {sorted(overlap)}"
            )


def main() -> None:
    args = parse_args()
    args.extension = normalize_extension(args.extension)

    validate_arguments(args)

    rng = random.Random(args.seed)

    print("\nDiscovering common utterances...")

    common_utterances = discover_common_utterances(
        audio_root=args.audio_root,
        datasets=args.datasets,
        systems=args.systems,
        extension=args.extension,
    )

    print(
        f"\nTotal common utterances across requested datasets: "
        f"{len(common_utterances)}"
    )

    selected_utterances = sample_utterances(
        common_utterances,
        args.num_utterances,
        rng,
    )

    print(
        f"Randomly selected utterances: "
        f"{len(selected_utterances)}"
    )

    items = choose_balanced_pairs(
        selected_utterances,
        args.systems,
        args.pairs_per_utterance,
        rng,
    )

    num_schedules = calculate_schedule_count(
        num_items=len(items),
        repetitions_per_item=args.repetitions_per_item,
        trials_per_block=args.trials_per_block,
        allow_partial=args.allow_partial_last_schedule,
    )

    total_trials_per_block = (
        len(items) * args.repetitions_per_item
    )

    print("\nDesign summary")
    print("--------------")
    print(f"Selected utterances:             {len(selected_utterances)}")
    print(f"Pairs per utterance:             {args.pairs_per_utterance}")
    print(f"Unique utterance-pair items:     {len(items)}")
    print(f"Repetitions per item per block:  {args.repetitions_per_item}")
    print(f"Total trials per block:          {total_trials_per_block}")
    print(f"Trials per participant/block:    {args.trials_per_block}")
    print(f"Participant schedules required:  {num_schedules}")
    print(
        f"Total trials per participant:    "
        f"{args.trials_per_block * 2}"
    )

    if num_schedules < args.repetitions_per_item:
        raise ValueError(
            "There are fewer participant schedules than repetitions per "
            "item. The same item could not be assigned to distinct "
            "listeners. Increase the number of schedules."
        )

    print("\nAssigning triangle trials...")

    triangle_schedules = assign_items_to_schedules(
        items=items,
        repetitions_per_item=args.repetitions_per_item,
        trials_per_block=args.trials_per_block,
        num_schedules=num_schedules,
        rng=rng,
        max_attempts=args.max_assignment_attempts,
    )

    forbidden_preference_items: dict[int, set[str]] = {
        schedule_index: {
            assignment["item"].item_id
            for assignment in schedule
        }
        for schedule_index, schedule in enumerate(
            triangle_schedules
        )
    }

    print("\nAssigning preference trials...")

    preference_schedules = assign_items_to_schedules(
        items=items,
        repetitions_per_item=args.repetitions_per_item,
        trials_per_block=args.trials_per_block,
        num_schedules=num_schedules,
        rng=rng,
        max_attempts=args.max_assignment_attempts,
        forbidden_item_ids_by_schedule=forbidden_preference_items,
    )

    verify_generated_design(
        items=items,
        triangle_schedules=triangle_schedules,
        preference_schedules=preference_schedules,
        repetitions_per_item=args.repetitions_per_item,
        trials_per_block=args.trials_per_block,
        allow_partial=args.allow_partial_last_schedule,
    )

    triangle_rows = create_triangle_rows(
        schedules=triangle_schedules,
        audio_root=args.audio_root,
        rng=rng,
    )

    preference_rows = create_preference_rows(
        schedules=preference_schedules,
        audio_root=args.audio_root,
        rng=rng,
    )

    prepare_output_directory(
        args.output_dir,
        args.overwrite,
    )

    write_selected_utterances(
        args.output_dir,
        selected_utterances,
    )
    write_item_pool(
        args.output_dir,
        items,
    )
    write_participant_schedules(
        args.output_dir,
        triangle_rows,
        preference_rows,
    )
    write_manifest(
        output_dir=args.output_dir,
        args=args,
        num_common_utterances=len(common_utterances),
        num_selected_utterances=len(selected_utterances),
        num_items=len(items),
        num_schedules=num_schedules,
        triangle_rows=triangle_rows,
        preference_rows=preference_rows,
    )

    print("\nGeneration completed successfully.")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(
        "Manifest: "
        f"{(args.output_dir / 'schedule_manifest.json').resolve()}"
    )
    print(
        "Participant schedules: "
        f"{(args.output_dir / 'participant_schedules').resolve()}"
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)