#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (C) 2026 Fang-Ju Sun and contributors
"""
Step 3 -- Cohort selection and construction of the model datasets.

Purpose
-------
This script reproduces Online Supplemental Figure 1. It applies the cohort
exclusion criteria to the labelled ICU cohort produced by steps 1 and 2, creates
one shared patient-level 80/20 train/internal-test split, and writes a flowchart
summary whose rows correspond to the boxes in the figure.

Important revisions in this version
-----------------------------------
1. ICU stay duration uses the published rule exactly:
       excluded_short_stay = ICU_stay_hours < 16
   Stays of exactly 16.000 hours are not excluded by this rule.

2. The sepsis and infection model datasets share one patient-level split,
   stratified on outcome so that the sepsis and infection prevalences are
   preserved between the development and the internal test cohort.

   Each patient is assigned to one of four strata formed by the joint outcome --
   (no sepsis, no infection), (no sepsis, infection), (sepsis, no infection),
   (sepsis, infection) -- and the test fraction is drawn inside each stratum.
   Crossing the two outcomes is what preserves both prevalences at once;
   stratifying on sepsis alone would leave the infection prevalence to chance.

   Stratification is at the patient level, because the split is at the patient
   level. Patients contribute unequal numbers of ICU stays, so the stay-level
   prevalences end up close but not identical. The achieved prevalences at both
   levels are written to `split_prevalence_check.csv`.

Outputs
-------
<outdir>/flowchart_summary.csv
<outdir>/excluded_stage1.csv
<outdir>/excluded_stage2.csv
<outdir>/study_cohort.csv
<outdir>/train_cohort.csv
<outdir>/test_cohort.csv
<outdir>/run_config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Canonical column names
# --------------------------------------------------------------------------- #

PATIENT_ID = "Pno"
STAY_ID = "adm_ICU_id"
STAY_KEY = [PATIENT_ID, STAY_ID]

ADMISSION_TIME = "ICU_admdatetime"
DISCHARGE_TIME = "ICU_disdatetime"

SEPSIS_LABEL = "label"                 # 1 = sepsis
INFECTION_LABEL = "infection"          # 1 = culture-confirmed infection
INFECTION_LABEL_ALIASES = ["culture_report"]

SEPSIS_ONSET = "sepsis_onset"
INFECTION_ONSET = "infection_onset"

AGE = "Age"
DEFAULT_VITAL_COLS = ["Temperature", "Pulse", "Respiration"]

LOGGER = logging.getLogger("cohort_selection")


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def load_table(spec: str, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Load a table from a delimited file or from a SQL query file."""
    if spec.startswith("sql:"):
        frame = _load_from_sql(Path(spec[4:]))
    else:
        frame = _load_from_file(Path(spec))

    frame.columns = [str(c).strip() for c in frame.columns]
    if column_map:
        renamed = {raw: canon for raw, canon in column_map.items() if raw in frame.columns}
        if renamed:
            LOGGER.info("column map applied: %s", renamed)
        frame = frame.rename(columns=renamed)

    frame = normalise_infection_label_alias(frame)
    return frame


def _load_from_file(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"input file not found: {path}")
    separator = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    return pd.read_csv(path, sep=separator, dtype=str, encoding="utf-8-sig")


def _load_from_sql(query_path: Path) -> pd.DataFrame:
    if not query_path.is_file():
        raise FileNotFoundError(f"SQL file not found: {query_path}")
    db_url = os.environ.get("DB_URL")
    if not db_url:
        raise ValueError(
            "SQL source requested but the DB_URL environment variable is not set "
            '(set DB_URL to a valid SQLAlchemy database URL in the local environment)'
        )
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:  # pragma: no cover
        raise ValueError("SQL source requires SQLAlchemy (pip install sqlalchemy oracledb)") from exc

    query = query_path.read_text(encoding="utf-8")
    LOGGER.info("executing %s against the configured database", query_path)
    engine = create_engine(db_url)
    with engine.connect() as connection:
        frame = pd.read_sql(text(query), connection)
    return frame.astype("object").where(frame.notna(), None).astype("string")


def normalise_infection_label_alias(df: pd.DataFrame) -> pd.DataFrame:
    """Accept either `infection` or the older split-script name `culture_report`."""
    if INFECTION_LABEL in df.columns:
        return df

    for alias in INFECTION_LABEL_ALIASES:
        if alias in df.columns:
            LOGGER.info("renaming infection label alias '%s' to '%s'", alias, INFECTION_LABEL)
            return df.rename(columns={alias: INFECTION_LABEL})

    return df


def to_datetime(df: pd.DataFrame, columns: list[str], required: bool = True) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            if required:
                raise KeyError(f"expected datetime column '{col}' is missing")
            continue
        original_na = df[col].isna()
        parsed = pd.to_datetime(df[col], errors="coerce")
        n_bad = int((parsed.isna() & ~original_na).sum())
        if n_bad > 0:
            LOGGER.warning("%s: %d value(s) could not be parsed as a datetime", col, n_bad)
        df[col] = parsed
    return df


def normalise_id(series: pd.Series) -> pd.Series:
    """Strip whitespace and a trailing .0 left by a float round-trip."""
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"{source} is missing required column(s): {missing}. "
            "Use --column-map RAW=CANONICAL to remap them."
        )


# --------------------------------------------------------------------------- #
# Cohort assembly
# --------------------------------------------------------------------------- #

def collapse_to_stays(df: pd.DataFrame, vital_cols: list[str], missing_rule: str) -> pd.DataFrame:
    """
    Return one row per ICU stay.

    missing_rule="any": a stay is incomplete if any row lacks the measurement.
    missing_rule="all": a stay is incomplete only if every row lacks it.
    """
    rows_per_stay = df.groupby(STAY_KEY, sort=False).size()
    LOGGER.info(
        "input granularity: %d row(s) for %d ICU stay(s) (median %.0f rows per stay)",
        len(df), len(rows_per_stay), rows_per_stay.median(),
    )
    if rows_per_stay.max() > 1:
        LOGGER.warning(
            "the input holds up to %d rows per ICU stay; vital-sign completeness is "
            "aggregated with --missing-rule %s",
            int(rows_per_stay.max()), missing_rule,
        )

    aggregations: dict[str, tuple[str, str]] = {}
    for col in df.columns:
        if col in STAY_KEY or col in vital_cols:
            continue
        aggregations[col] = (col, "first")

    stays = df.groupby(STAY_KEY, as_index=False).agg(**aggregations)

    for col in vital_cols:
        if col not in df.columns:
            LOGGER.warning("vital sign column '%s' is absent and cannot be checked", col)
            continue

        if missing_rule == "any":
            present = df.groupby(STAY_KEY, sort=False)[col].apply(lambda s: s.notna().all())
        else:
            present = df.groupby(STAY_KEY, sort=False)[col].apply(lambda s: s.notna().any())

        present = present.reset_index(name=f"{col}_present")
        stays = stays.merge(present, on=STAY_KEY, how="left")
        stays[f"{col}_missing"] = (~stays[f"{col}_present"].fillna(False)).astype(int)
        stays = stays.drop(columns=[f"{col}_present"])

    return stays


def apply_cohort_entry(stays: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Apply the study-period and age criteria that define the top flowchart box."""
    keep = pd.Series(True, index=stays.index)

    if args.start_date:
        keep &= stays[ADMISSION_TIME] >= pd.Timestamp(args.start_date)
    if args.end_date:
        keep &= stays[ADMISSION_TIME] <= pd.Timestamp(args.end_date)
    if args.min_age is not None:
        if AGE in stays.columns:
            age = pd.to_numeric(stays[AGE], errors="coerce")
            keep &= age >= args.min_age
        else:
            LOGGER.warning("--min-age given but column '%s' is absent; filter skipped", AGE)

    n_dropped = int((~keep).sum())
    if n_dropped:
        LOGGER.info("cohort entry criteria excluded %d ICU stay(s)", n_dropped)
    else:
        LOGGER.info("cohort entry criteria excluded no stays (already applied upstream)")
    return stays[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stage 1 -- ICU stay < 16 h and incomplete vital signs
# --------------------------------------------------------------------------- #

def apply_stage1(
    stays: pd.DataFrame,
    vital_cols: list[str],
    min_hours: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Flag short stays and stays with incomplete vital signs.

    The published figure says ICU stay < 16 hours. Therefore the rule here is
    strictly `< min_hours`, not `<= min_hours`.
    """
    stays = stays.copy()
    duration = (stays[DISCHARGE_TIME] - stays[ADMISSION_TIME]).dt.total_seconds() / 3600.0
    stays["ICU_stay_hours"] = duration

    n_invalid = int((duration < 0).sum())
    if n_invalid:
        LOGGER.warning("%d ICU stay(s) have a discharge time before admission", n_invalid)

    n_missing_time = int(duration.isna().sum())
    if n_missing_time:
        LOGGER.warning(
            "%d ICU stay(s) have no computable duration and are treated as short",
            n_missing_time,
        )

    short = (duration < min_hours) | duration.isna()
    stays["excluded_short_stay"] = short.astype(int)

    n_on_boundary = int((duration == min_hours).sum())
    LOGGER.info(
        "stage 1: %d stay(s) of exactly %.1f h; these are NOT excluded by the < %.1f h rule",
        n_on_boundary, min_hours, min_hours,
    )

    missing_cols = [f"{c}_missing" for c in vital_cols if f"{c}_missing" in stays.columns]
    if missing_cols:
        vital_missing = stays[missing_cols].max(axis=1).astype(bool)
    else:
        LOGGER.warning("no vital-sign columns available; that criterion is skipped")
        vital_missing = pd.Series(False, index=stays.index)

    stays["excluded_vital_missing"] = vital_missing.astype(int)

    excluded_mask = short | vital_missing
    counts = {
        "n_short_stay": int(short.sum()),
        "n_vital_missing_any": int(vital_missing.sum()),
        "n_overlap": int((short & vital_missing).sum()),
        "n_excluded": int(excluded_mask.sum()),
    }
    for col in vital_cols:
        flag = f"{col}_missing"
        if flag in stays.columns:
            counts[f"n_missing_{col}"] = int(stays[flag].sum())

    assert counts["n_excluded"] == (
        counts["n_short_stay"] + counts["n_vital_missing_any"] - counts["n_overlap"]
    ), "stage 1 counts do not reconcile"

    return (
        stays[~excluded_mask].reset_index(drop=True),
        stays[excluded_mask].reset_index(drop=True),
        counts,
    )


# --------------------------------------------------------------------------- #
# Stage 2 -- onset too close to ICU admission
# --------------------------------------------------------------------------- #

def onset_in_window(
    onset: pd.Series,
    admission: pd.Series,
    window: str,
    hours: float,
) -> pd.Series:
    """Return True where `onset` falls inside the exclusion window."""
    delta = pd.Timedelta(hours=hours)

    if window == "before":
        return onset.between(admission - delta, admission, inclusive="both")
    if window == "after":
        return onset.between(admission, admission + delta, inclusive="right")
    if window == "both":
        return onset.between(admission - delta, admission + delta, inclusive="both")
    if window == "up-to":
        return onset <= (admission + delta)

    raise ValueError(f"unknown --onset-window: {window}")


def apply_stage2(
    stays: pd.DataFrame,
    window: str,
    hours: float,
    sepsis_onset_col: str,
    infection_onset_col: str,
    floor_admission: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Exclude labelled stays whose sepsis or infection onset lies too close to
    ICU admission.

    The published criterion (Online Supplemental Figure 1) is "onset within 16
    hours of ICU admission", i.e. the onset falls in the first 16 hours of the
    stay. That is `--onset-window up-to`, the default. `before` tests the 16
    hours preceding admission and is a different criterion; it is retained only
    for comparison with the earlier version of this script.

    The rule is applied to labelled stays only, exactly as in the published
    flowchart. Control stays carry a randomly sampled index time instead of an
    onset, and that index time is NOT subject to this rule. Control stays whose
    index time is nevertheless too close to admission cannot support a full
    feature window in step 5 and would be dropped there; they are counted here
    so that the loss is visible rather than silent.
    """
    stays = stays.copy()

    # The published criterion is measured from the ICU admission timestamp
    # itself. Flooring it to the hour would move the boundary up to 59 minutes
    # earlier and widen the exclusion window to almost 17 hours, so it is only
    # done when explicitly requested for backwards comparison.
    reference = stays[ADMISSION_TIME].dt.floor("h") if floor_admission else stays[ADMISSION_TIME]

    sepsis_flag = (stays[SEPSIS_LABEL] == 1) & onset_in_window(
        stays[sepsis_onset_col], reference, window, hours
    )
    infection_flag = (stays[INFECTION_LABEL] == 1) & onset_in_window(
        stays[infection_onset_col], reference, window, hours
    )

    stays["excluded_sepsis_onset_near_admission"] = sepsis_flag.astype(int)
    stays["excluded_infection_onset_near_admission"] = infection_flag.astype(int)

    excluded_mask = sepsis_flag | infection_flag
    counts = {
        "n_sepsis_onset_near_admission": int(sepsis_flag.sum()),
        "n_infection_onset_near_admission": int(infection_flag.sum()),
        "n_overlap": int((sepsis_flag & infection_flag).sum()),
        "n_excluded": int(excluded_mask.sum()),
    }

    # Audit only: control index times that are too close to ICU admission to
    # support a feature window. These stays remain in the cohort, matching the
    # published flowchart, but step 5 will not be able to build a sample.
    control_near = (stays[SEPSIS_LABEL] != 1) & onset_in_window(
        stays[sepsis_onset_col], reference, window, hours
    )
    stays["control_index_time_near_admission"] = control_near.astype(int)
    counts["n_control_index_time_near_admission_not_excluded"] = int(control_near.sum())
    if int(control_near.sum()):
        LOGGER.warning(
            "%d control stay(s) have an index time within %.0f h of ICU admission. They are "
            "retained here, as in the published flowchart, but step 5 cannot build an "
            "8-hour feature window for them; report this in the limitations",
            int(control_near.sum()), hours,
        )

    assert counts["n_excluded"] == (
        counts["n_sepsis_onset_near_admission"]
        + counts["n_infection_onset_near_admission"]
        - counts["n_overlap"]
    ), "stage 2 counts do not reconcile"
    # The control audit above is deliberately excluded from this identity: those
    # stays are counted, not removed.

    return (
        stays[~excluded_mask].reset_index(drop=True),
        stays[excluded_mask].reset_index(drop=True),
        counts,
    )


# --------------------------------------------------------------------------- #
# Stage 3 -- one shared patient-level split for both model outcomes
# --------------------------------------------------------------------------- #

def build_patient_level_labels(
    df: pd.DataFrame,
    group_col: str,
    label_cols: list[str],
) -> pd.DataFrame:
    """
    Build the patient-level table used by the split.

    For each outcome, the patient-level label is 1 if any ICU stay of that
    patient is positive. The split is shared by both outcomes, and the strata
    are built from these patient-level labels.

    A missing or non-numeric outcome is an error rather than a silent zero: it
    would otherwise move the patient into the negative stratum and bias the
    split without any warning.
    """
    missing_cols = [c for c in label_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"outcome column(s) {missing_cols} not found. This shared split is used "
            "for both sepsis and infection models, so both validated 0/1 outcomes "
            "must be present. Use --column-map or rerun the upstream labelling step."
        )

    temp = df[[group_col] + label_cols].copy()
    for col in label_cols:
        values = pd.to_numeric(temp[col], errors="coerce")
        n_bad = int(values.isna().sum())
        if n_bad:
            raise ValueError(
                f"outcome column '{col}' contains {n_bad} missing or non-numeric "
                "value(s). Fix the labelling step; these stays must not be treated "
                "as negatives when forming the strata."
            )
        observed = sorted(pd.unique(values.astype(int)))
        if not set(observed).issubset({0, 1}):
            raise ValueError(f"outcome column '{col}' must be 0/1; observed {observed}.")
        temp[col] = values.astype(int)

    patient_labels = (
        temp.groupby(group_col, as_index=False)[label_cols]
        .max()
        .reset_index(drop=True)
    )
    return patient_labels


def build_strata(patient_labels: pd.DataFrame, stratify_by: str) -> pd.Series:
    """Return the stratum of each patient as a readable string.

    'joint' crosses the two outcomes, so both prevalences are preserved
    simultaneously. The single-outcome settings are for sensitivity analysis.
    """
    sepsis = patient_labels[SEPSIS_LABEL].astype(int)
    infection = patient_labels[INFECTION_LABEL].astype(int)

    if stratify_by == "joint":
        return "sepsis=" + sepsis.astype(str) + ",infection=" + infection.astype(str)
    if stratify_by == "sepsis":
        return "sepsis=" + sepsis.astype(str)
    if stratify_by == "infection":
        return "infection=" + infection.astype(str)
    raise ValueError(f"unknown --stratify-by value: {stratify_by}")


def stratified_patient_split(
    patient_labels: pd.DataFrame,
    group_col: str,
    strata: pd.Series,
    test_size: float,
    seed: int,
) -> tuple[set, pd.DataFrame]:
    """Draw the test patients independently within each stratum.

    Done directly rather than with StratifiedShuffleSplit so that the number of
    test patients per stratum is rounded explicitly, and so that a stratum too
    small to contribute a test patient is reported rather than raising.

    The draw depends only on `seed` and the patient identifiers, so it is
    reproducible and does not depend on the row order of the input.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    test_patients: set = set()
    rows: list[dict] = []

    for stratum in sorted(pd.unique(strata)):
        members = patient_labels.loc[strata == stratum, group_col]
        # Sort before shuffling so the result does not depend on the row order
        # of the input table.
        ordered = np.sort(members.to_numpy().astype(str))
        n = len(ordered)
        n_test = int(round(n * test_size))

        # Keep at least one patient on each side whenever the stratum can spare
        # one; a stratum of a single patient always goes to training.
        if n >= 2:
            n_test = min(max(n_test, 1), n - 1)
        else:
            n_test = 0

        drawn = rng.permutation(ordered)[:n_test]
        test_patients.update(drawn.tolist())

        if n < 2:
            LOGGER.warning(
                "stratum '%s' contains only %d patient(s) and was assigned entirely "
                "to the training cohort", stratum, n,
            )

        rows.append(
            {
                "stratum": stratum,
                "n_patients": n,
                "n_train_patients": n - n_test,
                "n_test_patients": n_test,
                "test_fraction": n_test / n if n else float("nan"),
            }
        )

    summary = pd.DataFrame(rows)
    LOGGER.info("stratified patient allocation:\n%s", summary.to_string(index=False))
    return test_patients, summary


def assign_shared_patient_split(
    cohort: pd.DataFrame,
    test_size: float,
    seed: int,
    group_col: str = PATIENT_ID,
    split_by: str = "patient",
    stratify_by: str = "joint",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign the train/test split, stratified on outcome at the patient level.

    patient mode:
        Every stay of one patient stays on the same side, and the test fraction
        is drawn inside each outcome stratum so that the sepsis and infection
        prevalences are preserved between the two cohorts.

    stay mode:
        Optional fallback for debugging only; not recommended for model
        reporting because a patient with multiple ICU stays may leak across
        train/test. It is stratified on the stay-level outcome.

    Returns the cohort with a `split` column, and the per-stratum allocation
    table.
    """
    cohort = cohort.copy()

    if split_by == "patient":
        patient_labels = build_patient_level_labels(
            cohort,
            group_col=group_col,
            label_cols=[SEPSIS_LABEL, INFECTION_LABEL],
        )
        patient_labels[group_col] = patient_labels[group_col].astype(str)
        strata = build_strata(patient_labels, stratify_by)

        test_groups, strata_summary = stratified_patient_split(
            patient_labels,
            group_col=group_col,
            strata=strata,
            test_size=test_size,
            seed=seed,
        )

        train_groups = set(patient_labels[group_col]) - test_groups
        if train_groups & test_groups:
            raise AssertionError("patient split failed: train/test patient overlap detected")

        is_test = cohort[group_col].astype(str).isin(test_groups)

    else:
        patient_labels = build_patient_level_labels(
            cohort.assign(**{group_col: cohort.index.astype(str)}),
            group_col=group_col,
            label_cols=[SEPSIS_LABEL, INFECTION_LABEL],
        )
        strata = build_strata(patient_labels, stratify_by)
        test_rows, strata_summary = stratified_patient_split(
            patient_labels,
            group_col=group_col,
            strata=strata,
            test_size=test_size,
            seed=seed,
        )
        is_test = cohort.index.astype(str).isin(test_rows)
        is_test = pd.Series(is_test, index=cohort.index)

    cohort["split"] = np.where(is_test, "test", "train")

    log_split_summary(cohort, group_col=group_col, seed=seed, test_size=test_size, split_by=split_by)
    return cohort, strata_summary


def log_split_summary(
    cohort: pd.DataFrame,
    group_col: str,
    seed: int,
    test_size: float,
    split_by: str,
) -> None:
    """Log row-level and patient-level prevalence for both outcomes."""
    n_train = int((cohort["split"] == "train").sum())
    n_test = int((cohort["split"] == "test").sum())
    achieved = n_test / len(cohort) if len(cohort) else float("nan")

    LOGGER.info(
        "shared split by %s (seed=%d): %d train / %d test stay(s), test fraction %.4f",
        split_by, seed, n_train, n_test, achieved,
    )
    if abs(achieved - test_size) > 0.02:
        LOGGER.warning(
            "the achieved test fraction %.4f differs from --test-size %.2f by more than "
            "2 percentage points; with --split-by patient this can happen when patients "
            "contribute unequal numbers of stays",
            achieved, test_size,
        )

    for label_col, label_name in [(SEPSIS_LABEL, "sepsis"), (INFECTION_LABEL, "infection")]:
        for split in ["train", "test"]:
            part = cohort[cohort["split"] == split]
            row_prev = float(pd.to_numeric(part[label_col], errors="coerce").mean())

            patient_prev = (
                part[[group_col, label_col]]
                .assign(**{label_col: pd.to_numeric(part[label_col], errors="coerce").fillna(0).astype(int)})
                .groupby(group_col)[label_col]
                .max()
                .mean()
            )

            LOGGER.info(
                "%s %s: rows=%d, patients=%d, row prevalence=%.4f, patient prevalence=%.4f",
                label_name,
                split,
                len(part),
                part[group_col].nunique(),
                row_prev,
                float(patient_prev),
            )


def build_prevalence_check(cohort: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Report the achieved prevalence of both outcomes in both cohorts.

    Both levels are reported. Patient level is where the stratification acts, so
    the two cohorts should match closely there. Stay level is what Table 1
    reports, and a patient-level split can only hold it approximately, because
    patients contribute unequal numbers of ICU stays.
    """
    rows: list[dict] = []
    for label_col, label_name in [(SEPSIS_LABEL, "sepsis"), (INFECTION_LABEL, "infection")]:
        values: dict[str, dict[str, float]] = {}
        for split in ("train", "test"):
            part = cohort[cohort["split"] == split]
            labels = pd.to_numeric(part[label_col], errors="coerce")
            values[split] = {
                "n_stays": int(len(part)),
                "n_patients": int(part[group_col].nunique()),
                "stay_prevalence": float(labels.mean()),
                "patient_prevalence": float(
                    part.assign(_l=labels.astype(int))
                    .groupby(group_col)["_l"].max().mean()
                ),
            }
        for level in ("stay", "patient"):
            key = f"{level}_prevalence"
            rows.append(
                {
                    "outcome": label_name,
                    "level": level,
                    "train": values["train"][key],
                    "test": values["test"][key],
                    "difference_pp": 100.0 * (values["test"][key] - values["train"][key]),
                    "n_train": values["train"][f"n_{level}s"],
                    "n_test": values["test"][f"n_{level}s"],
                }
            )

    check = pd.DataFrame(rows)
    LOGGER.info(
        "achieved prevalence by cohort:\n%s",
        check.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
    )
    return check


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def build_summary(
    n_entry: int,
    stage1: dict[str, int],
    n_eligible: int,
    stage2: dict[str, int],
    n_study: int,
    cohort: pd.DataFrame,
    vital_cols: list[str],
) -> pd.DataFrame:
    """Build one row per box of the published flowchart, and check arithmetic."""
    rows: list[dict] = [
        {"stage": "entry", "item": "Adult ICU admissions", "n_adm_ICU_id": n_entry},
        {
            "stage": "stage_1",
            "item": "Excluded: ICU stay < threshold",
            "n_adm_ICU_id": stage1["n_short_stay"],
        },
    ]

    for col in vital_cols:
        key = f"n_missing_{col}"
        if key in stage1:
            rows.append(
                {
                    "stage": "stage_1",
                    "item": f"Excluded: {col} missing",
                    "n_adm_ICU_id": stage1[key],
                }
            )

    rows += [
        {
            "stage": "stage_1",
            "item": "Excluded: any vital sign missing",
            "n_adm_ICU_id": stage1["n_vital_missing_any"],
        },
        {
            "stage": "stage_1",
            "item": "Excluded: met both criteria 1 and 2",
            "n_adm_ICU_id": stage1["n_overlap"],
        },
        {"stage": "stage_1", "item": "Excluded: total", "n_adm_ICU_id": stage1["n_excluded"]},
        {"stage": "stage_1", "item": "Eligible cohort", "n_adm_ICU_id": n_eligible},
        {
            "stage": "stage_2",
            "item": "Excluded: first sepsis onset within 16 hours of ICU admission",
            "n_adm_ICU_id": stage2["n_sepsis_onset_near_admission"],
        },
        {
            "stage": "stage_2",
            "item": "Excluded: first infection onset within 16 hours of ICU admission",
            "n_adm_ICU_id": stage2["n_infection_onset_near_admission"],
        },
        {
            "stage": "stage_2",
            "item": "Excluded: overlap of sepsis/infection onset criteria",
            "n_adm_ICU_id": stage2["n_overlap"],
        },
        {"stage": "stage_2", "item": "Excluded: total", "n_adm_ICU_id": stage2["n_excluded"]},
        {"stage": "stage_2", "item": "Study cohort", "n_adm_ICU_id": n_study},
    ]

    for split in ("train", "test"):
        part = cohort[cohort["split"] == split]
        rows += [
            {
                "stage": f"split_{split}",
                "item": f"{split.capitalize()} cohort",
                "n_adm_ICU_id": len(part),
            },
            {
                "stage": f"split_{split}",
                "item": "Sepsis",
                "n_adm_ICU_id": int((part[SEPSIS_LABEL] == 1).sum()),
            },
            {
                "stage": f"split_{split}",
                "item": "Non-sepsis",
                "n_adm_ICU_id": int((part[SEPSIS_LABEL] != 1).sum()),
            },
            {
                "stage": f"split_{split}",
                "item": "Infection",
                "n_adm_ICU_id": int((part[INFECTION_LABEL] == 1).sum()),
            },
            {
                "stage": f"split_{split}",
                "item": "Non-infection",
                "n_adm_ICU_id": int((part[INFECTION_LABEL] != 1).sum()),
            },
        ]

    assert n_eligible == n_entry - stage1["n_excluded"], "eligible cohort does not reconcile"
    assert n_study == n_eligible - stage2["n_excluded"], "study cohort does not reconcile"
    assert len(cohort) == n_study, "the split changed the size of the study cohort"

    for split in ("train", "test"):
        part = cohort[cohort["split"] == split]
        assert len(part) == int((part[SEPSIS_LABEL] == 1).sum()) + int((part[SEPSIS_LABEL] != 1).sum())
        assert len(part) == int((part[INFECTION_LABEL] == 1).sum()) + int((part[INFECTION_LABEL] != 1).sum())

    return pd.DataFrame(rows)


def print_flowchart(summary: pd.DataFrame) -> None:
    """Print the summary in the reading order of the published flowchart."""
    width = max(len(str(item)) for item in summary["item"]) + 2
    current_stage = None
    for row in summary.itertuples(index=False):
        if row.stage != current_stage:
            current_stage = row.stage
            print(f"\n--- {current_stage} ---")
        print(f"  {row.item:<{width}} {row.n_adm_ICU_id:>8,}")
    print()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def parse_column_map(pairs: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--column-map expects RAW=CANONICAL, got: {pair}")
        raw, canonical = pair.split("=", 1)
        mapping[raw.strip()] = canonical.strip()
    return mapping


def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    column_map = parse_column_map(args.column_map)
    vital_cols = args.vital_cols

    # ---- load -------------------------------------------------------------- #
    df = load_table(args.cohort, column_map)
    require_columns(
        df,
        STAY_KEY + [ADMISSION_TIME, DISCHARGE_TIME, SEPSIS_LABEL, SEPSIS_ONSET],
        args.cohort,
    )

    df = to_datetime(df, [ADMISSION_TIME, DISCHARGE_TIME, SEPSIS_ONSET])
    df = to_datetime(df, [INFECTION_ONSET], required=False)

    for col in STAY_KEY:
        df[col] = normalise_id(df[col])

    require_columns(df, [SEPSIS_LABEL, INFECTION_LABEL], args.cohort)
    for col in (SEPSIS_LABEL, INFECTION_LABEL):
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"label column '{col}' contains {int(values.isna().sum())} missing or "
                "non-numeric value(s); rerun the upstream labelling step"
            )
        observed = sorted(pd.unique(values.astype(int)))
        if not set(observed).issubset({0, 1}):
            raise ValueError(f"label column '{col}' must contain only 0/1; observed {observed}")
        df[col] = values.astype(int)

    infection_onset_col = args.infection_onset_col or INFECTION_ONSET
    require_columns(df, [args.sepsis_onset_col, infection_onset_col], args.cohort)
    LOGGER.info(
        "stage 2 uses '%s' for sepsis and '%s' for infection, window=%s, %.1f h",
        args.sepsis_onset_col, infection_onset_col, args.onset_window, args.onset_window_hours,
    )
    stays = collapse_to_stays(df, vital_cols, args.missing_rule)
    stays = apply_cohort_entry(stays, args)
    n_entry = len(stays)
    LOGGER.info("cohort entry: %d ICU stay(s)", n_entry)

    # ---- stage 1 ----------------------------------------------------------- #
    eligible, excluded1, stage1 = apply_stage1(
        stays=stays,
        vital_cols=vital_cols,
        min_hours=args.min_icu_stay_hours,
    )
    excluded1.to_csv(outdir / "excluded_stage1.csv", index=False, encoding="utf-8-sig")
    LOGGER.info("stage 1: %d excluded, %d eligible", stage1["n_excluded"], len(eligible))

    # ---- stage 2 ----------------------------------------------------------- #
    study, excluded2, stage2 = apply_stage2(
        stays=eligible,
        window=args.onset_window,
        hours=args.onset_window_hours,
        sepsis_onset_col=args.sepsis_onset_col,
        infection_onset_col=infection_onset_col,
        floor_admission=args.floor_admission_time,
    )
    excluded2.to_csv(outdir / "excluded_stage2.csv", index=False, encoding="utf-8-sig")
    LOGGER.info("stage 2: %d excluded, %d in the study cohort", stage2["n_excluded"], len(study))

    # ---- stage 3 ----------------------------------------------------------- #
    study, strata_summary = assign_shared_patient_split(
        cohort=study,
        test_size=args.test_size,
        seed=args.seed,
        group_col=PATIENT_ID,
        split_by=args.split_by,
        stratify_by=args.stratify_by,
    )
    strata_summary.to_csv(outdir / "split_strata_summary.csv", index=False, encoding="utf-8-sig")
    build_prevalence_check(study, group_col=PATIENT_ID).to_csv(
        outdir / "split_prevalence_check.csv", index=False, encoding="utf-8-sig"
    )

    study.to_csv(outdir / "study_cohort.csv", index=False, encoding="utf-8-sig")
    study[study["split"] == "train"].to_csv(outdir / "train_cohort.csv", index=False, encoding="utf-8-sig")
    study[study["split"] == "test"].to_csv(outdir / "test_cohort.csv", index=False, encoding="utf-8-sig")

    # ---- summary ----------------------------------------------------------- #
    summary = build_summary(
        n_entry=n_entry,
        stage1=stage1,
        n_eligible=len(eligible),
        stage2=stage2,
        n_study=len(study),
        cohort=study,
        vital_cols=vital_cols,
    )
    summary.to_csv(outdir / "flowchart_summary.csv", index=False, encoding="utf-8-sig")

    config = vars(args).copy()
    config["stage1_rule"] = f"ICU_stay_hours < {args.min_icu_stay_hours}"
    config["split_logic"] = (
        f"one shared patient-level split, stratified by {args.stratify_by} outcome, "
        "applied to sepsis and infection together"
    )
    (outdir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_flowchart(summary)
    LOGGER.info("flowchart summary written to %s", outdir / "flowchart_summary.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cohort selection and construction of the model datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--cohort", required=True, help="labelled cohort from step 2: file path or sql:<file>.sql")
    parser.add_argument("--outdir", default="output", help="output directory")
    parser.add_argument(
        "--column-map",
        nargs="*",
        default=[],
        metavar="RAW=CANONICAL",
        help="rename source columns before processing",
    )

    parser.add_argument("--start-date", default=None, help="earliest ICU admission date, inclusive")
    parser.add_argument("--end-date", default=None, help="latest ICU admission date, inclusive")
    parser.add_argument("--min-age", type=float, default=None, help="minimum age at ICU admission")

    parser.add_argument(
        "--min-icu-stay-hours",
        type=float,
        default=16.0,
        help="short-stay threshold; exclusion is strictly ICU_stay_hours < threshold",
    )
    parser.add_argument("--vital-cols", nargs="+", default=DEFAULT_VITAL_COLS, help="vital signs that must be recorded")
    parser.add_argument(
        "--missing-rule",
        choices=["any", "all"],
        default="any",
        help="'any' excludes a stay if any row lacks the measurement; 'all' excludes only if all rows lack it",
    )

    parser.add_argument(
        "--onset-window",
        choices=["before", "after", "both", "up-to"],
        default="up-to",
        help="position of the stage 2 exclusion window relative to ICU admission. "
             "'up-to' is the published criterion (onset <= admission + 16 h, i.e. onset "
             "within the first 16 hours of the ICU stay). 'before' tests the 16 hours "
             "preceding admission and was the default of the earlier version",
    )
    parser.add_argument("--onset-window-hours", type=float, default=16.0, help="width of the stage 2 exclusion window")
    parser.add_argument("--sepsis-onset-col", default=SEPSIS_ONSET, help="timestamp tested for the sepsis criterion")
    parser.add_argument(
        "--infection-onset-col",
        default=INFECTION_ONSET,
        help="timestamp tested for the infection criterion; use the infection onset "
             "produced by step 2",
    )

    parser.add_argument(
        "--floor-admission-time",
        action="store_true",
        help="floor the ICU admission timestamp to the hour before applying the stage 2 "
             "window; this widens the window by up to 59 minutes and is off by default",
    )
    parser.add_argument("--test-size", type=float, default=0.20, help="proportion assigned to the internal test set")
    parser.add_argument(
        "--split-by",
        choices=["patient", "stay"],
        default="patient",
        help="'patient' keeps all stays of one patient on the same side",
    )
    parser.add_argument(
        "--stratify-by",
        choices=["joint", "sepsis", "infection"],
        default="joint",
        help="outcome used to form the strata. 'joint' crosses sepsis and infection, "
             "which preserves both prevalences at once",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="fixed seed for the stratified allocation")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        run(args)
    except (FileNotFoundError, KeyError, ValueError, AssertionError) as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
