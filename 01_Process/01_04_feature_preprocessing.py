#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (C) 2026 Fang-Ju Sun and contributors
"""
Step 4 -- Feature preprocessing for Online Supplemental Table 1.

This script is intended for GitHub release with the analysis code. It rewrites the
original notebook-style clipping and imputation code as a reproducible command-line
pipeline.

Main tasks
----------
1. Apply the physiological clipping limits reported in Online Supplemental Table 1.
2. Apply the imputation rules reported in Online Supplemental Table 1:
   - APACHE II score: admission value / last observation carried forward.
   - Vital signs and 8-hour urine output: forward fill within 8 hours.
   - Laboratory variables: two-stage look-back imputation within 72 hours and then
     168 hours. For the P/F-ratio row, PaO2 is look-back imputed first; FiO2 is
     defaulted to 21% only when unavailable; P/F ratio is then derived.
   - Therapy/exposure count variables: missing values are assumed to be zero.
   - Remaining missing APACHE/vital values are imputed using training-cohort means.
   - Remaining missing laboratory values are imputed by reproducible random sampling
     within physiological reference ranges.
3. Count admission-level residual missingness by the fixed train/test split and
   write a CSV table matching the structure of Online Supplemental Table 1.

Important design choices
------------------------
- No train/test split is created here. The split should come from Step 3 and must be
  fixed before imputation. This prevents test-set information from influencing the
  training cohort.
- Training-cohort means are calculated only from rows with split == "train" after
  clipping and temporal carry-forward/look-back steps.
- The code accepts either string split labels ("train", "test") or the older numeric
  labels used in the private analysis code (1 = train, 2 = test).
- By default, temporal carry-forward and look-back imputation is grouped by
  adm_ICU_id and restricted to records within the same ICU episode. Values are
  not carried across separate ICU episodes. Use another encounter identifier
  only if the local source schema defines it as one ICU-episode-level identifier.
- Residual missingness is counted within the first 8 hours after ICU admission,
  matching Online Supplemental Table 1. Urine output is counted only from hour 4
  through hour 8 to allow for accumulation and recording lag.
- Residual missingness is counted BEFORE any imputation, including the zero
  assumption for the therapy/exposure variables. Those variables are filled in
  the final imputation step together with everything else, so their Table 1 row
  reports how often the value was genuinely absent rather than a structural
  zero. ``--legacy-zero-before-missingness`` restores the earlier ordering, in
  which those four rows always read 0 (0.00%).
- Metadata columns that the input table already carries are not re-merged from
  ``--cohort-input``. Doing so would leave a second copy under a
  ``<name>_cohort`` suffix, and step 5 treats unrecognised columns as model
  features, so identifiers and admission timestamps would end up forward filled
  into the feature matrix.
- The final random laboratory imputation is seeded and therefore reproducible.

Example
-------
python 01_04_feature_preprocessing.py \
    --vital-input output/study_cohort_features.csv \
    --lab-input output/lab_features.csv \
    --cohort-input output/study_cohort.csv \
    --outdir output/step4_preprocessing \
    --expected-train-n 11316 \
    --expected-test-n 2837

The script contains no patient-identifiable data. The user must provide local input
files created from the approved institutional workflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("feature_preprocessing_table1")


# --------------------------------------------------------------------------- #
# Feature definitions used by Online Supplemental Table 1.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for one feature row in Online Supplemental Table 1."""

    name: str
    display_name: str
    category: str
    procedure: str
    final_strategy: str
    clip_lower: float | None = None
    clip_upper: float | None = None
    reference_lower: float | None = None
    reference_upper: float | None = None
    clip_display: str | None = None

    @property
    def min_max_interval(self) -> str:
        if self.clip_display:
            return self.clip_display
        if self.clip_lower is None and self.clip_upper is None:
            return ""
        if self.clip_upper is None:
            return f">={self.clip_lower:g}"
        if self.clip_lower is None:
            return f"<={self.clip_upper:g}"
        return f"{self.clip_lower:g}-{self.clip_upper:g}"


FEATURE_SPECS: list[FeatureSpec] = [
    FeatureSpec(
        name="APACHEII_score",
        display_name="APACHE II score",
        category="apache",
        procedure="Admission value (last observation carried forward)",
        final_strategy="train_mean",
        clip_lower=0,
        clip_upper=71,
    ),
    FeatureSpec(
        name="Temperature",
        display_name="Temperature",
        category="vital",
        procedure="Forward fill within 8 hours",
        final_strategy="train_mean",
        clip_lower=25,
        clip_upper=40,
    ),
    FeatureSpec(
        name="Pulse",
        display_name="PR",
        category="vital",
        procedure="Forward fill within 8 hours",
        final_strategy="train_mean",
        clip_lower=0,
        clip_upper=200,
    ),
    FeatureSpec(
        name="Respiration",
        display_name="RR",
        category="vital",
        procedure="Forward fill within 8 hours",
        final_strategy="train_mean",
        clip_lower=0,
        clip_upper=50,
    ),
    FeatureSpec(
        name="SBP",
        display_name="SBP",
        category="vital",
        procedure="Forward fill within 8 hours",
        final_strategy="train_mean",
        clip_lower=0,
        clip_upper=300,
    ),
    FeatureSpec(
        name="GCS",
        display_name="GCS",
        category="vital",
        procedure="Forward fill within 8 hours",
        final_strategy="train_mean",
        clip_lower=3,
        clip_upper=15,
    ),
    FeatureSpec(
        name="Urine_output_8H",
        display_name="8-hour urine output",
        category="vital",
        procedure="Forward fill within 8 hours",
        final_strategy="train_mean",
        clip_lower=0,
        clip_upper=1500,
    ),
    FeatureSpec(
        name="WBC",
        display_name="WBC",
        category="lab",
        procedure="Two-stage look-back (72 hours, then 168 hours)",
        final_strategy="random_reference_range",
        clip_lower=0,
        clip_upper=200,
        reference_lower=4,
        reference_upper=10,
    ),
    FeatureSpec(
        name="CRP",
        display_name="CRP",
        category="lab",
        procedure="Two-stage look-back (72 hours, then 168 hours)",
        final_strategy="random_reference_range",
        clip_lower=0,
        clip_upper=100,
        reference_lower=0,
        reference_upper=0.79,
    ),
    FeatureSpec(
        name="PCT",
        display_name="PCT",
        category="lab",
        procedure="Two-stage look-back (72 hours, then 168 hours)",
        final_strategy="random_reference_range",
        clip_lower=0,
        clip_upper=200,
        reference_lower=0,
        reference_upper=0.09,
    ),
    FeatureSpec(
        name="Neutrophil_Seg",
        display_name="Segmented neutrophil percentage",
        category="lab",
        procedure="Two-stage look-back (72 hours, then 168 hours)",
        final_strategy="random_reference_range",
        clip_lower=0,
        clip_upper=100,
        reference_lower=55,
        reference_upper=75,
    ),
    FeatureSpec(
        name="PFratio",
        display_name="PaO2 / FiO2 ratio (P/F ratio)",
        category="lab",
        procedure="PaO2: Two-stage look-back (72 hours, then 168 hours); FiO2: defaulted to 21% (room air)",
        final_strategy="random_reference_range",
        clip_lower=0,
        clip_upper=800,
        reference_lower=400,
        reference_upper=800,
    ),
    FeatureSpec(
        name="Ventilator_days",
        display_name="Ventilator days",
        category="zero_assumed",
        procedure="Assumed zero",
        final_strategy="zero",
        clip_lower=0,
        clip_upper=None,
        clip_display=">0",
    ),
    FeatureSpec(
        name="Antibiotic_count",
        display_name="Antibiotic count",
        category="zero_assumed",
        procedure="Assumed zero",
        final_strategy="zero",
        clip_lower=0,
        clip_upper=None,
        clip_display=">0",
    ),
    FeatureSpec(
        name="Norepinephrine",
        display_name="Norepinephrine",
        category="zero_assumed",
        procedure="Assumed zero",
        final_strategy="zero",
        clip_lower=0,
        clip_upper=1,
    ),
    FeatureSpec(
        name="Dopamine",
        display_name="Dopamine",
        category="zero_assumed",
        procedure="Assumed zero",
        final_strategy="zero",
        clip_lower=0,
        clip_upper=40,
    ),
]

FEATURE_BY_NAME = {spec.name: spec for spec in FEATURE_SPECS}

PAO2_COL = "PaO2"
FIO2_COL = "FiO2"
PFRATIO_COL = "PFratio"

# Expected final mean values shown in the published Online Supplemental Table 1.
# These are not used for imputation unless --use-published-final-means is passed.
PUBLISHED_FINAL_MEANS = {
    "APACHEII_score": 18.44,
    "Temperature": 36.93,
    "Pulse": 86.55,
    "Respiration": 18.53,
    "SBP": 130.39,
    "GCS": 12.92,
    "Urine_output_8H": 657.86,
}

# Common aliases seen in the private analysis notebooks or raw warehouse extracts.
COLUMN_ALIASES = {
    "PR": "Pulse",
    "RR": "Respiration",
    "Urine_output": "Urine_output_8H",
    "UrineOutput8H": "Urine_output_8H",
    "lab_WBC": "WBC",
    "lab_CRP": "CRP",
    "lab_Procalcitonin": "PCT",
    "lab_PCT": "PCT",
    "lab_Seg": "Neutrophil_Seg",
    "Segmented_neutrophil_percentage": "Neutrophil_Seg",
    "Seg": "Neutrophil_Seg",
    "lab_Pao2": "PaO2",
    "lab_PaO2": "PaO2",
    "Pao2": "PaO2",
    "PAO2": "PaO2",
    "lab_FiO2": "FiO2",
    "FIO2": "FiO2",
    "FiO2_percent": "FiO2",
    "PF_ratio": "PFratio",
    "P_F_ratio": "PFratio",
    "PaO2_FiO2": "PFratio",
    "Ventilator days": "Ventilator_days",
    "Antibiotic count": "Antibiotic_count",
    "norepinephrine": "Norepinephrine",
    "dopamine": "Dopamine",
}


# --------------------------------------------------------------------------- #
# Basic I/O and validation helpers.
# --------------------------------------------------------------------------- #


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV/TSV/Excel file using a conservative default configuration."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype="object")
    sep = "\t" if suffix in {".tsv", ".tab"} else ","
    return pd.read_csv(path, sep=sep, dtype="object", encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def rename_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known source aliases only when the canonical column is absent."""
    df = df.copy()
    rename_map = {}
    for old, new in COLUMN_ALIASES.items():
        if old in df.columns and new not in df.columns:
            rename_map[old] = new
    if rename_map:
        LOGGER.info("renamed source columns: %s", rename_map)
        df = df.rename(columns=rename_map)
    return df


def require_columns(df: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{source} is missing required column(s): {missing}")


def normalise_id(series: pd.Series) -> pd.Series:
    """Strip whitespace and a trailing .0 produced by spreadsheet round-trips."""
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def convert_datetime(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            bad = int(df[col].isna().sum())
            if bad:
                LOGGER.debug("%s has %d missing/unparseable datetime value(s)", col, bad)
    return df


def convert_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalise_split_value(value: object) -> str | None:
    """Map older numeric split values to string labels used by Step 3."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "1.0", "train", "training", "train cohort"}:
        return "train"
    if text in {"2", "2.0", "test", "internal test", "internal_test", "test cohort"}:
        return "test"
    return text


def normalise_split_column(df: pd.DataFrame, split_col: str) -> pd.DataFrame:
    df = df.copy()
    if split_col not in df.columns:
        return df
    df[split_col] = df[split_col].map(normalise_split_value)
    return df


def merge_cohort_metadata(
    df: pd.DataFrame,
    cohort: pd.DataFrame | None,
    adm_col: str,
    split_col: str,
    metadata_cols: list[str],
    overwrite_split: bool,
) -> pd.DataFrame:
    """Attach the fixed split and core admission metadata created upstream."""
    df = df.copy()
    if cohort is None:
        if split_col not in df.columns:
            raise KeyError(
                f"'{split_col}' is absent. Provide --cohort-input from Step 3 or include the split column in the input table."
            )
        return normalise_split_column(df, split_col)

    cohort = cohort.copy()
    require_columns(cohort, [adm_col, split_col], "cohort input")
    cohort = normalise_split_column(cohort, split_col)
    cohort[adm_col] = normalise_id(cohort[adm_col])

    # Preserve order but avoid duplicated columns when, for example, group_col == adm_col.
    #
    # A metadata column that the input table already carries is NOT taken from
    # the cohort file. Merging it anyway produces a second copy under a
    # '<name>_cohort' suffix, and those copies travel downstream: step 5 treats
    # any column it does not recognise as a model feature, so identifiers and
    # admission timestamps would be forward filled into the feature matrix.
    # The merge key and the split are handled separately below.
    already_present = [
        col for col in metadata_cols
        if col in df.columns and col not in {adm_col, split_col}
    ]
    if already_present:
        LOGGER.info(
            "metadata column(s) already present in the input table and therefore not "
            "re-merged from the cohort file: %s", already_present
        )

    available_meta = []
    for col in [adm_col, split_col] + metadata_cols:
        if col in cohort.columns and col not in available_meta and col not in already_present:
            available_meta.append(col)
    meta = cohort[available_meta].drop_duplicates(subset=[adm_col])

    df[adm_col] = normalise_id(df[adm_col])
    if split_col in df.columns and overwrite_split:
        df = df.drop(columns=[split_col])
    elif split_col in df.columns and not overwrite_split:
        meta = meta.drop(columns=[split_col], errors="ignore")

    df = df.merge(meta, on=adm_col, how="left", suffixes=("", "_cohort"))
    df = normalise_split_column(df, split_col)

    # Defensive: nothing should collide now, but a stray duplicate would silently
    # become a model feature in step 5, so it is reported rather than left to be
    # discovered in the feature matrix.
    suffixed = [col for col in df.columns if col.endswith("_cohort")]
    if suffixed:
        LOGGER.warning("duplicate column(s) produced by the cohort merge: %s", suffixed)

    missing_split = int(df[split_col].isna().sum()) if split_col in df.columns else len(df)
    if missing_split:
        LOGGER.warning("%d row(s) do not have an assigned train/test split after metadata merge", missing_split)
    return df


# --------------------------------------------------------------------------- #
# Feature engineering helpers.
# --------------------------------------------------------------------------- #


def harmonise_pao2_fio2_columns(df: pd.DataFrame, pao2_col: str, fio2_col: str) -> pd.DataFrame:
    """Standardise PaO2 and FiO2 source column names without deriving P/F ratio yet.

    Online Supplemental Table 1 states that PaO2 is handled with the two-stage
    laboratory look-back and that FiO2 is defaulted to 21% room air. Therefore,
    the code must not rename a PaO2 column directly to PFratio. PaO2 is kept as a
    separate intermediate column, look-back imputed with the other laboratory
    variables, and only then divided by FiO2 to derive PFratio.
    """
    df = df.copy()
    rename_map: dict[str, str] = {}
    if pao2_col in df.columns and PAO2_COL not in df.columns:
        rename_map[pao2_col] = PAO2_COL
    if fio2_col in df.columns and FIO2_COL not in df.columns:
        rename_map[fio2_col] = FIO2_COL
    if rename_map:
        LOGGER.info("standardised oxygenation source columns: %s", rename_map)
        df = df.rename(columns=rename_map)
    return df


def derive_pfratio_after_pao2_lookback(df: pd.DataFrame) -> pd.DataFrame:
    """Derive P/F ratio after PaO2 look-back imputation and FiO2 defaulting.

    FiO2 is interpreted as a percentage when values are >1 (e.g., 21 or 50) and
    as a fraction when values are <=1 (e.g., 0.21 or 0.50). Missing or nonpositive
    FiO2 values are defaulted to 21% room air, matching the Table 1 footnote for
    the P/F-ratio preprocessing row.
    """
    df = df.copy()

    if PAO2_COL not in df.columns:
        if PFRATIO_COL in df.columns:
            LOGGER.warning(
                "%s is absent but an existing %s column is present. The existing %s will be kept, "
                "but this does not reproduce the Table 1 procedure unless it was derived after PaO2 look-back.",
                PAO2_COL, PFRATIO_COL, PFRATIO_COL,
            )
            return df
        LOGGER.warning("%s is absent; %s cannot be derived and will remain missing", PAO2_COL, PFRATIO_COL)
        df[PFRATIO_COL] = np.nan
        return df

    pao2 = pd.to_numeric(df[PAO2_COL], errors="coerce")

    if FIO2_COL in df.columns:
        fio2_raw = pd.to_numeric(df[FIO2_COL], errors="coerce")
    else:
        fio2_raw = pd.Series(np.nan, index=df.index, dtype="float64")
        LOGGER.info("%s is absent; all FiO2 values are defaulted to 21%% room air", FIO2_COL)

    invalid_fio2 = fio2_raw.notna() & (fio2_raw <= 0)
    if invalid_fio2.any():
        LOGGER.warning("%d nonpositive FiO2 value(s) were treated as missing and defaulted to 21%%", int(invalid_fio2.sum()))

    fio2_for_ratio = fio2_raw.mask(invalid_fio2).fillna(21.0)
    fio2_fraction = fio2_for_ratio.where(fio2_for_ratio <= 1, fio2_for_ratio / 100.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        df[PFRATIO_COL] = pao2 / fio2_fraction

    df["FiO2_used_for_PFratio"] = fio2_for_ratio
    LOGGER.info("derived %s from look-back-imputed %s; missing FiO2 defaulted to 21%%", PFRATIO_COL, PAO2_COL)
    return df


def clip_features(df: pd.DataFrame, specs: list[FeatureSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clip feature values to the prespecified physiological intervals."""
    df = df.copy()
    report_rows = []
    for spec in specs:
        if spec.name not in df.columns:
            report_rows.append(
                {
                    "feature": spec.name,
                    "status": "column_absent",
                    "clip_lower": spec.clip_lower,
                    "clip_upper": spec.clip_upper,
                    "outside_before": np.nan,
                    "outside_after": np.nan,
                    "n_missing_before": np.nan,
                }
            )
            continue

        df[spec.name] = pd.to_numeric(df[spec.name], errors="coerce")
        before = df[spec.name]
        outside = pd.Series(False, index=df.index)
        if spec.clip_lower is not None:
            outside |= before < spec.clip_lower
        if spec.clip_upper is not None:
            outside |= before > spec.clip_upper
        outside_before = int(outside.fillna(False).sum())
        missing_before = int(before.isna().sum())

        df[spec.name] = before.clip(lower=spec.clip_lower, upper=spec.clip_upper)

        after = df[spec.name]
        outside_after_mask = pd.Series(False, index=df.index)
        if spec.clip_lower is not None:
            outside_after_mask |= after < spec.clip_lower
        if spec.clip_upper is not None:
            outside_after_mask |= after > spec.clip_upper
        outside_after = int(outside_after_mask.fillna(False).sum())

        report_rows.append(
            {
                "feature": spec.name,
                "status": "clipped",
                "clip_lower": spec.clip_lower,
                "clip_upper": spec.clip_upper,
                "outside_before": outside_before,
                "outside_after": outside_after,
                "n_missing_before": missing_before,
            }
        )
    return df, pd.DataFrame(report_rows)


def observed_time_column(col: str) -> str:
    """Name of the hidden column holding the timestamp of the last real measurement."""
    return f"__observed_time__{col}"


def stage_column(col: str) -> str:
    """Name of the audit column recording which look-back stage filled a value."""
    return f"{col}__lookback_stage"


def forward_fill_within_hours(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    feature_cols: list[str],
    hours: float | None,
    stage_label: str | None = None,
) -> pd.DataFrame:
    """Carry forward the most recent *observed* value within a fixed window.

    The look-back limit is always measured from the timestamp of the last real
    measurement, never from the timestamp of a value that a previous call to
    this function already imputed. This matters when the two-stage laboratory
    procedure of Online Supplemental Table 1 is applied as two successive calls
    (72 hours, then 168 hours): without this safeguard the second pass treats
    the values filled by the first pass as observations, so a single real
    measurement can be chained forward far beyond the stated window (a
    measurement filled to +70 h can be carried another 168 h, giving an
    effective carry of 238 h, and so on without limit).

    The timestamp of the last observation is therefore carried in a hidden
    column, ``__observed_time__<feature>``, which survives between calls.
    :func:`drop_lookback_helpers` removes these columns before the table is
    written out.

    `stage_label`, when given, records in ``<feature>__lookback_stage`` which
    pass supplied each imputed value, so that the two stages remain auditable
    even though a value is only ever filled once.

    Row order is preserved, so that the returned frame stays aligned with the
    caller's other tables.
    """
    available_cols = [col for col in feature_cols if col in df.columns]
    if not available_cols:
        return df.copy()
    require_columns(df, [group_col, time_col], "forward-fill input")

    delta = None if hours is None else pd.Timedelta(hours=hours)
    df = df.copy()

    # Initialise, on first use, the timestamp of the last observed value.
    for col in available_cols:
        observed_col = observed_time_column(col)
        if observed_col not in df.columns:
            df[observed_col] = df[time_col].where(df[col].notna())
        if stage_label is not None and stage_column(col) not in df.columns:
            df[stage_column(col)] = pd.Series(pd.NA, index=df.index, dtype="object")

    order = df.index
    parts: list[pd.DataFrame] = []
    for _, group in df.groupby(group_col, sort=False, dropna=False):
        group = group.sort_values(time_col)
        current_time = group[time_col]

        for col in available_cols:
            observed_col = observed_time_column(col)
            value = group[col]

            previous_value = value.ffill()
            # Timestamp of the last genuine measurement, propagated forward.
            previous_observed = group[observed_col].ffill()

            fillable = value.isna() & previous_value.notna() & previous_observed.notna()
            if delta is not None:
                age = current_time - previous_observed
                fillable &= (age >= pd.Timedelta(0)) & (age <= delta)

            if fillable.any():
                group.loc[fillable, col] = previous_value.loc[fillable]
                group.loc[fillable, observed_col] = previous_observed.loc[fillable]
                if stage_label is not None:
                    group.loc[fillable, stage_column(col)] = stage_label

        parts.append(group)

    if not parts:
        return df

    filled = pd.concat(parts)
    return filled.reindex(order)


def drop_lookback_helpers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the hidden last-observation timestamps before writing a table."""
    helpers = [c for c in df.columns if c.startswith("__observed_time__")]
    return df.drop(columns=helpers) if helpers else df


def assume_zero_for_missing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Impute missing therapy/exposure variables as zero.

    This is the same rule that :func:`final_impute` applies to every feature
    whose ``final_strategy`` is ``"zero"``, so calling it early is redundant for
    the imputation itself and only changes what the Table 1 missingness count
    can see. It is therefore invoked only under
    ``--legacy-zero-before-missingness``; see :func:`preprocess_vital_table`.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def calculate_denominators(
    df: pd.DataFrame,
    adm_col: str,
    split_col: str,
    expected_train_n: int | None,
    expected_test_n: int | None,
    strict_expected_n: bool,
) -> dict[str, int]:
    """Calculate split denominators from unique ICU admissions and optionally validate them."""
    require_columns(df, [adm_col, split_col], "denominator input")
    denom = df.dropna(subset=[adm_col, split_col]).groupby(split_col)[adm_col].nunique().to_dict()
    denom = {str(k): int(v) for k, v in denom.items()}

    expected = {"train": expected_train_n, "test": expected_test_n}
    for split_name, expected_value in expected.items():
        if expected_value is None:
            continue
        observed = denom.get(split_name)
        if observed != expected_value:
            message = (
                f"{split_name} denominator differs from Online Supplemental Table 1: "
                f"observed {observed}, expected {expected_value}"
            )
            if strict_expected_n:
                raise AssertionError(message)
            LOGGER.warning(message)
    return denom


def restrict_assessment_window(
    df: pd.DataFrame,
    spec: FeatureSpec,
    time_col: str,
    icu_adm_col: str,
    assessment_window_hours: float | None,
    urine_start_hour: float,
) -> pd.DataFrame:
    """Select rows used for admission-level residual-missingness counting."""
    if time_col not in df.columns or icu_adm_col not in df.columns:
        LOGGER.warning(
            "'%s' or '%s' is absent, so the Table 1 assessment window could not be applied to "
            "%s; residual missingness is counted over every row of the stay instead",
            time_col, icu_adm_col, spec.name,
        )
        return df

    mask = df[time_col] >= df[icu_adm_col]
    if assessment_window_hours is not None:
        mask &= df[time_col] <= (df[icu_adm_col] + pd.Timedelta(hours=assessment_window_hours))
    if spec.name == "Urine_output_8H":
        mask &= df[time_col] >= (df[icu_adm_col] + pd.Timedelta(hours=urine_start_hour))
    return df.loc[mask].copy()


def admission_level_missingness(
    df: pd.DataFrame,
    specs: list[FeatureSpec],
    adm_col: str,
    split_col: str,
    time_col: str,
    icu_adm_col: str,
    denominators: dict[str, int],
    assessment_window_hours: float | None,
    urine_start_hour: float,
) -> pd.DataFrame:
    """Count unique ICU admissions with any residual missing record by split."""
    rows = []
    for spec in specs:
        if spec.name not in df.columns:
            for split_name, denominator in denominators.items():
                rows.append(
                    {
                        "feature": spec.name,
                        "display_name": spec.display_name,
                        "split": split_name,
                        "denominator": denominator,
                        "n_missing_adm_ICU_id": np.nan,
                        "percent_missing": np.nan,
                        "n_percent": "column absent",
                    }
                )
            continue

        df_window = restrict_assessment_window(
            df, spec, time_col, icu_adm_col, assessment_window_hours, urine_start_hour
        )
        for split_name, denominator in denominators.items():
            df_split = df_window[df_window[split_col] == split_name]
            n_missing = int(df_split.loc[df_split[spec.name].isna(), adm_col].dropna().nunique())
            pct = (n_missing / denominator * 100) if denominator else math.nan
            rows.append(
                {
                    "feature": spec.name,
                    "display_name": spec.display_name,
                    "split": split_name,
                    "denominator": denominator,
                    "n_missing_adm_ICU_id": n_missing,
                    "percent_missing": round(pct, 2) if pd.notna(pct) else np.nan,
                    "n_percent": f"{n_missing} ({pct:.2f}%)" if pd.notna(pct) else "NA",
                }
            )
    return pd.DataFrame(rows)


def training_means(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_col: str,
) -> dict[str, float]:
    """Calculate final mean-imputation values using the training cohort only."""
    train = df[df[split_col] == "train"]
    means: dict[str, float] = {}
    for col in feature_cols:
        if col not in train.columns:
            continue
        value = pd.to_numeric(train[col], errors="coerce").mean()
        if pd.notna(value):
            means[col] = float(value)
    return means


def final_impute(
    df: pd.DataFrame,
    specs: list[FeatureSpec],
    split_col: str,
    rng: np.random.Generator,
    use_published_final_means: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the final imputation step after temporal carry-forward/look-back."""
    df = df.copy()
    mean_features = [spec.name for spec in specs if spec.final_strategy == "train_mean" and spec.name in df.columns]
    calculated_means = training_means(df, mean_features, split_col)
    if use_published_final_means:
        final_means = {k: v for k, v in PUBLISHED_FINAL_MEANS.items() if k in df.columns}
        LOGGER.warning("using published Table 1 final means instead of recalculating training means")
    else:
        final_means = calculated_means

    report_rows = []
    for spec in specs:
        if spec.name not in df.columns:
            continue
        before_missing = int(df[spec.name].isna().sum())

        if spec.final_strategy == "zero":
            df[spec.name] = pd.to_numeric(df[spec.name], errors="coerce").fillna(0)
            final_description = "Assumed zero"

        elif spec.final_strategy == "train_mean":
            mean_value = final_means.get(spec.name)
            if mean_value is None or pd.isna(mean_value):
                LOGGER.warning("no training mean available for %s; missing values remain", spec.name)
                final_description = "training mean unavailable"
            else:
                df[spec.name] = pd.to_numeric(df[spec.name], errors="coerce").fillna(mean_value)
                final_description = f"mean value: {mean_value:.2f}"

        elif spec.final_strategy == "random_reference_range":
            if spec.reference_lower is None or spec.reference_upper is None:
                raise ValueError(f"reference range is missing for {spec.name}")
            df[spec.name] = pd.to_numeric(df[spec.name], errors="coerce")
            mask = df[spec.name].isna()
            if mask.any():
                sampled = rng.uniform(spec.reference_lower, spec.reference_upper, size=int(mask.sum()))
                df.loc[mask, spec.name] = sampled
            final_description = (
                "randomly sampled values within the physiological reference range "
                f"({spec.reference_lower:g}-{spec.reference_upper:g})"
            )

        else:
            raise ValueError(f"unknown final imputation strategy: {spec.final_strategy}")

        after_missing = int(df[spec.name].isna().sum())
        report_rows.append(
            {
                "feature": spec.name,
                "final_strategy": spec.final_strategy,
                "final_imputation": final_description,
                "n_missing_before_final_imputation": before_missing,
                "n_missing_after_final_imputation": after_missing,
                "calculated_train_mean": calculated_means.get(spec.name, np.nan),
                "published_table1_mean": PUBLISHED_FINAL_MEANS.get(spec.name, np.nan),
            }
        )
    return df, pd.DataFrame(report_rows)


def build_table1_output(
    missingness: pd.DataFrame,
    final_report: pd.DataFrame,
    specs: list[FeatureSpec],
) -> pd.DataFrame:
    """Build a Table 1-style output table from the audit summaries."""
    final_map = final_report.set_index("feature")["final_imputation"].to_dict() if not final_report.empty else {}
    rows = []
    for spec in specs:
        feature_missing = missingness[missingness["feature"] == spec.name]
        train_value = "column absent"
        test_value = "column absent"
        if not feature_missing.empty:
            train_rows = feature_missing[feature_missing["split"] == "train"]
            test_rows = feature_missing[feature_missing["split"] == "test"]
            if not train_rows.empty:
                train_value = str(train_rows.iloc[0]["n_percent"])
            if not test_rows.empty:
                test_value = str(test_rows.iloc[0]["n_percent"])
        rows.append(
            {
                "Features": spec.display_name,
                "source_column": spec.name,
                "Train Cohort n(%)": train_value,
                "Test Cohort n(%)": test_value,
                "Imputation procedure": spec.procedure,
                "Final imputation (train cohort)": final_map.get(spec.name, "column absent"),
                "min-max interval": spec.min_max_interval,
            }
        )
    return pd.DataFrame(rows)


def compare_with_published_means(final_report: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Compare calculated training means with the values printed in Table 1."""
    rows = []
    for feature, published in PUBLISHED_FINAL_MEANS.items():
        row = final_report[final_report["feature"] == feature]
        calculated = np.nan if row.empty else row.iloc[0]["calculated_train_mean"]
        difference = np.nan if pd.isna(calculated) else float(calculated) - published
        rows.append(
            {
                "feature": feature,
                "calculated_train_mean": calculated,
                "published_table1_mean": published,
                "difference": difference,
                "within_tolerance": bool(pd.notna(difference) and abs(difference) <= tolerance),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Pipeline.
# --------------------------------------------------------------------------- #


def preprocess_vital_table(args: argparse.Namespace, cohort: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vital = rename_aliases(load_table(args.vital_input))
    require_columns(vital, [args.adm_col], "vital input")
    vital[args.adm_col] = normalise_id(vital[args.adm_col])
    vital = merge_cohort_metadata(
        vital,
        cohort,
        adm_col=args.adm_col,
        split_col=args.split_col,
        metadata_cols=[
            args.icu_adm_col, args.icu_dis_col, args.patient_col, args.group_col,
            "label", "infection", "sepsis_onset", "infection_onset",
            "infection_index_time", "infection_index_source",
        ],
        overwrite_split=args.overwrite_split,
    )
    vital = convert_datetime(vital, [args.time_col, args.icu_adm_col, args.icu_dis_col])

    zero_cols = [spec.name for spec in FEATURE_SPECS if spec.category == "zero_assumed"]
    apache_cols = [spec.name for spec in FEATURE_SPECS if spec.category == "apache"]
    vital_cols = [spec.name for spec in FEATURE_SPECS if spec.category == "vital"]

    vital = convert_numeric(vital, apache_cols + vital_cols + zero_cols)

    # The zero assumption for therapy/exposure variables is applied by
    # final_impute, not here. Filling them at this point would make the Table 1
    # residual-missingness count structurally meaningless for those four rows:
    # every value is non-null by the time the count runs, so the table reports
    # 0 (0.00%) no matter how much of the source data was actually absent.
    # --legacy-zero-before-missingness restores the earlier ordering for anyone
    # reproducing output generated by the previous version of this script.
    if args.legacy_zero_before_missingness:
        LOGGER.warning(
            "--legacy-zero-before-missingness: therapy/exposure variables are zero-filled "
            "before the missingness count, so %s will report 0 (0.00%%) by construction",
            ", ".join(zero_cols),
        )
        vital = assume_zero_for_missing(vital, zero_cols)

    vital, clipping_report = clip_features(vital, [spec for spec in FEATURE_SPECS if spec.category != "lab"])

    if args.group_col not in vital.columns:
        LOGGER.warning("%s is absent; using %s for temporal carry-forward", args.group_col, args.adm_col)
        temporal_group_col = args.adm_col
    else:
        temporal_group_col = args.group_col

    # APACHE II is carried forward without an additional time limit, because it is an admission severity score.
    vital = forward_fill_within_hours(
        vital, temporal_group_col, args.time_col, apache_cols, hours=None, stage_label="locf",
    )
    vital = forward_fill_within_hours(
        vital, temporal_group_col, args.time_col, vital_cols,
        hours=args.vital_ffill_hours, stage_label="ffill_8h",
    )

    denominators = calculate_denominators(
        vital,
        args.adm_col,
        args.split_col,
        args.expected_train_n,
        args.expected_test_n,
        args.strict_expected_n,
    )
    missingness = admission_level_missingness(
        vital,
        [spec for spec in FEATURE_SPECS if spec.category in {"apache", "vital", "zero_assumed"}],
        args.adm_col,
        args.split_col,
        args.time_col,
        args.icu_adm_col,
        denominators,
        args.assessment_window_hours,
        args.urine_start_hour,
    )
    return vital, clipping_report, missingness


def preprocess_lab_table(args: argparse.Namespace, cohort: pd.DataFrame | None) -> tuple[pd.DataFrame | None, pd.DataFrame, pd.DataFrame]:
    if not args.lab_input:
        empty_report = pd.DataFrame()
        empty_missingness = pd.DataFrame()
        return None, empty_report, empty_missingness

    lab = rename_aliases(load_table(args.lab_input))
    require_columns(lab, [args.adm_col], "lab input")
    lab[args.adm_col] = normalise_id(lab[args.adm_col])
    lab = merge_cohort_metadata(
        lab,
        cohort,
        adm_col=args.adm_col,
        split_col=args.split_col,
        metadata_cols=[
            args.icu_adm_col, args.icu_dis_col, args.patient_col, args.group_col,
            "label", "infection", "sepsis_onset", "infection_onset",
            "infection_index_time", "infection_index_source",
        ],
        overwrite_split=args.overwrite_split,
    )
    lab = harmonise_pao2_fio2_columns(lab, args.pao2_col, args.fio2_col)
    lab = convert_datetime(lab, [args.time_col, args.icu_adm_col, args.icu_dis_col])

    lab_specs = [spec for spec in FEATURE_SPECS if spec.category == "lab"]
    direct_lab_specs = [spec for spec in lab_specs if spec.name != PFRATIO_COL]
    direct_lab_cols = [spec.name for spec in direct_lab_specs]
    lab_lookback_cols = [col for col in direct_lab_cols + [PAO2_COL] if col in lab.columns]

    # Convert direct laboratory variables and PaO2. P/F ratio is intentionally not
    # calculated until after PaO2 has completed the 72-hour and 168-hour look-back.
    lab = convert_numeric(lab, lab_lookback_cols + ([FIO2_COL] if FIO2_COL in lab.columns else []))
    lab, clipping_report_direct = clip_features(lab, direct_lab_specs)

    if args.group_col not in lab.columns:
        LOGGER.warning("%s is absent in lab input; using %s for laboratory look-back", args.group_col, args.adm_col)
        temporal_group_col = args.adm_col
    else:
        temporal_group_col = args.group_col

    # Two-stage look-back. Both stages measure the age of the value from the last
    # genuine measurement, so the second stage cannot chain onto a value that the
    # first stage imputed; the effective limit is exactly the stage-2 window.
    lab = forward_fill_within_hours(
        lab, temporal_group_col, args.time_col, lab_lookback_cols,
        hours=args.lab_lookback_stage1_hours, stage_label="stage1",
    )
    lab = forward_fill_within_hours(
        lab, temporal_group_col, args.time_col, lab_lookback_cols,
        hours=args.lab_lookback_stage2_hours, stage_label="stage2",
    )

    # Derive and clip P/F ratio after PaO2 look-back. This replaces the older
    # notebook fallback that renamed lab_Pao2 directly to PFratio.
    lab = derive_pfratio_after_pao2_lookback(lab)
    lab, clipping_report_pfratio = clip_features(lab, [spec for spec in lab_specs if spec.name == PFRATIO_COL])
    clipping_report = pd.concat([clipping_report_direct, clipping_report_pfratio], ignore_index=True)

    denominators = calculate_denominators(
        lab,
        args.adm_col,
        args.split_col,
        args.expected_train_n,
        args.expected_test_n,
        args.strict_expected_n,
    )
    missingness = admission_level_missingness(
        lab,
        lab_specs,
        args.adm_col,
        args.split_col,
        args.time_col,
        args.icu_adm_col,
        denominators,
        args.assessment_window_hours,
        args.urine_start_hour,
    )
    return lab, clipping_report, missingness


def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cohort = None
    if args.cohort_input:
        cohort = rename_aliases(load_table(args.cohort_input))
        require_columns(cohort, [args.adm_col, args.split_col], "cohort input")
        cohort[args.adm_col] = normalise_id(cohort[args.adm_col])
        cohort = normalise_split_column(cohort, args.split_col)
        cohort = convert_datetime(cohort, [args.icu_adm_col, args.icu_dis_col])

    vital, vital_clip_report, vital_missingness = preprocess_vital_table(args, cohort)
    lab, lab_clip_report, lab_missingness = preprocess_lab_table(args, cohort)

    # Final imputation is performed separately for the vital/intervention table and the laboratory table.
    rng = np.random.default_rng(args.random_seed)
    vital_specs = [spec for spec in FEATURE_SPECS if spec.category != "lab"]
    vital_final, vital_final_report = final_impute(
        vital,
        vital_specs,
        args.split_col,
        rng,
        use_published_final_means=args.use_published_final_means,
    )

    if lab is not None:
        lab_specs = [spec for spec in FEATURE_SPECS if spec.category == "lab"]
        lab_final, lab_final_report = final_impute(
            lab,
            lab_specs,
            args.split_col,
            rng,
            use_published_final_means=args.use_published_final_means,
        )
    else:
        lab_final = None
        lab_final_report = pd.DataFrame()

    missingness = pd.concat([vital_missingness, lab_missingness], ignore_index=True)
    final_report = pd.concat([vital_final_report, lab_final_report], ignore_index=True)
    clipping_report = pd.concat(
        [
            vital_clip_report.assign(source="vital"),
            lab_clip_report.assign(source="lab") if not lab_clip_report.empty else lab_clip_report,
        ],
        ignore_index=True,
    )
    table1 = build_table1_output(missingness, final_report, FEATURE_SPECS)
    mean_check = compare_with_published_means(final_report, args.mean_check_tolerance)

    write_csv(drop_lookback_helpers(vital_final), outdir / "vital_features_preprocessed.csv")
    if lab_final is not None:
        write_csv(drop_lookback_helpers(lab_final), outdir / "lab_features_preprocessed.csv")
    write_csv(table1, outdir / "online_supplemental_table1_reproduction.csv")
    write_csv(missingness, outdir / "residual_missingness_by_split.csv")
    write_csv(final_report, outdir / "final_imputation_report.csv")
    write_csv(clipping_report, outdir / "physiological_clipping_report.csv")
    write_csv(mean_check, outdir / "published_final_mean_check.csv")

    run_config = vars(args).copy()
    run_config["feature_specs"] = [asdict(spec) for spec in FEATURE_SPECS]
    (outdir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    LOGGER.info("wrote Table 1 reproduction to %s", outdir / "online_supplemental_table1_reproduction.csv")
    LOGGER.info("wrote final vital features to %s", outdir / "vital_features_preprocessed.csv")
    if lab_final is not None:
        LOGGER.info("wrote final laboratory features to %s", outdir / "lab_features_preprocessed.csv")


# --------------------------------------------------------------------------- #
# Command-line interface.
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply clipping and imputation rules for Online Supplemental Table 1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vital-input", required=True, help="CSV/TSV/Excel file containing vital-sign and non-lab features")
    parser.add_argument("--lab-input", default=None, help="optional CSV/TSV/Excel file containing laboratory features")
    parser.add_argument("--cohort-input", default=None, help="optional Step 3 cohort file containing adm_ICU_id and split")
    parser.add_argument("--outdir", default="output/step4_preprocessing", help="output directory")

    parser.add_argument("--adm-col", default="adm_ICU_id", help="ICU admission identifier column")
    parser.add_argument("--patient-col", default="Pno", help="patient identifier column")
    parser.add_argument("--group-col", default="adm_ICU_id", help="grouping column for temporal carry-forward/look-back")
    parser.add_argument("--split-col", default="split", help="fixed train/test split column")
    parser.add_argument("--time-col", default="date", help="measurement timestamp column")
    parser.add_argument("--icu-adm-col", default="ICU_admdatetime", help="ICU admission timestamp column")
    parser.add_argument("--icu-dis-col", default="ICU_disdatetime", help="ICU discharge timestamp column")
    parser.add_argument("--overwrite-split", action="store_true", help="replace an existing split column with the one in --cohort-input")

    parser.add_argument("--pao2-col", default="lab_Pao2", help="source PaO2 column; PaO2 is look-back imputed before deriving PFratio")
    # NOTE: argparse applies %-formatting to every help string, so a literal
    # percent sign must be written as '%%' or --help raises/renders garbage.
    parser.add_argument("--fio2-col", default="FiO2", help="source FiO2 column; missing values are defaulted to 21%% room air for PFratio")

    parser.add_argument("--vital-ffill-hours", type=float, default=8.0, help="forward-fill window for vital signs")
    parser.add_argument("--lab-lookback-stage1-hours", type=float, default=72.0, help="first laboratory look-back window")
    parser.add_argument("--lab-lookback-stage2-hours", type=float, default=168.0, help="second laboratory look-back window")
    parser.add_argument(
        "--assessment-window-hours",
        type=float,
        default=8.0,
        help="Table 1 missingness window after ICU admission; default 8 hours. Use a negative value only for sensitivity checks.",
    )
    parser.add_argument("--urine-start-hour", type=float, default=4.0, help="urine output missingness is assessed after this many hours")

    # No default: these are cohort sizes specific to the originating study, and
    # hard-coding them would make every other site's run emit a mismatch warning
    # on start-up. The published values are 11316 / 2837; pass them explicitly to
    # check a reproduction against Online Supplemental Table 1.
    parser.add_argument("--expected-train-n", type=int, default=None, help="expected train admission count; the published value is 11316")
    parser.add_argument("--expected-test-n", type=int, default=None, help="expected test admission count; the published value is 2837")
    parser.add_argument(
        "--legacy-zero-before-missingness",
        action="store_true",
        help="apply the zero assumption for therapy/exposure variables before counting "
             "residual missingness, as the previous version did. This forces those rows of "
             "Table 1 to 0 (0.00%%) regardless of the data and is provided only to reproduce "
             "output from that version",
    )
    parser.add_argument("--strict-expected-n", action="store_true", help="fail if observed denominators differ from expected Table 1 counts")
    parser.add_argument("--random-seed", type=int, default=42, help="seed for random reference-range laboratory imputation")
    parser.add_argument(
        "--use-published-final-means",
        action="store_true",
        help="use the final means printed in Table 1 rather than recalculating from the training cohort",
    )
    parser.add_argument("--mean-check-tolerance", type=float, default=0.01, help="tolerance for comparing calculated means with published Table 1 means")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.assessment_window_hours is not None and args.assessment_window_hours < 0:
        args.assessment_window_hours = None

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
