#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (C) 2026 Fang-Ju Sun and contributors

"""
Step 05 - Construction of 8-hour feature-window samples for the early
prediction of sepsis and infection in the ICU.

==========================================================================
Timeline (Online Supplemental Figure 4)
==========================================================================

     t-8            t              t+8          t+9
      |--------------|--------------|------------|
      | feature (8h) |  lead  (8h)  | pred. (1h) |

    Feature window    : (t-8, t]     -> 8 hourly rows: t-7, t-6, ..., t
                                        (the row at t IS included)
    Lead-time window  : (t, t+8]     -> never used as model input
    Prediction window : (t+8, t+9]   -> the onset must fall here

Anchoring the prediction time t
-------------------------------
The task-specific index time T is already available upstream for every ICU
admission:

    * sepsis task   : ``sepsis_onset`` is the true onset for cases and the
                      seeded pseudo-index time for controls.
    * infection task: ``infection_index_time`` is the true culture-confirmed
                      infection onset for cases and the seeded sepsis index
                      time for controls. ``infection_onset`` remains empty for
                      infection-negative stays and is not the default anchor.

Because cases and controls both carry a task-specific index timestamp, t is
derived with one rule for both groups, avoiding a systematic difference in
time-since-admission:

    t = ceil_to_hour(T) - (lead_hours + prediction_hours)        [figure4]

so that T always satisfies T in (t + 8, t + 9].  Example: T = 20:30 gives
t = 12:00, t+8 = 20:00, t+9 = 21:00, and 20:30 lies inside (20:00, 21:00].

    t = floor_to_hour(T) - lead_hours                            [minus_lead]

is retained as an explicit compatibility/sensitivity mode. The default
``figure4`` rule enforces the complete 8-hour lead interval followed by the
1-hour prediction interval.

Because t is taken from the onset timestamp rather than from the last row
carrying ``label == 1``, the feature window always ends *before* the first
onset and can therefore never contain post-onset, post-treatment signals
(antibiotics, vasopressors, rising lactate).

Missing-value handling
----------------------
* Forward fill only.  Backward fill is never applied anywhere in this file,
  because it would carry information from the future into the feature window.
* Forward fill is applied to feature columns only, never to the admission id,
  the timestamp, the label column or the onset columns.
* Forward fill is time limited (``ffill_limit``), so a measurement taken many
  hours earlier is not presented to the model as a current value.
* A ``<column>__was_missing`` indicator records whether the value was actually
  charted at that hour, since missingness itself carries information.

Assumptions inherited from the upstream pipeline
------------------------------------------------
The input table is already resampled to a strict one-row-per-hour grid per
ICU admission.  This is verified and reported, not re-created; set
``--strict-hourly`` if offending admissions should be excluded instead of
merely logged.

Step 4 writes the vital/intervention table and the laboratory table separately,
both keyed on ``adm_ICU_id`` with the measurement timestamp in a column named
``date``.  Joining them onto one hourly grid is a local step and is not part of
this file.  The resulting table is what ``--input`` expects, and because this
script defaults to ``charttime``, a table carrying the step-4 timestamp column
must be run with ``--time-col date``.

Every column that is not a recognised identifier, timestamp, label or onset is
treated as a model feature and forward filled.  Metadata is recognised either by
name (``never_ffill``) or by suffix (``never_ffill_suffixes``), the latter
covering the ``<feature>__lookback_stage`` audit columns emitted by step 4 and
any ``<name>_cohort`` duplicates left by a merge.  Anything else added to the
input table upstream will be filled and reported as a feature.

Usage
-----
    python 01_05_build_feature_windows.py --self-test
    python 01_05_build_feature_windows.py --input hourly_icu.parquet --task both
    python 01_05_build_feature_windows.py --input hourly_icu.csv --time-col date

Author  : Fang-Ju Sun
License : PolyForm-Noncommercial-1.0.0
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("feature_window")

__all__ = [
    "WindowConfig",
    "prediction_time",
    "build_samples",
    "summarize",
    "to_sequence_array",
]


# ==========================================================================
# Configuration
# ==========================================================================
@dataclass
class WindowConfig:
    """All tunable parameters of the sample-construction procedure."""

    # ---- column names -------------------------------------------------
    id_col: str = "adm_ICU_id"
    time_col: str = "charttime"
    onset_col: str = "sepsis_onset"   # true onset (cases) / pseudo-onset (controls)
    onset_fallback_col: Optional[str] = None
    label_col: str = "label"          # 1 = event group; used for grouping and audit
    task_name: str = "sepsis"         # written into the output tables

    # ---- timeline (Supplemental Figure 4) -----------------------------
    feature_hours: int = 8
    lead_hours: int = 8
    prediction_hours: int = 1

    # "figure4"    : t = ceil_to_hour(T)  - (lead + prediction)  -> T in (t+8, t+9]
    # "minus_lead" : t = floor_to_hour(T) - lead                 -> legacy reading
    t_rule: str = "figure4"

    # True  -> feature window (t-8, t]  = rows t-7 ... t   (Figure 4)
    # False -> feature window [t-8, t)  = rows t-8 ... t-1 (previous code)
    include_t: bool = True

    # ---- missing-value handling ---------------------------------------
    ffill_limit: Optional[int] = 8
    add_missing_indicator: bool = True
    add_staleness: bool = False       # adds <col>__hours_since_measured
    never_ffill: Sequence[str] = field(
        default_factory=lambda: (
            "Pno", "Firstcaseno", "Caseno", "Bedns", "Bedno",
            "ICU_admdatetime", "ICU_disdatetime", "split",
            "label", "sepsis_label", "infection", "infection_label",
            "culture_report", "sepsis_onset", "infection_onset",
            "infection_index_time", "infection_index_source",
        )
    )

    # Any column whose name ends with one of these is metadata, not a feature.
    # Listing individual names is not enough: step 4 emits one
    # `<feature>__lookback_stage` audit column per imputed variable, and a merge
    # against the cohort file can leave `<name>_cohort` duplicates of the
    # identifier and admission timestamps. Neither set is known in advance, and
    # anything not recognised here is forward filled and reported as a feature.
    never_ffill_suffixes: Sequence[str] = field(
        default_factory=lambda: (
            "__lookback_stage", "_cohort", "__was_missing", "__hours_since_measured",
        )
    )

    # ---- data-quality policy ------------------------------------------
    verify_hourly_grid: bool = True
    strict_hourly: bool = False       # True -> exclude irregular admissions

    def __post_init__(self) -> None:
        if self.feature_hours < 1:
            raise ValueError("feature_hours must be >= 1")
        if self.t_rule not in {"figure4", "minus_lead"}:
            raise ValueError("t_rule must be 'figure4' or 'minus_lead'")

    # ------------------------------------------------------------------
    @property
    def protected_cols(self) -> Tuple[str, ...]:
        """Columns that must never be forward filled."""
        cols = [
            self.id_col,
            self.time_col,
            self.onset_col,
            self.label_col,
            *( [self.onset_fallback_col] if self.onset_fallback_col else [] ),
            *self.never_ffill,
        ]
        return tuple(dict.fromkeys(cols))

    @property
    def window_offsets(self) -> Tuple[int, int]:
        """Positional offsets (start, stop_exclusive) relative to t."""
        if self.include_t:
            return -self.feature_hours + 1, 1
        return -self.feature_hours, 0

    def describe(self) -> str:
        lo, hi = self.window_offsets
        bound = "t]" if self.include_t else "t)"
        return (
            f"task={self.task_name} | onset={self.onset_col}"
            + (f" (fallback {self.onset_fallback_col})" if self.onset_fallback_col else "")
            + " | "
            f"feature window (t-{self.feature_hours}, {bound} "
            f"= offsets {lo}..{hi - 1} ({self.feature_hours} rows) | "
            f"lead={self.lead_hours}h | prediction={self.prediction_hours}h | "
            f"t_rule={self.t_rule} | ffill_limit={self.ffill_limit}"
        )


# ==========================================================================
# Prediction-time anchoring
# ==========================================================================
def normalise_id(series: pd.Series) -> pd.Series:
    """Strip whitespace and a trailing ``.0`` left by a float round-trip.

    Steps 1 to 4 read every table with ``dtype=str`` and apply this to each
    identifier, so that the join keys agree whichever source the table came
    from. This step must do the same. A plain ``read_csv`` infers a numeric
    dtype for a numeric admission id, and float64 as soon as the column has a
    single missing value, so ``12`` is read back as ``12.0``; the sample id then
    becomes ``sepsis__12.0`` and no longer matches the cohort tables written by
    the earlier steps.
    """
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def _ceil_to_hour(ts: pd.Timestamp) -> pd.Timestamp:
    floored = ts.floor("h")
    return floored if floored == ts else floored + pd.Timedelta(hours=1)


def prediction_time(onset: pd.Timestamp, cfg: WindowConfig) -> pd.Timestamp:
    """Map an onset timestamp T to the prediction time t of Figure 4."""
    if cfg.t_rule == "figure4":
        return _ceil_to_hour(onset) - pd.Timedelta(
            hours=cfg.lead_hours + cfg.prediction_hours
        )
    return onset.floor("h") - pd.Timedelta(hours=cfg.lead_hours)


# ==========================================================================
# Per-admission preparation
# ==========================================================================
def _feature_columns(stay: pd.DataFrame, cfg: WindowConfig) -> List[str]:
    """Columns treated as model features, i.e. everything that is not metadata."""
    suffixes = tuple(cfg.never_ffill_suffixes)
    return [
        c for c in stay.columns
        if c not in cfg.protected_cols and not (suffixes and c.endswith(suffixes))
    ]


def _prepare_stay(stay: pd.DataFrame, cfg: WindowConfig) -> pd.DataFrame:
    """Sort, de-duplicate and forward fill one ICU admission.

    The missing indicators are computed on the raw values, i.e. before the
    forward fill, so that they describe what was actually charted at that hour.
    """
    stay = stay.copy()
    stay[cfg.time_col] = pd.to_datetime(stay[cfg.time_col])
    stay = (
        stay.sort_values(cfg.time_col, kind="mergesort")
        .drop_duplicates(subset=[cfg.time_col], keep="last")
        .reset_index(drop=True)
    )

    feature_cols = _feature_columns(stay, cfg)
    if not feature_cols:
        return stay

    raw_missing = stay[feature_cols].isna()
    extra: List[pd.DataFrame] = []

    if cfg.add_missing_indicator:
        indicator = raw_missing.astype("int8")
        indicator.columns = [f"{c}__was_missing" for c in feature_cols]
        extra.append(indicator)

    if cfg.add_staleness:
        # Hours elapsed since the column was last actually measured.
        stale: Dict[str, np.ndarray] = {}
        row_idx = np.arange(len(stay))
        for col in feature_cols:
            observed = np.flatnonzero(~raw_missing[col].to_numpy())
            hours = np.full(len(stay), np.nan)
            if observed.size:
                pos = np.searchsorted(observed, row_idx, side="right") - 1
                valid = pos >= 0
                hours[valid] = row_idx[valid] - observed[pos[valid]]
            stale[f"{col}__hours_since_measured"] = hours
        extra.append(pd.DataFrame(stale, index=stay.index))

    # Forward fill only: past -> future.  bfill would leak the future.
    stay[feature_cols] = stay[feature_cols].ffill(limit=cfg.ffill_limit)

    if extra:
        stay = pd.concat([stay, *extra], axis=1)
    return stay


def _hourly_grid_ok(stay: pd.DataFrame, cfg: WindowConfig) -> bool:
    """Confirm that consecutive rows are exactly one hour apart."""
    if len(stay) < 2:
        return True
    deltas = stay[cfg.time_col].diff().dropna().dt.total_seconds() / 3600.0
    return bool(np.all(deltas == 1.0))


# ==========================================================================
# Main builder
# ==========================================================================
def build_samples(
    df: pd.DataFrame,
    cfg: Optional[WindowConfig] = None,
    stay_ids: Optional[Sequence] = None,
    progress: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Build one 8-hour feature window per ICU admission.

    Returns a dict with
        ``windows``    : long table, ``feature_hours`` rows per sample
        ``samples``    : one row per sample (id, t, onset, label, group)
        ``exclusions`` : one row per excluded admission, with the reason
    """
    cfg = cfg or WindowConfig()

    for col in (cfg.id_col, cfg.time_col, cfg.onset_col):
        if col not in df.columns:
            raise KeyError(f"required column '{col}' is missing from the input table")
    if cfg.onset_fallback_col and cfg.onset_fallback_col not in df.columns:
        raise KeyError(
            f"onset fallback column '{cfg.onset_fallback_col}' is missing from the input table"
        )

    has_label = cfg.label_col in df.columns
    if not has_label:
        raise KeyError(
            f"label column '{cfg.label_col}' is missing. The sepsis and infection "
            "tasks require separate validated 0/1 outcome columns."
        )

    LOGGER.info("Configuration: %s", cfg.describe())

    if stay_ids is not None:
        requested = pd.Index(pd.unique(pd.Series(list(stay_ids))))
        missing_ids = requested.difference(pd.Index(df[cfg.id_col].unique()))
        df = df[df[cfg.id_col].isin(requested)]
    else:
        missing_ids = pd.Index([])

    groups = list(df.groupby(cfg.id_col, sort=False))
    iterator = groups
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(groups, desc=f"Building {cfg.task_name} windows")
        except ImportError:
            LOGGER.debug("tqdm not installed; continuing without a progress bar")

    lo, hi = cfg.window_offsets
    window_frames: List[pd.DataFrame] = []
    sample_rows: List[dict] = []
    exclusions: List[dict] = [
        {cfg.id_col: sid, "reason": "id_not_found_in_table"} for sid in missing_ids
    ]

    def drop(stay_id, reason: str) -> None:
        exclusions.append({cfg.id_col: stay_id, "reason": reason})

    for stay_id, raw_stay in iterator:
        stay = _prepare_stay(raw_stay, cfg)

        # ---- hourly-grid verification ---------------------------------
        if cfg.verify_hourly_grid and not _hourly_grid_ok(stay, cfg):
            if cfg.strict_hourly:
                drop(stay_id, "irregular_hourly_grid")
                continue
            LOGGER.warning("admission %s is not on a strict hourly grid", stay_id)

        # ---- onset timestamp ------------------------------------------
        # Event stays carry a clinical onset. Non-event stays have none, so they
        # fall back to the index time drawn in step 1 with a fixed seed. Without
        # this fallback the infection task would produce no negative samples at
        # all, because step 2 writes infection_onset for positive stays only.
        onset_values = pd.to_datetime(stay[cfg.onset_col], errors="coerce").dropna()
        onset_source = cfg.onset_col
        if onset_values.empty and cfg.onset_fallback_col:
            onset_values = pd.to_datetime(
                stay[cfg.onset_fallback_col], errors="coerce"
            ).dropna()
            onset_source = cfg.onset_fallback_col
        if onset_values.empty:
            drop(stay_id, "missing_onset_time")
            continue
        onset = onset_values.min()   # the first onset, never a later episode

        # ---- prediction time t ----------------------------------------
        t_time = prediction_time(onset, cfg)
        matches = np.flatnonzero(stay[cfg.time_col].to_numpy() == np.datetime64(t_time))
        if matches.size == 0:
            drop(stay_id, "t_outside_stay")
            continue
        t_pos = int(matches[0])

        if t_pos + lo < 0 or t_pos + hi > len(stay):
            drop(stay_id, "insufficient_history_before_t")
            continue

        # ---- group ----------------------------------------------------
        labels = pd.to_numeric(stay[cfg.label_col], errors="coerce")
        if labels.isna().any():
            raise ValueError(
                f"admission {stay_id}: label column '{cfg.label_col}' contains missing "
                "or non-numeric value(s)"
            )
        observed = sorted(pd.unique(labels.astype(int)))
        if not set(observed).issubset({0, 1}):
            raise ValueError(
                f"admission {stay_id}: label column '{cfg.label_col}' must contain 0/1; "
                f"observed {observed}"
            )
        is_event = int((labels == 1).any())
        group = "positive" if is_event else "negative"

        # ---- feature window -------------------------------------------
        window = stay.iloc[t_pos + lo : t_pos + hi].copy()
        if len(window) != cfg.feature_hours:   # defensive, should never trigger
            drop(stay_id, "unexpected_window_length")
            continue

        sample_id = f"{cfg.task_name}__{stay_id}"
        window.insert(0, "sample_id", sample_id)
        window["hour_from_t"] = np.arange(lo, hi, dtype=int)
        window["sample_group"] = group
        window["sample_label"] = is_event
        window["t_time"] = t_time
        window["onset_time"] = onset
        window["onset_source"] = onset_source
        window["task"] = cfg.task_name

        window_frames.append(window)
        sample_rows.append(
            {
                "sample_id": sample_id,
                cfg.id_col: stay_id,
                "task": cfg.task_name,
                "sample_group": group,
                "sample_label": is_event,
                "t_time": t_time,
                "onset_time": onset,
                "onset_source": onset_source,
                "lead_time_hours": (onset - t_time).total_seconds() / 3600.0,
                "window_start": window[cfg.time_col].iloc[0],
                "window_end": window[cfg.time_col].iloc[-1],
                "t_position": t_pos,
                "hours_in_icu_before_t": t_pos,
                "stay_length_hours": len(stay),
            }
        )

    windows = (
        pd.concat(window_frames, axis=0, ignore_index=True)
        if window_frames
        else pd.DataFrame()
    )
    return {
        "windows": windows,
        "samples": pd.DataFrame(sample_rows),
        "exclusions": pd.DataFrame(exclusions, columns=[cfg.id_col, "reason"]),
    }


# ==========================================================================
# Reporting
# ==========================================================================
def summarize(
    result: Dict[str, pd.DataFrame], cfg: Optional[WindowConfig] = None
) -> Dict[str, pd.DataFrame]:
    """Summary tables for the methods section and the participant flow diagram."""
    cfg = cfg or WindowConfig()
    windows = result["windows"]
    samples = result["samples"]
    exclusions = result["exclusions"]

    if samples.empty:
        raise ValueError("no samples were generated - check the configuration")

    n_pos = int((samples["sample_label"] == 1).sum())
    n_neg = int((samples["sample_label"] == 0).sum())

    sample_summary = pd.DataFrame(
        {
            "Group": ["Positive", "Negative", "Total"],
            "ICU_admissions": [n_pos, n_neg, n_pos + n_neg],
            "Rows_in_windows": [
                n_pos * cfg.feature_hours,
                n_neg * cfg.feature_hours,
                (n_pos + n_neg) * cfg.feature_hours,
            ],
        }
    )

    # Cases and controls should now be comparable on these variables; a large
    # difference would point to a residual sampling bias.
    balance = (
        samples.groupby("sample_group")[
            ["hours_in_icu_before_t", "stay_length_hours", "lead_time_hours"]
        ]
        .describe()
        .T.reset_index()
        .rename(columns={"level_0": "Variable", "level_1": "Statistic"})
    )

    counts = windows.groupby("sample_id").size()
    integrity = pd.DataFrame(
        {
            "Item": [
                "Rows in long table",
                "Expected rows",
                "Samples with wrong length",
                "Unique samples",
                "Unique ICU admissions",
                "Duplicated (sample_id, hour_from_t)",
                "Windows reaching the onset",
                "Lead time outside (8, 9] hours",
                "Excluded ICU admissions",
            ],
            "Count": [
                len(windows),
                (n_pos + n_neg) * cfg.feature_hours,
                int((counts != cfg.feature_hours).sum()),
                int(windows["sample_id"].nunique()),
                int(windows[cfg.id_col].nunique()),
                int(windows.duplicated(["sample_id", "hour_from_t"]).sum()),
                int((samples["window_end"] >= samples["onset_time"]).sum()),
                int(
                    (
                        (samples["lead_time_hours"] <= cfg.lead_hours)
                        | (
                            samples["lead_time_hours"]
                            > cfg.lead_hours + cfg.prediction_hours
                        )
                    ).sum()
                ),
                len(exclusions),
            ],
        }
    )

    exclusion_summary = (
        exclusions.groupby("reason").size().rename("Count").reset_index()
        if not exclusions.empty
        else pd.DataFrame(columns=["reason", "Count"])
    )

    onset_source_summary = (
        samples.groupby(["sample_group", "onset_source"]).size()
        .rename("Count").reset_index()
        if "onset_source" in samples.columns
        else pd.DataFrame(columns=["sample_group", "onset_source", "Count"])
    )

    meta = {
        "sample_id",
        "hour_from_t",
        "sample_group",
        "sample_label",
        "t_time",
        "onset_time",
        "onset_source",
        "task",
        *cfg.protected_cols,
    }
    # Same rule as _feature_columns, so the report describes exactly the columns
    # that were forward filled and nothing else.
    suffixes = tuple(cfg.never_ffill_suffixes)
    feature_cols = [
        c
        for c in windows.columns
        if c not in meta and not (suffixes and c.endswith(suffixes))
    ]
    missing_report = (
        windows[feature_cols]
        .isna()
        .mean()
        .mul(100)
        .round(2)
        .rename("Missing_percent_after_ffill")
        .reset_index()
        .rename(columns={"index": "Column"})
        .sort_values("Missing_percent_after_ffill", ascending=False, ignore_index=True)
        if feature_cols
        else pd.DataFrame(columns=["Column", "Missing_percent_after_ffill"])
    )

    return {
        "sample_summary": sample_summary,
        "group_balance": balance,
        "integrity": integrity,
        "exclusion_summary": exclusion_summary,
        "onset_source_summary": onset_source_summary,
        "missing_report": missing_report,
    }


def to_sequence_array(
    windows: pd.DataFrame,
    feature_cols: Sequence[str],
    cfg: Optional[WindowConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape the long table into ``(n_samples, feature_hours, n_features)``.

    Rows are ordered by ``hour_from_t``, so the time axis always runs from the
    oldest hour of the window to t.
    """
    cfg = cfg or WindowConfig()
    ordered = windows.sort_values(["sample_id", "hour_from_t"], kind="mergesort")

    counts = ordered.groupby("sample_id").size()
    if not (counts == cfg.feature_hours).all():
        raise ValueError("some samples do not contain exactly feature_hours rows")

    sample_ids = counts.index.to_numpy()
    x = ordered[list(feature_cols)].to_numpy(dtype=float).reshape(
        len(sample_ids), cfg.feature_hours, len(feature_cols)
    )
    y = ordered.groupby("sample_id")["sample_label"].first().to_numpy()
    return x, y, sample_ids


# ==========================================================================
# Self-test on synthetic data
# ==========================================================================
def make_synthetic_table(n_stays: int = 40, seed: int = 0) -> pd.DataFrame:
    """Synthetic hourly ICU table used by ``--self-test``."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_stays):
        n_hours = int(rng.integers(6, 60))
        start = pd.Timestamp("2020-01-01") + pd.Timedelta(
            hours=int(rng.integers(0, 500))
        )
        times = pd.date_range(start, periods=n_hours, freq="h")

        is_event = bool(rng.random() < 0.45)
        # Both groups receive an onset timestamp; for controls this is the
        # pseudo-onset drawn upstream with a fixed random seed.
        onset_hour = int(rng.integers(0, n_hours))
        onset = times[onset_hour] + pd.Timedelta(minutes=int(rng.integers(0, 60)))

        label = np.zeros(n_hours, dtype=int)
        if is_event:
            label[onset_hour:] = 1

        # Culture-confirmed infection: as in step 2, infection_onset exists for
        # positive stays only, so the negatives must rely on the fallback.
        is_infection = bool(rng.random() < 0.3)
        infection_label = np.full(n_hours, int(is_infection), dtype=int)

        heart_rate = rng.normal(85, 12, n_hours)
        heart_rate[rng.random(n_hours) < 0.35] = np.nan
        lactate = rng.normal(1.8, 0.6, n_hours)
        lactate[rng.random(n_hours) < 0.70] = np.nan

        frames.append(
            pd.DataFrame(
                {
                    "adm_ICU_id": f"ICU{i:03d}",
                    "charttime": times,
                    "heart_rate": heart_rate,
                    "lactate": lactate,
                    "label": label,
                    "infection": infection_label,
                    "sepsis_onset": onset,
                    "infection_onset": (
                        onset - pd.Timedelta(hours=2) if is_infection else pd.NaT
                    ),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _self_test() -> None:
    df = make_synthetic_table()
    cfg = WindowConfig(add_staleness=True)
    result = build_samples(df, cfg, progress=False)
    windows, samples = result["windows"], result["samples"]

    # shape and ordering of every window
    assert (windows.groupby("sample_id").size() == cfg.feature_hours).all()
    assert set(windows["hour_from_t"].unique()) == set(
        range(-cfg.feature_hours + 1, 1)
    )
    assert not windows.duplicated(["sample_id", "hour_from_t"]).any()

    # Figure-4 timing: the onset must lie in (t+8, t+9]
    lead = samples["lead_time_hours"]
    assert (
        (lead > cfg.lead_hours) & (lead <= cfg.lead_hours + cfg.prediction_hours)
    ).all()

    # no window may reach the onset, and every window must end exactly at t
    assert (samples["window_end"] < samples["onset_time"]).all()
    assert (samples["window_end"] == samples["t_time"]).all()

    # the label column must never be forward filled
    one = df[df["adm_ICU_id"] == "ICU000"]
    pd.testing.assert_series_equal(
        _prepare_stay(one, cfg)["label"].reset_index(drop=True),
        one.sort_values("charttime")["label"].reset_index(drop=True),
    )
    assert "heart_rate__was_missing" in windows.columns
    assert "heart_rate__hours_since_measured" in windows.columns

    x, y, ids = to_sequence_array(windows, ["heart_rate", "lactate"], cfg)
    assert x.shape == (len(ids), cfg.feature_hours, 2)
    assert y.shape == (len(ids),)

    # the legacy slice can still be reproduced for comparison
    legacy = WindowConfig(
        include_t=False,
        t_rule="minus_lead",
        ffill_limit=None,
        add_missing_indicator=False,
    )
    legacy_result = build_samples(df, legacy, progress=False)
    assert set(legacy_result["windows"]["hour_from_t"].unique()) == set(range(-8, 0))

    # ---- infection task: the negatives must survive via the fallback ------
    infection_cfg = WindowConfig(
        task_name="infection",
        onset_col="infection_onset",
        onset_fallback_col="sepsis_onset",
        label_col="infection",
    )
    infection = build_samples(df, infection_cfg, progress=False)
    inf_samples = infection["samples"]
    assert not inf_samples.empty
    # Regression for the bug where every control was dropped as
    # "missing_onset_time" because step 2 leaves infection_onset empty.
    assert (inf_samples["sample_label"] == 0).any(), "infection task produced no negatives"
    assert (inf_samples["sample_label"] == 1).any(), "infection task produced no positives"
    assert set(inf_samples.loc[inf_samples["sample_label"] == 1, "onset_source"]) == {
        "infection_onset"
    }
    assert set(inf_samples.loc[inf_samples["sample_label"] == 0, "onset_source"]) == {
        "sepsis_onset"
    }
    # Regression for the bug where the infection dataset carried the sepsis label.
    inf_label_by_stay = (
        df.groupby("adm_ICU_id")["infection"].max().rename("expected").reset_index()
    )
    check = inf_samples.merge(inf_label_by_stay, on="adm_ICU_id")
    assert (check["sample_label"] == check["expected"]).all(), "infection label mismatch"

    print("self-test passed\n")
    report = summarize(result, cfg)
    print(report["sample_summary"].to_string(index=False))
    print()
    print(report["integrity"].to_string(index=False))
    print()
    print(report["exclusion_summary"].to_string(index=False))


# ==========================================================================
# Command-line interface
# ==========================================================================
# Each prediction task has its own onset timestamp AND its own label column.
# Step 3 emits `label` for sepsis and `infection` for culture-confirmed
# infection; using `label` for both would silently attach the sepsis outcome to
# the infection dataset.
_TASKS: Dict[str, Dict[str, str]] = {
    "sepsis": {
        "onset_col": "sepsis_onset",
        "label_col": "label",
        "onset_fallback_col": "",          # step 1 fills sepsis_onset for every stay
    },
    "infection": {
        "onset_col": "infection_index_time",
        "label_col": "infection",
        "onset_fallback_col": "",  # Step 2 defines infection_index_time for both classes
    },
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build 8-hour feature-window samples (Supplemental Figure 4)."
    )
    parser.add_argument("--input", help="CSV or Parquet table with hourly ICU rows")
    parser.add_argument(
        "--task", choices=["sepsis", "infection", "both"], default="both"
    )
    parser.add_argument("--output-prefix", default="feature_window_8H")
    parser.add_argument("--id-col", default="adm_ICU_id")
    parser.add_argument("--time-col", default="charttime")
    parser.add_argument(
        "--label-col", default=None,
        help="override the label column implied by --task ('label' for sepsis, "
             "'infection' for infection)",
    )
    parser.add_argument(
        "--onset-col", default=None, help="override the onset column implied by --task"
    )
    parser.add_argument(
        "--onset-fallback-col", default=None,
        help="optional fallback timestamp used when the task's onset column is empty; "
             "the default infection workflow uses infection_index_time from step 2 and "
             "therefore needs no fallback",
    )
    parser.add_argument("--feature-hours", type=int, default=8)
    parser.add_argument("--lead-hours", type=int, default=8)
    parser.add_argument("--t-rule", choices=["figure4", "minus_lead"], default="figure4")
    parser.add_argument(
        "--legacy-window",
        action="store_true",
        help="use the previous [t-8, t-1] slice instead of (t-8, t]",
    )
    parser.add_argument(
        "--ffill-limit", type=int, default=8, help="use -1 for an unlimited forward fill"
    )
    parser.add_argument("--no-missing-indicator", action="store_true")
    parser.add_argument(
        "--staleness",
        action="store_true",
        help="add <col>__hours_since_measured columns",
    )
    parser.add_argument(
        "--strict-hourly",
        action="store_true",
        help="exclude admissions that are not on a strict hourly grid",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if args.self_test:
        _self_test()
        return 0
    if not args.input:
        parser.error("--input is required unless --self-test is used")

    if args.input.endswith((".parquet", ".pq")):
        df = pd.read_parquet(args.input)
    else:
        # dtype=str for the identifier only: the feature columns must stay numeric.
        df = pd.read_csv(args.input, dtype={args.id_col: "string"})

    if args.id_col not in df.columns:
        parser.error(f"identifier column '{args.id_col}' is not present in {args.input}")
    df[args.id_col] = normalise_id(df[args.id_col])

    tasks = list(_TASKS) if args.task == "both" else [args.task]
    for task in tasks:
        defaults = _TASKS[task]
        fallback = (
            defaults["onset_fallback_col"]
            if args.onset_fallback_col is None
            else args.onset_fallback_col
        )
        cfg = WindowConfig(
            id_col=args.id_col,
            time_col=args.time_col,
            onset_col=args.onset_col or defaults["onset_col"],
            onset_fallback_col=fallback or None,
            label_col=args.label_col or defaults["label_col"],
            task_name=task,
            feature_hours=args.feature_hours,
            lead_hours=args.lead_hours,
            t_rule=args.t_rule,
            include_t=not args.legacy_window,
            ffill_limit=None if args.ffill_limit < 0 else args.ffill_limit,
            add_missing_indicator=not args.no_missing_indicator,
            add_staleness=args.staleness,
            strict_hourly=args.strict_hourly,
        )

        result = build_samples(df, cfg)
        report = summarize(result, cfg)

        print(f"\n########## {task.upper()} ##########")
        for name, table in report.items():
            print(f"\n===== {name} =====")
            print(table.to_string(index=False))

        prefix = f"{args.output_prefix}_{task}"
        result["windows"].to_csv(f"{prefix}_windows.csv", index=False)
        result["samples"].to_csv(f"{prefix}_samples.csv", index=False)
        result["exclusions"].to_csv(f"{prefix}_exclusions.csv", index=False)
        report["sample_summary"].to_csv(f"{prefix}_summary.csv", index=False)
        LOGGER.info("files written with prefix '%s'", prefix)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
