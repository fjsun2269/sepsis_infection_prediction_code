#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (C) 2026 Fang-Ju Sun and contributors
"""
Step 1 -- Automatic labelling of sepsis onset in an ICU cohort (Sepsis-3).

Source data model
-----------------
The identifiers below follow the schema of the originating institution's Oracle
clinical data warehouse. Their names and their scopes are institutional
conventions rather than part of any published definition, and a site running
this code on another data source will need to map its own schema onto them. The
48-hour episode rule that governs adm_ICU_id is a separate matter with its own
provenance in the critical care literature, set out further below:

    Pno           patient identifier, stable across all hospital admissions.
    Firstcaseno   hospital-admission identifier. It spans the whole hospital
                  stay, so it is carried by emergency-department records before
                  admission and by general-ward records after ICU discharge. One
                  Firstcaseno may contain several ICU episodes, when a patient
                  moves to a ward and returns to the ICU without leaving
                  hospital.
    Caseno        case number of the individual encounter within the admission.
    adm_ICU_id    ICU-episode identifier. Its scope is the ICU only: records
                  originating outside the ICU do not carry it. An ICU discharge
                  followed by a readmission within 48 hours is treated as the
                  same episode and keeps the same identifier; see "Provenance of
                  the 48-hour rule" below.

What follows from that data model, rather than from Sepsis-3:
  * the 48-hour episode-merging rule and its restriction to a single
    Firstcaseno (`--merge-stay-gap-hours`, `--merge-scope`);
  * linking cultures on Pno + Firstcaseno + time in step 2, because adm_ICU_id
    cannot retrieve emergency-department or general-ward records.

Everything else -- the culture / antibiotic pairing windows, the SOFA search
window and the organ-dysfunction threshold -- follows the published Sepsis-3
criteria and is not site-specific.

Column names are set in the constants block below and can be remapped at run
time with ``--column-map RAW=CANONICAL``, so no source schema is hard-coded
into the logic.

Provenance of the 48-hour rule
------------------------------
Treating a return to the ICU within a short interval as a continuation of the
same stay is standard practice in critical care cohort construction, but the
interval is not standardised and should be reported explicitly rather than
assumed:

  * 48 h is the interval that critical care organisations have proposed for the
    early ICU readmission quality indicator, and is the most widely used
    threshold for distinguishing early from late readmission.
  * MIMIC-III uses 24 h: a return to an ICU bed within 24 hours of transfer to a
    ward is treated as one continuous stay under a single ICUSTAY_ID. MIMIC-IV
    merges only consecutive ICU transfers and deliberately leaves
    non-consecutive stays separate, on the grounds that planned transfers out
    for a procedure cannot be reliably distinguished from unanticipated
    readmissions, and leaves the decision to the investigator.
  * Published reviews note that readmission definitions vary considerably and
    that there is little empirical evidence identifying an optimal interval.

The threshold is therefore exposed as `--merge-stay-gap-hours` (default 48) so
that the choice is visible and can be varied in a sensitivity analysis. In
multicentre data, ICU readmission rates are on the order of 2% within 24 h, 3%
within 48 h and 5% within 72 h of discharge, so the number of stays affected by
moving the threshold is small but non-zero.

References
----------
Singer M, et al. The Third International Consensus Definitions for Sepsis and
    Septic Shock (Sepsis-3). JAMA. 2016;315(8):801-810.
Seymour CW, et al. Assessment of Clinical Criteria for Sepsis. JAMA.
    2016;315(8):762-774.
Kramer AA, Higgins TL, Zimmerman JE. The association between ICU readmission
    rate and patient outcomes. Crit Care Med. 2013;41(1):24-33.
Johnson AEW, et al. MIMIC-III, a freely accessible critical care database. Sci
    Data. 2016;3:160035. (ICU stay continuity defined at 24 hours.)
Johnson AEW, et al. MIMIC-IV, a freely accessible electronic health record
    dataset. Sci Data. 2023;10:1.

Pipeline
--------
Stage 0 -- cohort definition
    Two identifiers with different scopes are used throughout. ``adm_ICU_id``
    is the unique identifier of one ICU admission and is the cohort key; by
    convention an ICU discharge followed by a readmission within 48 hours is the
    same admission and carries the same identifier. ``Firstcaseno`` is the
    hospital-admission identifier: one hospital admission may contain several
    ICU episodes, when a patient moves to a general ward and returns to the ICU
    without leaving hospital. Records from the emergency department and from
    general wards carry Firstcaseno but not adm_ICU_id, which is why step 2
    links cultures on Firstcaseno plus time rather than on adm_ICU_id.

    This stage enforces the 48-hour convention: consecutive ICU stays separated
    by no more than `--merge-stay-gap-hours` (default 48 h; see "Provenance of
    the 48-hour rule" above) are collapsed into a single episode, keeping the
    identifier of the first stay, with the earliest admission and latest
    discharge as its boundaries. `--merge-scope` limits merging to stays sharing
    a Firstcaseno (the default), so that a patient discharged from hospital and
    admitted again within 48 hours starts a new episode rather than continuing
    the previous one. The mapping is written to ``icu_stay_map.csv``. Where the
    source extraction already assigns one identifier per episode -- the expected
    case -- this stage is a no-op and the log reports zero merges; it is
    retained so that the rule is enforced and visible in the published code
    rather than assumed.

Stage 1 -- suspected infection (SI)
    An SI event is a pair of orders placed within a fixed interval:
      * culture first     -> antibiotic within `--culture-to-abx-hours` (72 h)
      * antibiotic first  -> culture within `--abx-to-culture-hours`  (24 h)
    The suspected infection time (`t_suspicion`) is the timestamp of the
    earlier of the two orders.

    These two windows are anchored on a PAIR OF ORDERS and their orientation is
    asymmetric on purpose: an antibiotic started empirically should be followed
    quickly by a specimen (24 h), whereas a specimen sent first may wait longer
    for therapy (72 h). This follows Sepsis-3 (Singer 2016; Seymour 2016) and
    Online Supplemental Figure 2.

    Do not confuse this with the window in step 2. Step 2 uses the same two
    numbers for a different purpose: it opens an interval of -24 h to +72 h
    around the labelled sepsis onset and asks whether any positive culture falls
    inside it. The two rules therefore use different anchors and orientations.
    The exact values used by this implementation are exposed as command-line
    arguments and are written to ``run_config.json``.

Stage 2 -- sepsis onset
    For each SI event the SOFA table is searched over
      [t_suspicion - `--sofa-lookback-hours`, t_suspicion + `--sofa-lookahead-hours`]
    (default -48 h / +24 h; see Online Supplemental Figure 2 of the associated
    publication for the timeline). Sepsis onset is the earliest measurement in
    that window meeting the organ-dysfunction criterion. SI events are evaluated
    in chronological order and the first qualifying event labels the stay, so
    each ICU stay contributes at most one row.

    Organ-dysfunction criterion (`--sofa-mode`), all using `--sofa-threshold`
    (default 2 points):

      total (default)
          total SOFA >= threshold. Sepsis-3 assumes a baseline SOFA of 0 in
          patients without known pre-existing organ dysfunction, so for a cohort
          admitted through the emergency department this is equivalent to
          delta-SOFA >= 2 with a baseline of 0.
      delta-window-min
          SOFA - (minimum SOFA observed anywhere in the search window) >= threshold.
          NOTE: the minimum may occur *after* the candidate time point, so this
          variant looks ahead. Prefer ``delta-running-min`` when the label must
          be causal with respect to time.
      delta-running-min
          SOFA - (minimum SOFA observed in the window up to and including the
          candidate time point) >= threshold. Same intent as delta-window-min
          without look-ahead.
      delta-prewindow
          SOFA - (last SOFA recorded strictly before the window, or 0 if none)
          >= threshold.

    `--onset-time` controls the timestamp written to the output:
      sofa (default) -> the first qualifying SOFA measurement
      suspicion      -> t_suspicion
      earliest       -> min(t_suspicion, first qualifying SOFA time)

Stage 3 -- reference time points for control stays
    Each ICU stay with no sepsis label receives one index time for the non-sepsis
    class. Two reproducible profiles are available:

      legacy (default)
          Recreates the manuscript-aligned implementation, including its
          per-row reseeding behaviour and use of the full ICU-stay duration.

      uniform
          Draws one genuinely uniform index time per stay from the interval
          between ICU admission and ICU discharge minus
          ``--control-margin-hours``. The draw is derived from ``--seed`` and the
          stay identifier, so adding, removing, or reordering other stays does
          not change existing index times.

Reproducibility profiles
------------------------
The manuscript-aligned settings are ``--sofa-mode total`` and
``--control-sampling legacy``. Alternative SOFA definitions and the uniform
control sampler are provided for explicit sensitivity analyses. All resolved
parameters are written to ``run_config.json``; ``--compare-sofa-modes`` writes
``sofa_mode_comparison.csv`` so that changes in case assignment can be audited.

Data sources
------------
Every ``--orders`` / ``--sofa`` / ``--si-codes`` argument accepts either

    a delimited text file      e.g.  data/orders.csv
    a SQL query file           e.g.  sql:sql/orders.sql

In SQL mode the query text is read from the given ``.sql`` file and executed
against the database named by the ``DB_URL`` environment variable, for example

    export DB_URL="<SQLAlchemy database URL>"
    python 01_01_sepsis_onset_auto_label.py --orders sql:sql/orders.sql ...

This keeps hospital-specific schema, table and column names inside local .sql
files and credentials inside the environment, so neither is committed to the
public repository. Column names that differ from the canonical ones below can
be remapped on the command line with ``--column-map RAW=CANONICAL``.

Outputs
-------
<outdir>/suspected_infection.csv  every SI event pair (audit trail)
<outdir>/sepsis_onset.csv         one row per ICU stay, with `label` and
                                  `sepsis_onset` (the index time for both
                                  sepsis cases and controls)
<outdir>/run_config.json          all resolved parameters, for reproducibility

Usage
-----
    python 01_01_sepsis_onset_auto_label.py \
        --si-codes data/si_codes.csv \
        --orders   data/orders.csv \
        --sofa     data/sofa.csv \
        --outdir   output

This file contains no patient-identifiable data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Canonical column names.
#
# These follow the originating institution's Oracle clinical data warehouse and
# are institutional conventions rather than part of the Sepsis-3 definition. To
# run the script against another schema, either edit this block or pass
# --column-map RAW=CANONICAL at run time; the logic below refers only to these
# names, never to a source table or column directly. See "Source data model" in
# the module docstring for the meaning and scope of each identifier.
# --------------------------------------------------------------------------- #

PATIENT_ID = "Pno"          # patient identifier
STAY_ID = "adm_ICU_id"      # ICU-stay identifier, unique within a patient
STAY_KEY = [PATIENT_ID, STAY_ID]

ORDER_TIME = "Scrn"         # order timestamp in the orders table
ORDER_CODE = "Mcode"        # order code in the orders table

SOFA_TIME = "date"          # timestamp of a SOFA measurement
SOFA_SCORE = "SOFA"         # total SOFA score

ADMISSION_ID = "Firstcaseno"  # hospital-admission identifier, spans several ICU episodes
ADMISSION_TIME = "ICU_admdatetime"
DISCHARGE_TIME = "ICU_disdatetime"

# Stay-level columns carried through to the output; missing ones are skipped.
# Firstcaseno is carried because step 2 needs it to scope the culture search.
STAY_KEY_COLUMNS = [
    PATIENT_ID, STAY_ID, ADMISSION_ID, "Caseno", "Bedns", "Bedno",
    ADMISSION_TIME, DISCHARGE_TIME,
]

SOFA_COMPONENTS = [
    "respiration_24hours",
    "coagulation_24hours",
    "liver_24hours",
    "cardiovascular_24hours",
    "cns_24hours",
    "renal_24hours",
]

SOFA_MODES = ("total", "delta-window-min", "delta-running-min", "delta-prewindow")

LOGGER = logging.getLogger("sepsis_onset")


# --------------------------------------------------------------------------- #
# Data-source layer -- file or SQL, so that Oracle object names stay external
# --------------------------------------------------------------------------- #

def load_table(spec: str, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Load a table from a delimited file or from a SQL query file.

    `spec` is either a path to a .csv/.tsv/.txt file, or ``sql:<path>.sql``.
    Everything is read as ``str``: identifiers must never be coerced to numeric
    types, because that turns ``"0012"`` into ``12`` and ``"12"`` into ``12.0``
    when nulls are present, silently breaking the joins between tables.
    Timestamps are converted afterwards by :func:`to_datetime`, since passing
    ``parse_dates`` together with ``dtype=str`` to ``read_csv`` is undefined.
    """
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
    return frame


def _load_from_file(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"input file not found: {path}")
    separator = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    return pd.read_csv(path, sep=separator, dtype=str, encoding="utf-8-sig")


def _load_from_sql(query_path: Path) -> pd.DataFrame:
    """Execute the query stored in `query_path` against ``$DB_URL``."""
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
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ValueError("SQL source requires SQLAlchemy (pip install sqlalchemy oracledb)") from exc

    query = query_path.read_text(encoding="utf-8")
    LOGGER.info("executing %s against the configured database", query_path)
    engine = create_engine(db_url)
    with engine.connect() as connection:
        frame = pd.read_sql(text(query), connection)
    # Normalise to strings so that the file and SQL paths behave identically.
    # Oracle returns identifiers as NUMBER, which would otherwise arrive as floats.
    return frame.astype("object").where(frame.notna(), None).astype("string")


def to_datetime(df: pd.DataFrame, columns: list[str], required: bool = True) -> pd.DataFrame:
    """Convert the given columns to datetime64, coercing unparsable values."""
    for col in columns:
        if col not in df.columns:
            if required:
                raise KeyError(f"expected datetime column '{col}' is missing")
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        n_bad = int(parsed.isna().sum() - df[col].isna().sum())
        if n_bad > 0:
            LOGGER.warning("%s: %d value(s) could not be parsed as a datetime", col, n_bad)
        df[col] = parsed
    return df


def normalise_id(series: pd.Series) -> pd.Series:
    """Strip whitespace and a trailing ``.0`` left by a float round-trip.

    Applied to every identifier in every table so that the join keys always
    agree, whichever source the table came from.
    """
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"{source} is missing required column(s): {missing}. "
            "Use --column-map RAW=CANONICAL to remap them."
        )


def available(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return the subset of `columns` present in `df`, warning about the rest."""
    present = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        LOGGER.warning("column(s) not found and therefore skipped: %s", missing)
    return present


# --------------------------------------------------------------------------- #
# Stage 1 -- suspected infection
# --------------------------------------------------------------------------- #

def load_si_codes(spec: str, column_map: dict[str, str]) -> tuple[set[str], set[str]]:
    """Load the order codes defining cultures and antibiotics.

    Codes listed in both categories are reported and removed, because such a
    code makes the culture / antibiotic ordering of an SI pair ambiguous.
    """
    table = load_table(spec, column_map)
    require_columns(table, ["culture", "antibiotics"], spec)

    culture = set(table["culture"].dropna().str.strip()) - {""}
    antibiotics = set(table["antibiotics"].dropna().str.strip()) - {""}

    overlap = culture & antibiotics
    if overlap:
        LOGGER.warning(
            "%d order code(s) listed as both culture and antibiotic were removed "
            "from both sets: %s",
            len(overlap), sorted(overlap)[:10],
        )
        culture -= overlap
        antibiotics -= overlap

    if not culture or not antibiotics:
        raise ValueError("both the culture and the antibiotic code list must be non-empty")

    LOGGER.info("SI reference codes: %d culture, %d antibiotic", len(culture), len(antibiotics))
    return culture, antibiotics


def tag_orders(orders: pd.DataFrame, culture: set[str], antibiotics: set[str]) -> pd.DataFrame:
    """Add an ``si_type`` column ('C', 'A' or NA) to the orders table.

    Set membership is used rather than list membership: scanning a list once per
    order row is O(n_orders * n_codes).
    """
    code = orders[ORDER_CODE].astype("string").str.strip()
    si_type = pd.Series(pd.NA, index=orders.index, dtype="string")
    si_type[code.isin(culture)] = "C"
    si_type[code.isin(antibiotics)] = "A"
    orders = orders.assign(si_type=si_type)
    LOGGER.info("orders tagged: %s", si_type.value_counts(dropna=True).to_dict())
    return orders


SI_EVENT_COLUMNS = [
    "t_suspicion", "code_start", "start_type", "t_end", "code_end", "end_type",
]


def _pair_events(
    first: pd.DataFrame,
    second: pd.DataFrame,
    first_type: str,
    second_type: str,
    window_hours: float,
) -> pd.DataFrame:
    """Match each order in `first` with the earliest order in `second` at or
    after it and within `window_hours`.

    Uses :func:`numpy.searchsorted`, i.e. O(n log n) per ICU stay, instead of
    re-slicing the stay's DataFrame once per row, which is O(n^2).
    """
    if first.empty or second.empty:
        return pd.DataFrame(columns=SI_EVENT_COLUMNS)

    t_first = first[ORDER_TIME].to_numpy(dtype="datetime64[ns]")
    code_first = first[ORDER_CODE].to_numpy()
    t_second = second[ORDER_TIME].to_numpy(dtype="datetime64[ns]")
    code_second = second[ORDER_CODE].to_numpy()

    order = np.argsort(t_second, kind="stable")
    t_second, code_second = t_second[order], code_second[order]

    idx = np.searchsorted(t_second, t_first, side="left")  # first t_second >= t_first
    in_range = idx < len(t_second)

    matched = np.zeros(len(t_first), dtype=bool)
    limit = t_first + np.timedelta64(int(round(window_hours * 3600)), "s")
    matched[in_range] = t_second[idx[in_range]] <= limit[in_range]
    if not matched.any():
        return pd.DataFrame(columns=SI_EVENT_COLUMNS)

    hit = idx[matched]
    return pd.DataFrame(
        {
            "t_suspicion": t_first[matched],   # the earlier order of the pair
            "code_start": code_first[matched],
            "start_type": first_type,
            "t_end": t_second[hit],
            "code_end": code_second[hit],
            "end_type": second_type,
        }
    )


def extract_suspected_infection(
    orders: pd.DataFrame,
    culture_to_abx_hours: float,
    abx_to_culture_hours: float,
) -> pd.DataFrame:
    """Return every SI event pair, one row per pair, sorted chronologically."""
    tagged = orders.dropna(subset=["si_type", ORDER_TIME])
    events: list[pd.DataFrame] = []

    for (pno, stay), stay_orders in tagged.groupby(STAY_KEY, sort=False):
        cultures = stay_orders[stay_orders["si_type"] == "C"]
        antibiotics = stay_orders[stay_orders["si_type"] == "A"]
        pairs = pd.concat(
            [
                _pair_events(cultures, antibiotics, "C", "A", culture_to_abx_hours),
                _pair_events(antibiotics, cultures, "A", "C", abx_to_culture_hours),
            ],
            ignore_index=True,
        )
        if pairs.empty:
            continue
        pairs[PATIENT_ID], pairs[STAY_ID] = pno, stay
        events.append(pairs)

    if not events:
        LOGGER.warning("no suspected-infection event was found")
        return pd.DataFrame(columns=STAY_KEY + SI_EVENT_COLUMNS)

    si = pd.concat(events, ignore_index=True)[STAY_KEY + SI_EVENT_COLUMNS]
    si = si.sort_values(STAY_KEY + ["t_suspicion"]).reset_index(drop=True)
    LOGGER.info(
        "suspected infection: %d event(s) in %d ICU stay(s)",
        len(si), si.drop_duplicates(STAY_KEY).shape[0],
    )
    return si


# --------------------------------------------------------------------------- #
# Stage 2 -- sepsis onset
# --------------------------------------------------------------------------- #

def _baseline_vector(
    scores: np.ndarray,
    prewindow_score: float,
    sofa_mode: str,
) -> np.ndarray:
    """Return the per-measurement baseline implied by `sofa_mode`."""
    if sofa_mode == "total":
        # Sepsis-3 assumes a baseline of 0 in patients without known chronic
        # organ dysfunction, so the absolute score is already the increase.
        return np.zeros_like(scores)
    if sofa_mode == "delta-window-min":
        return np.full_like(scores, scores.min())
    if sofa_mode == "delta-running-min":
        return np.minimum.accumulate(scores)
    if sofa_mode == "delta-prewindow":
        return np.full_like(scores, prewindow_score)
    raise ValueError(f"unknown --sofa-mode: {sofa_mode}")


def assign_sepsis_onset(
    si_events: pd.DataFrame,
    sofa: pd.DataFrame,
    lookback_hours: float,
    lookahead_hours: float,
    threshold: float,
    sofa_mode: str,
    onset_time: str,
) -> pd.DataFrame:
    """Label ICU stays whose SI event is accompanied by organ dysfunction.

    Returns one row per labelled ICU stay: the first SI event meeting the
    criterion wins and the remaining events for that stay are skipped.
    """
    lookback = pd.Timedelta(hours=lookback_hours)
    lookahead = pd.Timedelta(hours=lookahead_hours)

    # Group the SOFA table once. Filtering the full table inside an `apply`
    # would be O(n_events * n_sofa_rows).
    sofa_by_stay = {key: group for key, group in sofa.groupby(STAY_KEY, sort=False)}
    stay_columns = available(sofa, STAY_KEY_COLUMNS)
    component_columns = available(sofa, SOFA_COMPONENTS)

    records: list[dict] = []
    for key, events in si_events.groupby(STAY_KEY, sort=False):
        stay_sofa = sofa_by_stay.get(key)
        if stay_sofa is None or stay_sofa.empty:
            continue

        for event in events.sort_values("t_suspicion").itertuples(index=False):
            window_start = event.t_suspicion - lookback
            window_end = event.t_suspicion + lookahead
            window = stay_sofa[
                stay_sofa[SOFA_TIME].between(window_start, window_end, inclusive="both")
            ]
            if window.empty:
                continue

            before = stay_sofa[stay_sofa[SOFA_TIME] < window_start]
            prewindow = float(before[SOFA_SCORE].iloc[-1]) if not before.empty else 0.0

            scores = window[SOFA_SCORE].to_numpy(dtype=float)
            baseline = _baseline_vector(scores, prewindow, sofa_mode)
            qualifies = (scores - baseline) >= threshold
            if not qualifies.any():
                continue

            position = int(np.argmax(qualifies))  # earliest qualifying measurement
            hit = window.iloc[position]
            t_sofa = hit[SOFA_TIME]

            if onset_time == "suspicion":
                onset = event.t_suspicion
            elif onset_time == "earliest":
                onset = min(event.t_suspicion, t_sofa)
            else:  # "sofa"
                onset = t_sofa

            record = {col: hit[col] for col in stay_columns + component_columns}
            record.update(
                {
                    "sepsis_onset": onset,
                    "t_suspicion": event.t_suspicion,
                    "t_sofa_threshold": t_sofa,
                    SOFA_SCORE: hit[SOFA_SCORE],
                    "sofa_baseline": float(baseline[position]),
                    "sofa_delta": float(scores[position] - baseline[position]),
                    "start_type": event.start_type,
                    "code_start": event.code_start,
                    "end_type": event.end_type,
                    "code_end": event.code_end,
                    "label": 1,
                }
            )
            records.append(record)
            break  # one label per ICU stay

    if not records:
        LOGGER.warning("no ICU stay met the sepsis criteria")
        return pd.DataFrame(columns=stay_columns + ["sepsis_onset", "label", SOFA_SCORE])

    cases = pd.DataFrame.from_records(records).sort_values(STAY_KEY).reset_index(drop=True)
    LOGGER.info("sepsis cases labelled: %d ICU stay(s) [--sofa-mode %s]", len(cases), sofa_mode)
    return cases


def compare_sofa_modes(
    si_events: pd.DataFrame,
    sofa: pd.DataFrame,
    lookback_hours: float,
    lookahead_hours: float,
    threshold: float,
    onset_time: str,
) -> pd.DataFrame:
    """Label the cohort under every organ-dysfunction definition and compare.

    The function reports how many ICU stays are labelled under each supported
    SOFA baseline definition and how much the assignments overlap. It is intended
    for sensitivity analysis and for auditing the effect of changing the
    organ-dysfunction definition. ``delta-prewindow`` uses the last score before
    the search window and falls back to 0 when no earlier score exists.
    """
    rows: list[dict] = []
    labelled: dict[str, set] = {}

    for mode in SOFA_MODES:
        cases = assign_sepsis_onset(
            si_events, sofa,
            lookback_hours=lookback_hours,
            lookahead_hours=lookahead_hours,
            threshold=threshold,
            sofa_mode=mode,
            onset_time=onset_time,
        )
        labelled[mode] = (
            set(map(tuple, cases[STAY_KEY].to_numpy())) if not cases.empty else set()
        )

    reference = labelled["total"]
    for mode in SOFA_MODES:
        keys = labelled[mode]
        rows.append(
            {
                "sofa_mode": mode,
                "definition": {
                    "total": "absolute SOFA >= threshold (published analysis; "
                             "Sepsis-3 baseline of 0)",
                    "delta-window-min": "increase over the window minimum "
                                        "(wording of Supplemental Figure 2)",
                    "delta-running-min": "increase over the running minimum, no look-ahead",
                    "delta-prewindow": "increase over the last score before the window "
                                       "(pre-ICU baseline), 0 if none",
                }[mode],
                "n_sepsis_stays": len(keys),
                "n_shared_with_total": len(keys & reference),
                "n_only_in_this_mode": len(keys - reference),
                "n_only_in_total": len(reference - keys),
            }
        )

    comparison = pd.DataFrame(rows)
    LOGGER.info("SOFA definition comparison:\n%s", comparison.to_string(index=False))
    return comparison


# --------------------------------------------------------------------------- #
# Stage 3 -- reference time points for control stays
# --------------------------------------------------------------------------- #

def merge_icu_stays(sofa: pd.DataFrame, gap_hours: float, scope: str) -> pd.DataFrame:
    """Collapse consecutive ICU stays of one patient into a single episode.

    A discharge followed by a readmission within `gap_hours` (default 48 h) is
    treated as one continuous ICU stay, so that a patient who leaves the ICU and
    returns the same day is not split into two cohort entries with two
    independent labels. Overlapping stays (a negative gap) are always merged.

    `scope` limits which stays may be merged with each other:

        admission (default)
            only stays sharing a Firstcaseno, i.e. belonging to the same
            hospital admission. A patient discharged from hospital and admitted
            again within 48 hours starts a new admission and therefore a new
            episode, which is the intended reading of the rule.
        patient
            any two stays of the same patient. Fallback when the extraction has
            no Firstcaseno.

    The episode inherits the identifier of its first stay, so the returned key
    remains traceable to the source record. Episode-level admission and
    discharge times are the earliest admission and the latest discharge of the
    constituent stays.

    Returns a mapping table with one row per source stay:
        Pno, adm_ICU_id_source, adm_ICU_id, ICU_admdatetime, ICU_disdatetime

    The operation is idempotent: if the upstream extraction already merged
    stays, no further merging occurs and the log reports zero merges.
    """
    columns = [PATIENT_ID, STAY_ID, ADMISSION_TIME, DISCHARGE_TIME]
    missing = [c for c in columns if c not in sofa.columns]
    if missing:
        LOGGER.warning("ICU stay merging skipped, missing column(s): %s", missing)
        return pd.DataFrame(columns=columns)

    if scope == "admission" and ADMISSION_ID not in sofa.columns:
        raise KeyError(
            f"--merge-scope admission requires the SOFA table to contain '{ADMISSION_ID}'. "
            "Use --column-map to supply it, or --merge-scope patient."
        )
    group_keys = [PATIENT_ID, ADMISSION_ID] if scope == "admission" else [PATIENT_ID]

    stays = sofa[list(dict.fromkeys(columns + group_keys))]
    stays = stays.drop_duplicates(subset=STAY_KEY, keep="first")
    stays = stays.dropna(subset=[ADMISSION_TIME, DISCHARGE_TIME]).copy()
    if stays.empty:
        return pd.DataFrame(columns=columns)

    if scope == "admission":
        # A stay without a Firstcaseno cannot be grouped with any other; give it
        # its own key rather than dropping it from the cohort.
        n_unkeyed = int(stays[ADMISSION_ID].isna().sum())
        if n_unkeyed:
            LOGGER.warning("%d ICU stay(s) have no %s and were treated as separate admissions",
                           n_unkeyed, ADMISSION_ID)
        stays[ADMISSION_ID] = stays[ADMISSION_ID].fillna("__" + stays[STAY_ID].astype(str))

    stays = stays.sort_values(group_keys + [ADMISSION_TIME]).reset_index(drop=True)
    grouped = stays.groupby(group_keys, sort=False)

    # Running latest discharge so far, so that a stay nested inside an earlier
    # one does not start a new episode.
    latest_discharge = grouped[DISCHARGE_TIME].cummax()
    previous_end = latest_discharge.groupby([stays[k] for k in group_keys]).shift(1)
    gap = (stays[ADMISSION_TIME] - previous_end).dt.total_seconds() / 3600.0
    starts_episode = previous_end.isna() | (gap > gap_hours)

    stays["_episode"] = starts_episode.groupby([stays[k] for k in group_keys]).cumsum()
    episode = stays.groupby(group_keys + ["_episode"], sort=False)
    stays["adm_ICU_id_source"] = stays[STAY_ID]
    stays[STAY_ID] = episode[STAY_ID].transform("first")
    stays[ADMISSION_TIME] = episode[ADMISSION_TIME].transform("min")
    stays[DISCHARGE_TIME] = episode[DISCHARGE_TIME].transform("max")

    n_episodes = int(stays[STAY_KEY].drop_duplicates().shape[0])
    LOGGER.info(
        "ICU stay merging (gap <= %.0f h, scope=%s): %d source stay(s) -> %d episode(s), "
        "%d merged away",
        gap_hours, scope, len(stays), n_episodes, len(stays) - n_episodes,
    )
    return stays[[PATIENT_ID, "adm_ICU_id_source", STAY_ID, ADMISSION_TIME, DISCHARGE_TIME]]


def apply_stay_map(df: pd.DataFrame, stay_map: pd.DataFrame, source: str) -> pd.DataFrame:
    """Replace the stay identifier of `df` with the merged-episode identifier.

    Rows whose stay is absent from the mapping keep their original identifier
    and are reported, because silently dropping them would shrink the cohort.
    """
    if stay_map.empty:
        return df

    mapping = stay_map.rename(columns={"adm_ICU_id_source": STAY_ID, STAY_ID: "_episode_id"})
    keep = [PATIENT_ID, STAY_ID, "_episode_id", ADMISSION_TIME, DISCHARGE_TIME]
    merged = df.merge(mapping[keep], on=STAY_KEY, how="left", suffixes=("", "_episode"))

    unmapped = merged["_episode_id"].isna()
    if int(unmapped.sum()):
        LOGGER.warning("%s: %d row(s) had no ICU stay in the mapping and were left unchanged",
                       source, int(unmapped.sum()))

    merged["adm_ICU_id_source"] = merged[STAY_ID]
    merged[STAY_ID] = merged["_episode_id"].fillna(merged[STAY_ID])
    # Adopt the episode-level boundaries where the table carries them.
    for col in (ADMISSION_TIME, DISCHARGE_TIME):
        episode_col = f"{col}_episode"
        if col in merged.columns and episode_col in merged.columns:
            merged[col] = merged[episode_col].fillna(merged[col])
            merged = merged.drop(columns=[episode_col])
        elif episode_col in merged.columns:
            merged = merged.rename(columns={episode_col: col})
    return merged.drop(columns=["_episode_id"])


def build_stay_table(sofa: pd.DataFrame) -> pd.DataFrame:
    """One row per ICU stay, taken from the SOFA table (the cohort definition)."""
    stays = sofa[available(sofa, STAY_KEY_COLUMNS)].drop_duplicates(subset=STAY_KEY, keep="first")
    return stays.sort_values(STAY_KEY).reset_index(drop=True)


def sample_control_times_legacy(controls: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the index times used in the published analysis.

    Retained for reproducibility only; ``--control-sampling uniform`` is the
    recommended setting for any new analysis. The original implementation called
    ``random.seed(0)`` *inside* the per-row function, so the pseudo-random stream
    was reset before every draw and the offset became a deterministic function of
    the stay length rather than a uniform random draw. The whole ICU stay was
    used as the sampling range, with no margin before discharge.

    The behaviour is replicated exactly, except that zero-length stays are
    skipped with a warning instead of raising ``ValueError``.
    """
    import random  # local import: only the legacy path uses the stdlib generator

    controls = controls.copy()
    offsets: list[float] = []
    keep: list[bool] = []
    for admission, discharge in zip(controls[ADMISSION_TIME], controls[DISCHARGE_TIME]):
        delta = discharge - admission
        total_seconds = delta.days * 24 * 60 * 60 + delta.seconds
        if total_seconds <= 0:
            offsets.append(0.0)
            keep.append(False)
            continue
        random.seed(0)  # re-seeded per row, as in the published implementation
        offsets.append(random.randrange(total_seconds))
        keep.append(True)

    n_dropped = len(keep) - sum(keep)
    if n_dropped:
        LOGGER.warning("%d control stay(s) with a non-positive length were dropped", n_dropped)

    controls["sepsis_onset"] = controls[ADMISSION_TIME] + pd.to_timedelta(offsets, unit="s")
    controls["label"] = 0
    LOGGER.warning(
        "legacy control sampling in use: index times reproduce the published cohort but are "
        "not uniform random draws"
    )
    return controls[pd.Series(keep, index=controls.index)]


def _stay_seed(seed: int, patient: str, stay: str) -> int:
    """Derive a per-stay seed from the run seed and the stay identifier.

    Seeding each draw from the identifier rather than from position in the table
    means the index time of a control stay depends only on `seed` and that
    stay's own key. Adding, removing or reordering other patients therefore
    leaves every remaining index time unchanged, which a single sequential
    generator would not guarantee.
    """
    digest = hashlib.blake2b(f"{seed}|{patient}|{stay}".encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "big")


def sample_control_times(controls: pd.DataFrame, seed: int, margin_hours: float) -> pd.DataFrame:
    """Draw exactly one reference (index) time per control stay.

    One draw per control stay, uniform over
    [ICU admission, ICU discharge - `margin_hours`], reproducible for a given
    `--seed`. Two defects of the original implementation are fixed here:

      * ``random.seed`` was called inside the per-row function, so the generator
        was reset before every draw and the offset became a deterministic
        function of the stay length rather than a uniform draw;
      * a zero-length sampling window raised ``ValueError``.

    The margin keeps the index time far enough from discharge for a prediction
    horizon to exist after it; shorter stays are dropped and reported.
    """
    controls = controls.copy()
    admission = controls[ADMISSION_TIME]
    latest = controls[DISCHARGE_TIME] - pd.Timedelta(hours=margin_hours)

    span_seconds = (latest - admission).dt.total_seconds()
    usable = span_seconds >= 1  # at least a one-second sampling window
    if int((~usable).sum()):
        LOGGER.warning(
            "%d control stay(s) shorter than the %.1f h margin were dropped",
            int((~usable).sum()), margin_hours,
        )
    controls, span_seconds = controls[usable].copy(), span_seconds[usable]

    offsets = [
        int(np.random.default_rng(_stay_seed(seed, patient, stay)).integers(0, int(span)))
        for patient, stay, span in zip(controls[PATIENT_ID], controls[STAY_ID], span_seconds)
    ]
    controls["sepsis_onset"] = controls[ADMISSION_TIME] + pd.to_timedelta(offsets, unit="s")
    controls["label"] = 0
    LOGGER.info("control index times: %d stay(s), one draw each, seed=%d", len(controls), seed)
    return controls


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

    # ---- load -------------------------------------------------------------- #
    culture_codes, abx_codes = load_si_codes(args.si_codes, column_map)

    orders = load_table(args.orders, column_map)
    require_columns(orders, STAY_KEY + [ORDER_TIME, ORDER_CODE], args.orders)
    orders = to_datetime(orders, [ORDER_TIME])
    for col in available(orders, STAY_KEY + [ADMISSION_ID]):
        orders[col] = normalise_id(orders[col])
    LOGGER.info("orders loaded: %d row(s)", len(orders))

    sofa = load_table(args.sofa, column_map)
    require_columns(sofa, STAY_KEY + [SOFA_TIME, SOFA_SCORE], args.sofa)
    sofa = to_datetime(sofa, [SOFA_TIME])
    sofa = to_datetime(sofa, [ADMISSION_TIME, DISCHARGE_TIME], required=False)
    for col in available(sofa, STAY_KEY + [ADMISSION_ID, "Caseno"]):
        sofa[col] = normalise_id(sofa[col])
    for col in [SOFA_SCORE] + available(sofa, SOFA_COMPONENTS):
        sofa[col] = pd.to_numeric(sofa[col], errors="coerce")
    sofa = sofa.dropna(subset=[SOFA_TIME, SOFA_SCORE])
    sofa = sofa.sort_values(STAY_KEY + [SOFA_TIME]).reset_index(drop=True)
    LOGGER.info("SOFA records loaded: %d row(s) in %d ICU stay(s)",
                len(sofa), sofa.drop_duplicates(STAY_KEY).shape[0])

    # ---- merge ICU readmissions into single episodes ----------------------- #
    stay_map = merge_icu_stays(sofa, args.merge_stay_gap_hours, args.merge_scope)
    if not stay_map.empty:
        stay_map.to_csv(outdir / "icu_stay_map.csv", index=False, encoding="utf-8-sig")
        sofa = apply_stay_map(sofa, stay_map, "SOFA")
        orders = apply_stay_map(orders, stay_map, "orders")
        sofa = sofa.sort_values(STAY_KEY + [SOFA_TIME]).reset_index(drop=True)

    # ---- stage 1 ----------------------------------------------------------- #
    orders = tag_orders(orders, culture_codes, abx_codes)
    si_events = extract_suspected_infection(
        orders,
        culture_to_abx_hours=args.culture_to_abx_hours,
        abx_to_culture_hours=args.abx_to_culture_hours,
    )
    si_events.to_csv(outdir / "suspected_infection.csv", index=False, encoding="utf-8-sig")

    # ---- stage 2 ----------------------------------------------------------- #
    cases = assign_sepsis_onset(
        si_events, sofa,
        lookback_hours=args.sofa_lookback_hours,
        lookahead_hours=args.sofa_lookahead_hours,
        threshold=args.sofa_threshold,
        sofa_mode=args.sofa_mode,
        onset_time=args.onset_time,
    )

    if args.compare_sofa_modes:
        comparison = compare_sofa_modes(
            si_events, sofa,
            lookback_hours=args.sofa_lookback_hours,
            lookahead_hours=args.sofa_lookahead_hours,
            threshold=args.sofa_threshold,
            onset_time=args.onset_time,
        )
        comparison.to_csv(outdir / "sofa_mode_comparison.csv",
                          index=False, encoding="utf-8-sig")

    # ---- stage 3 ----------------------------------------------------------- #
    stays = build_stay_table(sofa)
    labelled_keys = set(map(tuple, cases[STAY_KEY].to_numpy())) if not cases.empty else set()
    is_case = pd.Series([tuple(k) in labelled_keys for k in stays[STAY_KEY].to_numpy()],
                        index=stays.index)
    if args.control_sampling == "legacy":
        controls = sample_control_times_legacy(stays[~is_case])
    else:
        controls = sample_control_times(stays[~is_case], args.seed, args.control_margin_hours)

    cohort = pd.concat([cases, controls], ignore_index=True)
    cohort["sepsis_onset"] = cohort["sepsis_onset"].dt.floor("h")

    # Control index times are reproducible for a given --seed, but they are not
    # subject to the 16-hour rule that step 3 applies to labelled onsets. A
    # control index time closer than 16 h to ICU admission leaves no room for the
    # 8-hour feature window plus the 8-hour lead time, so step 5 will not be able
    # to build a sample for that stay. Count them here so the loss is visible.
    if ADMISSION_TIME in cohort.columns:
        hours_after_admission = (
            (cohort["sepsis_onset"] - cohort[ADMISSION_TIME]).dt.total_seconds() / 3600.0
        )
        too_early = (cohort["label"] == 0) & (hours_after_admission < args.min_index_hours)
        if int(too_early.sum()):
            LOGGER.warning(
                "%d control index time(s) fall less than %.0f h after ICU admission and cannot "
                "support an 8-hour feature window plus an 8-hour lead time in step 5",
                int(too_early.sum()), args.min_index_hours,
            )
        cohort["index_time_within_min_hours"] = too_early.astype(int)

    ordered = available(cohort, STAY_KEY_COLUMNS) + [
        "sepsis_onset", "label", SOFA_SCORE, "sofa_baseline", "sofa_delta",
        "t_suspicion", "t_sofa_threshold", "start_type", "code_start",
        "end_type", "code_end",
    ]
    cohort = cohort[[c for c in ordered if c in cohort.columns]]
    cohort = cohort.sort_values(STAY_KEY).reset_index(drop=True)

    out_path = outdir / "sepsis_onset.csv"
    cohort.to_csv(out_path, index=False, encoding="utf-8-sig")
    (outdir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    n_case = int((cohort["label"] == 1).sum())
    LOGGER.info(
        "cohort written to %s: %d stay(s), %d sepsis (%.1f%%), %d control",
        out_path, len(cohort), n_case, 100.0 * n_case / max(len(cohort), 1), len(cohort) - n_case,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automatic labelling of sepsis onset (Sepsis-3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--si-codes", required=True,
                        help="culture / antibiotic order codes: file path or sql:<file>.sql")
    parser.add_argument("--orders", required=True,
                        help="ICU orders: file path or sql:<file>.sql")
    parser.add_argument("--sofa", required=True,
                        help="SOFA scores: file path or sql:<file>.sql")
    parser.add_argument("--outdir", default="output", help="output directory")
    parser.add_argument("--column-map", nargs="*", default=[], metavar="RAW=CANONICAL",
                        help="rename source columns, e.g. PATIENT_NO=Pno ORDER_DTM=Scrn")

    parser.add_argument("--culture-to-abx-hours", type=float, default=72.0,
                        help="antibiotic must follow the culture within this window")
    parser.add_argument("--abx-to-culture-hours", type=float, default=24.0,
                        help="culture must follow the antibiotic within this window")

    parser.add_argument("--sofa-lookback-hours", type=float, default=48.0,
                        help="SOFA search window before the suspected-infection time")
    parser.add_argument("--sofa-lookahead-hours", type=float, default=24.0,
                        help="SOFA search window after the suspected-infection time")
    parser.add_argument("--sofa-threshold", type=float, default=2.0,
                        help="organ-dysfunction threshold, in SOFA points")
    parser.add_argument("--sofa-mode", choices=list(SOFA_MODES), default="total",
                        help="baseline used for the SOFA increase; see the module docstring")
    parser.add_argument("--onset-time", choices=["sofa", "suspicion", "earliest"], default="sofa",
                        help="definition of the sepsis onset timestamp")

    parser.add_argument("--compare-sofa-modes", action="store_true",
                        help="also label the cohort under every --sofa-mode and write "
                             "sofa_mode_comparison.csv for sensitivity analysis")
    parser.add_argument("--merge-stay-gap-hours", type=float, default=48.0,
                        help="a readmission within this many hours of the previous ICU discharge "
                             "is treated as the same ICU stay; 0 disables merging")
    parser.add_argument("--merge-scope", choices=["admission", "patient"], default="admission",
                        help="'admission' merges only ICU stays sharing a Firstcaseno, i.e. "
                             "within one hospital admission; 'patient' ignores that boundary")
    parser.add_argument("--control-sampling", choices=["legacy", "uniform"], default="legacy",
                        help="'legacy' reproduces the manuscript-aligned index times; "
                             "'uniform' is available for sensitivity analyses and new studies")
    parser.add_argument("--min-index-hours", type=float, default=16.0,
                        help="audit threshold only: control index times closer than this to ICU "
                             "admission are counted and flagged, matching the 16-hour rule that "
                             "step 3 applies to labelled onsets. No stay is removed here")
    parser.add_argument("--control-margin-hours", type=float, default=6.0,
                        help="exclude this much time before discharge when sampling controls "
                             "(ignored by --control-sampling legacy)")
    parser.add_argument("--seed", type=int, default=0,
                        help="fixed random seed; the control index times are fully reproducible "
                             "for a given seed and input")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
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
    except (FileNotFoundError, KeyError, ValueError) as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
