#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (C) 2026 Fang-Ju Sun and contributors
"""
Step 2 -- Automatic labelling of culture-confirmed infection.

This script consumes the cohort produced by ``01_01_sepsis_onset_auto_label.py``
and searches, for every ICU stay, the microbiological culture reports ordered
inside the infection-confirmation window anchored on the sepsis onset time:

    [ sepsis_onset - `--window-before-hours`,
      sepsis_onset + `--window-after-hours` ]        (default -24 h / +72 h)

Infection onset is then the timestamp of the earliest positive culture inside
that window. A culture result is not a point in time in the way a vital sign is:
the specimen is taken at one moment, the organism grows over the following days,
and the report is finalised later still. So the label cannot be built forwards
from a culture. It is built backwards from the sepsis onset -- take the onset,
open a window around it, and ask whether any positive culture falls inside.


Two distinct 24 h / 72 h windows -- read this before comparing with the paper
----------------------------------------------------------------------------
The numbers 24 and 72 appear twice in this pipeline, in two different steps,
anchored on two different events, and in opposite orientations. Conflating them
is the single easiest way to misread the code against the manuscript, so both
are set out here in full.

STEP 1 (`01_01_sepsis_onset_auto_label.py`) -- suspected infection time.
    Anchored on a *pair of orders*. Which order comes first decides the window:

        culture first    -> antibiotic must follow within 72 h
        antibiotic first -> culture must follow within 24 h

    This is the Sepsis-3 operationalisation (Singer 2016; Seymour 2016) and is
    the orientation implemented in step 1 and drawn in Online Supplemental
    Figure 2. The clinical reading is asymmetric on purpose: an antibiotic
    started empirically should be followed quickly by a specimen, whereas a
    specimen sent first may wait longer for therapy.

STEP 2 (this script) -- culture-confirmed infection.
    Anchored on the *sepsis onset*, which step 1 has already labelled. The
    window is symmetric in intent and covers the interval over which a specimen
    would plausibly belong to the same episode:

        sepsis_onset - 24 h   ...   sepsis_onset + 72 h

    The 72 h side is the longer one because culture turnaround is 3-5 days:
    a specimen taken at or shortly after onset is reported days later, and the
    order time of that specimen still lies inside the window.

Timeline, with the worked example the study team uses:

    sepsis onset T = 2022-03-20 00:00

        2022-03-19 00:00              T              2022-03-23 00:00
              |---------------------- | ---------------------------|
                     -24 h         onset                 +72 h
              <------------ infection-confirmation window --------->

    Every positive culture whose order time falls in that interval belongs to
    this episode. If the earliest of them was ordered on 2022-03-20 06:00, then

        infection      = 1
        infection_onset = 2022-03-20 06:00

    A positive culture ordered on 2022-03-24 is outside the window and does not
    label this episode.

For reproducibility, the order-pairing and infection-confirmation windows are
kept as separate command-line parameters and are written to ``run_config.json``.
Users adapting the pipeline should verify that both sets of windows match their
protocol before running downstream analyses.

Source data model
-----------------
The identifiers used below follow the schema of the originating institution's
Oracle clinical data warehouse. They are institutional conventions, not part of
the Sepsis-3 definition, and a site running this code on another data source
will need to map its own schema onto them:

    Pno           patient identifier, stable across all hospital admissions.
    Firstcaseno   hospital-admission identifier, spanning the whole hospital
                  stay and therefore carried by emergency-department records
                  before admission and general-ward records after ICU discharge.
                  One Firstcaseno may contain several ICU episodes.
    adm_ICU_id    ICU-episode identifier, scoped to the ICU only.

Two identifiers matter here and they have different scopes. ``adm_ICU_id``
designates one ICU admission; it is not carried by records originating in the
emergency department before admission or on a general ward after discharge --
which is where a large part of this cohort's confirming specimens are obtained.
``Firstcaseno`` is carried by all of them, but because a single hospital
admission may contain several ICU episodes it does not identify an episode on
its own.

Reports are therefore matched on ``Pno`` + ``Firstcaseno`` + the infection
window (``--join-on admission``, the default): the two identifiers scope the
search to the right hospitalisation, and the time window selects the ICU
episode. ``--join-on patient`` reproduces the original patient-only match, and
``--join-on stay`` is available for extractions confined to the ICU; the latter
discards every pre-admission and post-discharge specimen.

The window lengths themselves follow Sepsis-3 and are not site-specific; the
choice of join key is a consequence of the data model above. Column names are
set in the constants block below and can be remapped at run time with
``--column-map RAW=CANONICAL``, so no source schema is hard-coded into the
logic.

Cultures are deliberately NOT restricted to the ICU stay.
``--restrict-to-icu-stay`` is available for cohorts where every specimen is
expected to originate inside the ICU, but it is off by default.

A specimen can still fall inside the windows of two ICU episodes of the same
hospital admission. Step 1 merges episodes separated by 48 hours or less, so
this requires two episodes whose index times are close but more than 48 hours
apart. Such specimens are reported in the log, and ``--multi-stay nearest``
assigns each to the episode whose index time is closest.

`sepsis_onset` is the index time of the stay: the labelled onset for sepsis
cases and the reference time sampled for control stays in step 1 with a fixed
seed, so that it is fully reproducible. Applying the same window to both groups
keeps the infection label comparable across the cohort; the positive rate in
each group is reported at the end of the run as a sanity check.

One consequence of that design should be stated in the manuscript rather than
left implicit. The infection label is ascertained inside a window anchored on
the stay's index time. For sepsis cases that anchor is a clinical event, while
for control stays it is a sampled reference time, so a control stay whose
positive culture falls outside the window is labelled non-infection even though
an infection occurred at some other point in the stay. The infection label is
therefore "culture-confirmed infection around the index time", not "infection at
any time during the ICU stay", and the reported prevalence should be read that
way. `--window-before-hours` / `--window-after-hours` control how wide that
window is.

Definition of a positive culture (`--positive-mode`), since this is site- and
laboratory-specific:

    organism-count  a positive report has `--organism-count-col` >= 1, i.e. at
                    least one organism was isolated
    flag            a positive report has `--positive-flag-col` equal to one of
                    `--positive-flag-values`
    all             every row of the culture file is already a positive report
                    (the file was pre-filtered upstream)

Contaminants are not removed automatically. If the protocol excludes common
skin flora isolated from a single blood-culture set, list them with
`--exclude-organisms` and state the rule in the manuscript.

Relationship to the published analysis
--------------------------------------
The default settings reproduce the published analysis: cultures are selected by
the time window alone, with no ICU-stay restriction and no collapsing of repeat
reports. The two optional settings below change the result and are off by
default:

    --restrict-to-icu-stay  drop cultures ordered outside the ICU stay. Not used
                            in the publication, and inappropriate for a cohort
                            admitted from the emergency department or a general
                            ward, where the confirming specimen often precedes
                            ICU admission.
    --dedup-hours H         collapse repeat reports of the same specimen within
                            H hours (0, the default, disables this).

Confirm the reproduction by checking the case and control counts against the
published cohort table before relying on any downstream output.

Data sources
------------
``--cohort`` and ``--cultures`` accept either

    a delimited text file      e.g.  output/sepsis_onset.csv
    a SQL query file           e.g.  sql:sql/culture_report.sql

In SQL mode the query is read from the given ``.sql`` file and executed against
the database named by the ``DB_URL`` environment variable:

    export DB_URL="<SQLAlchemy database URL>"
    python 01_02_infection_onset_auto_label.py --cultures sql:sql/culture_report.sql ...

Hospital-specific schema, table and column names therefore live in local .sql
files and credentials in the environment, and neither is committed to the
public repository. Source columns whose names differ from the canonical ones
below are remapped with ``--column-map RAW=CANONICAL``.

Outputs
-------
<outdir>/culture_positive.csv   the cleaned, deduplicated positive-culture table
<outdir>/culture_in_window.csv  every positive culture inside a stay's window
                                (long format, one row per culture, audit trail)
<outdir>/infection_onset.csv    one row per ICU stay, with `infection`,
                                `infection_onset` (positives only),
                                `infection_index_time` (defined for every stay,
                                consumed by step 5) and a summary of the isolates
<outdir>/run_config.json        all resolved parameters, for reproducibility

Usage
-----
    python 01_02_infection_onset_auto_label.py \
        --cohort   output/sepsis_onset.csv \
        --cultures data/culture_report.csv \
        --outdir   output

This file contains no patient-identifiable data.
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
# Canonical column names.
#
# These follow the originating institution's Oracle clinical data warehouse and
# are institutional conventions rather than part of the Sepsis-3 definition. To
# run the script against another schema, either edit this block or pass
# --column-map RAW=CANONICAL at run time; the logic below refers only to these
# names, never to a source table or column directly. See "Source data model" in
# the module docstring for the meaning and scope of each identifier.
# --------------------------------------------------------------------------- #

PATIENT_ID = "Pno"
STAY_ID = "adm_ICU_id"
STAY_KEY = [PATIENT_ID, STAY_ID]

INDEX_TIME = "sepsis_onset"     # index time of the stay, produced by step 1
CULTURE_TIME = "Orderdatetime"  # order timestamp of a culture report

ADMISSION_ID = "Firstcaseno"  # hospital-admission identifier, spans several ICU episodes
ADMISSION_TIME = "ICU_admdatetime"
DISCHARGE_TIME = "ICU_disdatetime"

# Descriptive culture columns carried into the long-format output, if present.
CULTURE_INFO_COLUMNS = ["Dic Name", "Mcode", "Bednsno", "Organ_Count", "sputum_gp"]

# Stay-level columns carried into the per-stay output, if present.
STAY_KEY_COLUMNS = [
    PATIENT_ID, STAY_ID, ADMISSION_ID, "Caseno", "Bedns", "Bedno",
    ADMISSION_TIME, DISCHARGE_TIME, INDEX_TIME, "label", "SOFA",
]

LOGGER = logging.getLogger("infection_onset")


# --------------------------------------------------------------------------- #
# Data-source layer -- file or SQL, so that Oracle object names stay external
# --------------------------------------------------------------------------- #

def load_table(spec: str, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Load a table from a delimited file or from a SQL query file.

    `spec` is either a path to a .csv/.tsv/.txt file, or ``sql:<path>.sql``.
    Everything is read as ``str``: identifiers must never be coerced to numeric
    types, because that turns ``"0012"`` into ``12`` and ``"12"`` into ``12.0``
    when nulls are present, silently breaking the join between the cohort and
    the culture table. Timestamps are converted afterwards by
    :func:`to_datetime`, since ``parse_dates`` together with ``dtype=str`` in a
    single ``read_csv`` call is undefined behaviour.
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
    # Oracle returns identifiers as NUMBER; normalise so that the file and SQL
    # paths behave identically downstream.
    return frame.astype("object").where(frame.notna(), None).astype("string")


def to_datetime(df: pd.DataFrame, columns: list[str], required: bool = True) -> pd.DataFrame:
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

    Applied to both tables, so that the join keys always agree; the original
    script applied this fix to one table only.
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
    present = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        LOGGER.warning("column(s) not found and therefore skipped: %s", missing)
    return present


# --------------------------------------------------------------------------- #
# Stage 1 -- load and clean the culture reports
# --------------------------------------------------------------------------- #

def load_cohort(spec: str, column_map: dict[str, str]) -> pd.DataFrame:
    """Load the step-1 cohort and verify that it has one row per ICU stay."""
    cohort = load_table(spec, column_map)
    require_columns(cohort, STAY_KEY + [INDEX_TIME, "label"], spec)
    cohort = to_datetime(cohort, [INDEX_TIME])
    cohort = to_datetime(cohort, [ADMISSION_TIME, DISCHARGE_TIME], required=False)
    for col in [c for c in STAY_KEY + [ADMISSION_ID, "Caseno"] if c in cohort.columns]:
        cohort[col] = normalise_id(cohort[col])
    # The sepsis label comes from step 1, which always writes 0 or 1. A missing
    # value here therefore means the upstream file is wrong, and defaulting it to
    # 0 would hide that by turning the affected stays into apparent non-sepsis
    # controls -- which would also shift the infection prevalence reported at the
    # end of this run.
    label_values = pd.to_numeric(cohort["label"], errors="coerce")
    n_bad_label = int(label_values.isna().sum())
    if n_bad_label:
        raise ValueError(
            f"{spec}: 'label' contains {n_bad_label} missing or non-numeric value(s). "
            "Re-run step 1; these stays must not be defaulted to non-sepsis."
        )
    observed_label = sorted(pd.unique(label_values.astype(int)))
    if not set(observed_label).issubset({0, 1}):
        raise ValueError(f"{spec}: 'label' must be 0/1; observed {observed_label}.")
    cohort["label"] = label_values.astype(int)

    n_missing = int(cohort[INDEX_TIME].isna().sum())
    if n_missing:
        LOGGER.warning("%d stay(s) without an index time were dropped", n_missing)
        cohort = cohort.dropna(subset=[INDEX_TIME])

    duplicated = int(cohort.duplicated(subset=STAY_KEY).sum())
    if duplicated:
        raise ValueError(
            f"the cohort file contains {duplicated} duplicated ICU stay(s); step 1 must emit "
            "exactly one row per stay, otherwise the merge below multiplies rows"
        )

    LOGGER.info("cohort loaded: %d ICU stay(s), %d sepsis case(s)",
                len(cohort), int((cohort["label"] == 1).sum()))
    return cohort.reset_index(drop=True)


def load_cultures(spec: str, column_map: dict[str, str], dedup_hours: float) -> pd.DataFrame:
    """Load the raw culture reports and clean them.

    This stage replaces the extraction code that was removed from the original
    script because it embedded hospital-internal (Oracle) table and column
    names; those now live in the .sql file or in ``--column-map``.

    Cleaning steps:
      * identifiers normalised to strings, timestamps parsed;
      * rows without a patient identifier or an order timestamp dropped, since
        they cannot be assigned to a window;
      * exact duplicate rows removed (repeated exports of the same report);
      * optionally, repeat reports of the same specimen for the same patient
        within `dedup_hours` collapsed to the first one, so a single specimen
        processed in several batches is not counted several times.
    """
    cultures = load_table(spec, column_map)
    require_columns(cultures, [PATIENT_ID, CULTURE_TIME], spec)
    n_raw = len(cultures)

    cultures[PATIENT_ID] = normalise_id(cultures[PATIENT_ID])
    for col in (ADMISSION_ID, STAY_ID):
        if col in cultures.columns:
            cultures[col] = normalise_id(cultures[col])
    cultures = to_datetime(cultures, [CULTURE_TIME])
    cultures = cultures.dropna(subset=[PATIENT_ID, CULTURE_TIME])
    cultures = cultures.drop_duplicates()

    if dedup_hours > 0:
        specimen_columns = [c for c in ["Dic Name", "Mcode"] if c in cultures.columns]
        if specimen_columns:
            cultures = cultures.sort_values([PATIENT_ID] + specimen_columns + [CULTURE_TIME])
            group = cultures.groupby([PATIENT_ID] + specimen_columns, sort=False)[CULTURE_TIME]
            gap_hours = group.diff().dt.total_seconds() / 3600.0
            keep = gap_hours.isna() | (gap_hours > dedup_hours)
            n_collapsed = int((~keep).sum())
            if n_collapsed:
                LOGGER.info("%d repeat report(s) of the same specimen within %.1f h collapsed",
                            n_collapsed, dedup_hours)
            cultures = cultures[keep]
        else:
            LOGGER.warning("no specimen column found; repeat-report deduplication skipped")

    cultures = cultures.sort_values([PATIENT_ID, CULTURE_TIME]).reset_index(drop=True)
    LOGGER.info("culture reports: %d raw row(s) -> %d after cleaning", n_raw, len(cultures))
    return cultures


def filter_positive_cultures(cultures: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Keep only the culture reports that count as positive."""
    n_before = len(cultures)

    if args.positive_mode == "organism-count":
        col = args.organism_count_col
        require_columns(cultures, [col], "culture table")
        count = pd.to_numeric(cultures[col], errors="coerce")
        n_bad = int(count.isna().sum() - cultures[col].isna().sum())
        if n_bad:
            LOGGER.warning("%s: %d non-numeric value(s) treated as missing", col, n_bad)
        # A report with no organism count is a no-growth report in this
        # laboratory's schema, so it counts as negative. That is the intended
        # behaviour, but the number of reports it applies to decides part of the
        # infection prevalence, so it is reported rather than left implicit: an
        # unexpectedly large number here usually means the extraction dropped the
        # count column for a subset of specimens.
        n_no_count = int(count.isna().sum())
        if n_no_count:
            LOGGER.warning(
                "%s: %d culture report(s) have no organism count and are counted as "
                "negative (%.1f%% of all reports). Verify that this reflects no growth "
                "rather than a gap in the extraction",
                col, n_no_count, 100.0 * n_no_count / max(len(cultures), 1),
            )
        positive = cultures[count.fillna(0) >= 1]

    elif args.positive_mode == "flag":
        col = args.positive_flag_col
        if not col:
            raise ValueError("--positive-flag-col is required when --positive-mode=flag")
        require_columns(cultures, [col], "culture table")
        values = {v.strip().casefold() for v in args.positive_flag_values}
        flag = cultures[col].astype("string").str.strip().str.casefold()
        positive = cultures[flag.isin(values)]

    else:  # "all"
        positive = cultures

    if args.exclude_organisms:
        col = args.organism_name_col
        if col in positive.columns:
            pattern = "|".join(args.exclude_organisms)
            excluded = positive[col].astype("string").str.contains(pattern, case=False, na=False)
            LOGGER.info("%d report(s) matching the contaminant list were excluded",
                        int(excluded.sum()))
            positive = positive[~excluded]
        else:
            LOGGER.warning("--exclude-organisms given but column '%s' is absent", col)

    LOGGER.info("positive cultures (%s): %d of %d report(s) retained",
                args.positive_mode, len(positive), n_before)
    return positive.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stage 2 -- interval join against the infection-confirmation window
# --------------------------------------------------------------------------- #

def cultures_in_window(
    cohort: pd.DataFrame,
    cultures: pd.DataFrame,
    before_hours: float,
    after_hours: float,
    restrict_to_icu_stay: bool,
    multi_stay: str,
    join_on: str,
) -> pd.DataFrame:
    """Return every positive culture falling inside a stay's infection window.

    The join is a single merge followed by a vectorised time filter. The
    original implementation looped over stays and called ``pd.concat`` on a
    growing global DataFrame once per iteration, which copies the whole result
    on every append (quadratic runtime).

    `join_on` selects the merge key; in every mode the infection window then
    decides which reports belong to the episode.

        admission (default)
                 merge on (Pno, Firstcaseno). Firstcaseno identifies one hospital
                 admission and is carried by emergency-department and general-ward
                 records as well as ICU records, so it scopes the search to the
                 correct hospitalisation without discarding anything obtained
                 outside the ICU. One hospital admission can contain several ICU
                 episodes, so Firstcaseno alone does not identify an episode --
                 the time window does that.
        patient  merge on Pno alone. Fallback for extractions without
                 Firstcaseno; a specimen from a different hospitalisation of the
                 same patient can then be attributed to this episode if it falls
                 within the window.
        stay     merge on (Pno, adm_ICU_id). Only appropriate when the culture
                 extraction is confined to the ICU: adm_ICU_id designates one ICU
                 admission and is not carried by emergency-department or
                 general-ward records, so any report without one is dropped.

    A specimen can still fall inside the windows of two ICU episodes of the same
    admission. Step 1 merges episodes separated by 48 hours or less, so this
    requires two episodes whose index times are close but more than 48 hours
    apart. Such specimens are reported, and ``--multi-stay nearest`` assigns each
    to the episode whose index time is closest.

    Specimens obtained outside the ICU stay -- in the emergency department or on
    a general ward -- are retained by default, because that is where a large
    part of this cohort is cultured. `restrict_to_icu_stay` reinstates the
    boundary check for cohorts where it applies.
    """
    key_by_mode = {
        "admission": [PATIENT_ID, ADMISSION_ID],
        "patient": [PATIENT_ID],
        "stay": STAY_KEY,
    }
    join_keys = key_by_mode[join_on]

    for key in join_keys:
        if key not in cultures.columns:
            raise KeyError(
                f"--join-on {join_on} requires the culture table to contain '{key}'. "
                "Use --column-map to supply it, or choose another --join-on."
            )
        n_unkeyed = int(cultures[key].isna().sum())
        if n_unkeyed:
            LOGGER.warning(
                "--join-on %s: %d culture report(s) have no %s and will be dropped",
                join_on, n_unkeyed, key,
            )
    LOGGER.info("culture reports are joined to the cohort on %s", join_keys)

    cohort_columns = available(cohort, STAY_KEY_COLUMNS)
    culture_columns = join_keys + [CULTURE_TIME] + available(cultures, CULTURE_INFO_COLUMNS)

    # Row identity lets us detect one specimen being claimed by several stays.
    cultures = cultures.reset_index(drop=True)
    cultures = cultures.assign(culture_row_id=cultures.index.astype("int64"))

    merged = cohort[cohort_columns].merge(
        cultures[culture_columns + ["culture_row_id"]],
        on=join_keys, how="inner", suffixes=("", "_culture"),
    )
    if merged.empty:
        LOGGER.warning("no culture report matched the cohort on %s", join_keys)
        return merged

    # The window is anchored on the stay's index time -- the labelled sepsis
    # onset for cases, the reference time sampled in step 1 for controls -- and
    # runs backwards `before_hours` and forwards `after_hours` from it. Both
    # boundaries are inclusive.
    window_start = merged[INDEX_TIME] - pd.Timedelta(hours=before_hours)
    window_end = merged[INDEX_TIME] + pd.Timedelta(hours=after_hours)
    merged["infection_window_start"] = window_start
    merged["infection_window_end"] = window_end
    in_window = merged[CULTURE_TIME].between(window_start, window_end, inclusive="both")

    if restrict_to_icu_stay and {ADMISSION_TIME, DISCHARGE_TIME} <= set(merged.columns):
        in_stay = merged[CULTURE_TIME].between(
            merged[ADMISSION_TIME], merged[DISCHARGE_TIME], inclusive="both"
        )
        n_outside = int((in_window & ~in_stay).sum())
        if n_outside:
            LOGGER.info("%d culture(s) inside the window but outside the ICU stay were dropped",
                        n_outside)
        in_window &= in_stay

    result = merged[in_window].copy()
    # Signed: negative = specimen ordered before the sepsis onset.
    result["hours_from_index"] = (
        (result[CULTURE_TIME] - result[INDEX_TIME]).dt.total_seconds() / 3600.0
    )

    if ADMISSION_TIME in result.columns:
        n_pre_icu = int((result[CULTURE_TIME] < result[ADMISSION_TIME]).sum())
        if n_pre_icu:
            LOGGER.info("%d culture(s) were obtained before ICU admission (emergency department) "
                        "and are retained", n_pre_icu)
        n_post_icu = int((result[CULTURE_TIME] > result[DISCHARGE_TIME]).sum()) \
            if DISCHARGE_TIME in result.columns else 0
        if n_post_icu:
            LOGGER.info("%d culture(s) were obtained after ICU discharge (general ward) and are "
                        "retained", n_post_icu)

    if join_on != "stay":
        result = _resolve_multi_stay(result, multi_stay)
    result = result.sort_values(STAY_KEY + [CULTURE_TIME]).reset_index(drop=True)

    LOGGER.info("%d positive culture(s) inside the window, across %d ICU stay(s)",
                len(result), result.drop_duplicates(STAY_KEY).shape[0])
    return result


def _resolve_multi_stay(result: pd.DataFrame, multi_stay: str) -> pd.DataFrame:
    """Report, and optionally resolve, specimens claimed by several ICU stays."""
    claimed = result["culture_row_id"].duplicated(keep=False)
    n_ambiguous = int(result.loc[claimed, "culture_row_id"].nunique())
    if not n_ambiguous:
        return result

    LOGGER.warning(
        "%d specimen(s) fall inside the infection window of more than one ICU stay of the "
        "same patient", n_ambiguous,
    )
    if multi_stay != "nearest":
        return result

    keep = (
        result.assign(_distance=result["hours_from_index"].abs())
        .sort_values(["culture_row_id", "_distance"])
        .drop_duplicates(subset="culture_row_id", keep="first")
        .index
    )
    LOGGER.info("each ambiguous specimen was assigned to the ICU stay with the closest index time")
    return result.loc[keep]


# --------------------------------------------------------------------------- #
# Stage 3 -- collapse to one row per ICU stay and merge back onto the cohort
# --------------------------------------------------------------------------- #

def summarise_by_stay(in_window: pd.DataFrame, organism_col: str | None) -> pd.DataFrame:
    """Collapse the long-format table to one row per ICU stay.

    This stage restores the aggregation that the original script performed on an
    intermediate table before the final merge. Aggregating first is what keeps
    the merge one-to-one: joining the long culture table straight onto the
    cohort would multiply a stay into as many rows as it has cultures.
    """
    summary_columns = STAY_KEY + [
        "infection", "infection_onset", "n_positive_cultures", "hours_from_index",
        "infection_window_start", "infection_window_end",
    ]
    if in_window.empty:
        return pd.DataFrame(columns=summary_columns)

    # `infection_onset` is the EARLIEST positive culture inside the window, not
    # the earliest culture overall and not the report time: the window is opened
    # around the sepsis onset and the first qualifying specimen inside it dates
    # the infection. `hours_from_index` is signed, so a negative value means the
    # specimen preceded the sepsis onset, which is common when the patient was
    # cultured in the emergency department before deteriorating.
    aggregations = dict(
        infection_onset=(CULTURE_TIME, "min"),
        n_positive_cultures=(CULTURE_TIME, "size"),
        hours_from_index=("hours_from_index", "min"),
    )
    # Carrying the window boundaries through to the output makes the labelling
    # auditable: a reader can confirm from the CSV alone that every
    # infection_onset falls inside its own window, without re-running anything.
    for column in ("infection_window_start", "infection_window_end"):
        if column in in_window.columns:
            aggregations[column] = (column, "first")

    summary = in_window.groupby(STAY_KEY, as_index=False).agg(**aggregations)

    if organism_col and organism_col in in_window.columns:
        organisms = (
            in_window.dropna(subset=[organism_col])
            .groupby(STAY_KEY)[organism_col]
            .agg(lambda values: "; ".join(sorted(set(values.astype(str)))))
            .rename("organisms")
            .reset_index()
        )
        summary = summary.merge(organisms, on=STAY_KEY, how="left")

    summary["infection"] = 1
    return summary


def merge_infection_label(cohort: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Left-join the per-stay summary onto the cohort and fill the negatives."""
    labelled = cohort.merge(summary, on=STAY_KEY, how="left", validate="one_to_one")
    if len(labelled) != len(cohort):
        raise AssertionError("the merge changed the number of rows; check the stay keys")
    labelled["infection"] = labelled["infection"].fillna(0).astype(int)
    labelled["n_positive_cultures"] = labelled["n_positive_cultures"].fillna(0).astype(int)

    # A stay without a positive culture has no infection onset. Step 5 anchors the
    # feature window on a timestamp that must exist for both classes, so a single
    # index time is emitted here: the first positive culture for infection cases,
    # and otherwise the step-1 index time (the labelled sepsis onset, or for
    # control stays the reference time drawn with a fixed seed). Keeping
    # `infection_onset` empty for the negatives, and providing this column
    # alongside it, makes the distinction explicit rather than implicit.
    labelled["infection_index_time"] = labelled["infection_onset"].fillna(labelled[INDEX_TIME])
    labelled["infection_index_source"] = np.where(
        labelled["infection_onset"].notna(), "infection_onset", INDEX_TIME
    )
    return labelled


def report_label_agreement(labelled: pd.DataFrame) -> None:
    """Print the sepsis x infection cross-tabulation as a sanity check."""
    crosstab = pd.crosstab(labelled["label"], labelled["infection"])
    LOGGER.info("sepsis label (rows) x culture-confirmed infection (columns):\n%s", crosstab)
    for label_value, group in labelled.groupby("label"):
        LOGGER.info("label=%d: %.1f%% of stays have a positive culture in the window (n=%d)",
                    label_value, 100.0 * group["infection"].mean(), len(group))


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
    LOGGER.info(
        "infection-confirmation window: [sepsis_onset - %.0f h, sepsis_onset + %.0f h]. "
        "This is anchored on the sepsis onset and is NOT the step-1 order-pairing rule "
        "(culture->antibiotic 72 h, antibiotic->culture 24 h); see the module docstring",
        args.window_before_hours, args.window_after_hours,
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    column_map = parse_column_map(args.column_map)

    cohort = load_cohort(args.cohort, column_map)
    cultures = load_cultures(args.cultures, column_map, args.dedup_hours)

    positive = filter_positive_cultures(cultures, args)
    positive.to_csv(outdir / "culture_positive.csv", index=False, encoding="utf-8-sig")

    in_window = cultures_in_window(
        cohort, positive,
        before_hours=args.window_before_hours,
        after_hours=args.window_after_hours,
        restrict_to_icu_stay=args.restrict_to_icu_stay,
        multi_stay=args.multi_stay,
        join_on=args.join_on,
    )
    in_window.to_csv(outdir / "culture_in_window.csv", index=False, encoding="utf-8-sig")

    summary = summarise_by_stay(in_window, args.organism_name_col)
    labelled = merge_infection_label(cohort, summary)

    out_path = outdir / "infection_onset.csv"
    labelled.to_csv(out_path, index=False, encoding="utf-8-sig")
    (outdir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    LOGGER.info("cohort written to %s: %d stay(s)", out_path, len(labelled))
    report_label_agreement(labelled)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automatic labelling of culture-confirmed infection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cohort", required=True,
                        help="sepsis_onset.csv from step 1: file path or sql:<file>.sql")
    parser.add_argument("--cultures", required=True,
                        help="culture reports: file path or sql:<file>.sql")
    parser.add_argument("--outdir", default="output", help="output directory")
    parser.add_argument("--column-map", nargs="*", default=[], metavar="RAW=CANONICAL",
                        help="rename source columns, e.g. PATIENT_NO=Pno ORDER_DTM=Orderdatetime")

    parser.add_argument("--window-before-hours", type=float, default=24.0,
                        help="infection-confirmation window BEFORE the sepsis onset. Note that "
                             "this window is anchored on the sepsis onset and is a different "
                             "thing from the order-pairing windows of step 1, which are anchored "
                             "on a culture/antibiotic pair; see the module docstring")
    parser.add_argument("--window-after-hours", type=float, default=72.0,
                        help="infection-confirmation window AFTER the sepsis onset. Longer than "
                             "the backward side because culture turnaround is 3-5 days")
    parser.add_argument("--restrict-to-icu-stay", action="store_true",
                        help="drop cultures ordered outside the ICU stay; off by default, since "
                             "specimens from the emergency department or a general ward before "
                             "ICU admission are part of the infection window")
    parser.add_argument("--join-on", choices=["admission", "patient", "stay"],
                        default="admission",
                        help="merge key between cohort and cultures; the default scopes the "
                             "search to one hospital admission (Pno + Firstcaseno) and lets the "
                             "time window pick the ICU episode")
    parser.add_argument("--multi-stay", choices=["all", "nearest"], default="all",
                        help="how to handle a specimen falling inside the window of more than "
                             "one ICU episode (--join-on admission or patient)")

    parser.add_argument("--positive-mode", choices=["organism-count", "flag", "all"],
                        default="organism-count", help="rule defining a positive culture")
    parser.add_argument("--organism-count-col", default="Organ_Count",
                        help="column holding the number of isolated organisms")
    parser.add_argument("--positive-flag-col", default=None,
                        help="column holding an explicit positive/negative flag")
    parser.add_argument("--positive-flag-values", nargs="+", default=["1", "P", "positive"],
                        help="values of the flag column denoting a positive report")
    parser.add_argument("--organism-name-col", default="sputum_gp",
                        help="column holding the organism or specimen group name")
    parser.add_argument("--exclude-organisms", nargs="*", default=[],
                        help="regular expressions of organisms treated as contaminants")
    parser.add_argument("--dedup-hours", type=float, default=0.0,
                        help="collapse repeat reports of the same specimen within this many "
                             "hours; 0 disables")

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
    except (FileNotFoundError, KeyError, ValueError, AssertionError) as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
