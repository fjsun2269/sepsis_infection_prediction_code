#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (C) 2026 Fang-Ju Sun and contributors
"""
No-balancing reduced-feature TabNet model for sepsis internal and external validation

This script is a GitHub-public, command-line version of the no-balancing
sensitivity analysis. It assumes that cohort selection, train/test splitting,
physiological clipping, and missing-data imputation have already been completed.

Important:
    - This script does NOT apply propensity score matching.
    - This script does NOT apply BorderlineSMOTE or any other oversampling.
    - This script does NOT perform probability calibration.
    - Z-score scaling is fitted only on the training fold during cross-validation
      and only on the full development cohort for the final model.
    - Everything else is held identical to the main analysis, so that the
      comparison isolates the effect of class balancing. In particular APACHE II
      is aggregated as the maximum within the 8-hour window
      (`--apache-aggregation max`, the default), which is what the main-analysis
      and calibrated scripts do. An earlier version of this file took the first
      non-missing value instead, which meant the sensitivity analysis differed
      from the main analysis in two ways at once.

Expected input:
    A preprocessed development/train CSV and an independent internal test CSV.
    For reduced external validation scripts, an external CSV is also required.

The input can be either:
    1. already engineered one-row-per-ICU-admission features containing the
       final model columns, or
    2. long-format 8-hour rows containing the raw variables listed below.
       In that case, this script will aggregate each ICU admission into the
       predefined minimum/maximum feature representation.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:  # Allow --help and static inspection without optional ML dependency.
    TabNetClassifier = None  # type: ignore[assignment,misc]


LOGGER = logging.getLogger("no_balance_model")


# The public feature names used by the main-analysis scripts and by the
# manuscript supplement. The internal column names below are the older notebook
# ones; both were appearing in this repository's output files, so every table
# written from here reports the public name alongside the internal one.
# Computation is unaffected: only the labels change.
PUBLIC_FEATURE_NAMES = {
    "min_PFratio": "min_PaO2_FiO2_ratio",
    "min_GCS": "min_GCS",
    "min_Urine_output_8H": "min_8_hour_urine_output",
    "max_Temperature": "max_Temperature",
    "max_Pulse": "max_PR",
    "max_Respiration": "max_RR",
    "max_SBP": "max_SBP",
    "APACHEII_score": "APACHE_II_score",
    "max_WBC": "max_WBC",
    "max_CRP": "max_CRP",
    "max_PCT": "max_PCT",
    "max_Neutrophil_Seg": "max_Segmented_neutrophil_percentage",
    "max_Ventilator_days": "max_Ventilator_days",
    "max_Antibiotics": "max_Antibiotic_count",
    "max_Norepinephrine": "max_Norepinephrine",
    "max_Dopamine": "max_Dopamine",
}


def to_public(name: str) -> str:
    """Public, manuscript-facing name of an internal feature column."""
    return PUBLIC_FEATURE_NAMES.get(name, name)


OUTCOME_NAME = "sepsis"
DEFAULT_TARGET_COL = "label"
DEFAULT_EXTERNAL_TARGET_COL = "sepsis_label"
OUTPUT_SUBDIR = "no_balance_reduced_external_sepsis"

RAW_FEATURE_COLUMNS: List[str] = ['PFratio', 'GCS', 'Urine_output_8H', 'Temperature', 'Pulse', 'Respiration', 'SBP', 'WBC', 'CRP', 'Ventilator_days', 'Antibiotics']

MODEL_FEATURE_COLUMNS: List[str] = ['min_PFratio', 'min_GCS', 'min_Urine_output_8H', 'max_Temperature', 'max_Pulse', 'max_Respiration', 'max_SBP', 'max_WBC', 'max_CRP', 'max_Ventilator_days', 'max_Antibiotics']


COLUMN_ALIASES = {
    # Engineered features may arrive under the public names written by the
    # main-analysis scripts; accept them and map back to the internal names.
    "min_PaO2_FiO2_ratio": "min_PFratio",
    "min_8_hour_urine_output": "min_Urine_output_8H",
    "max_PR": "max_Pulse",
    "max_RR": "max_Respiration",
    "APACHE_II_score": "APACHEII_score",
    "max_Antibiotic_count": "max_Antibiotics",
    "max_Segmented_neutrophil_percentage": "max_Neutrophil_Seg",
    "PaO2_FiO2_ratio": "PFratio",
    "PaO2_FiO2": "PFratio",
    "PF_ratio": "PFratio",
    "PaO2/FiO2": "PFratio",
    "APACHE_II_score": "APACHEII_score",
    "APACHE II score": "APACHEII_score",
    "8_hour_urine_output": "Urine_output_8H",
    "8-hour urine output": "Urine_output_8H",
    "PR": "Pulse",
    "RR": "Respiration",
    "Antibiotic_count": "Antibiotics",
    "Segmented_neutrophil_percentage": "Neutrophil_Seg",
    "Segmented neutrophil percentage": "Neutrophil_Seg",
}

KEEP_COLUMNS: List[str] = [
    "Pno",
    "Firstcaseno",
    "Caseno",
    "Bedns",
    "Bedno",
    "ICU_admdatetime",
    "ICU_disdatetime",
    "adm_ICU_id",
    "sepsis_onset",
    "date",
    "sepsis_label",
    "culture_report",
    "label",
    "Gender",
    "Age",
    "CCI_score",
    "APACHEII_score",
]



IDENTIFIER_COLUMNS: List[str] = [
    "Pno", "Firstcaseno", "Caseno", "Bedns", "Bedno",
    "ICU_admdatetime", "ICU_disdatetime", "adm_ICU_id",
    "sepsis_onset", "infection_onset", "date", "charttime",
]


def drop_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove direct and quasi identifiers before writing anything to disk."""

    return df.drop(columns=[c for c in IDENTIFIER_COLUMNS if c in df.columns])


@dataclass
class DatasetBundle:
    """Container for model-ready features and labels."""

    features: pd.DataFrame
    labels: np.ndarray
    metadata: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="No-balancing reduced-feature TabNet model for sepsis internal and external validation"
    )
    parser.add_argument(
        "--train-input",
        required=True,
        help="Path to the completed development/training cohort CSV.",
    )
    parser.add_argument(
        "--test-input",
        required=True,
        help="Path to the completed independent internal test cohort CSV.",
    )
    parser.add_argument(
        "--external-input",
        required=True,
        help="Path to the completed external validation cohort CSV.",
    )
    parser.add_argument(
        "--target-col",
        default=DEFAULT_TARGET_COL,
        help=(
            "Outcome column for development and internal test data. "
            f"Default: {DEFAULT_TARGET_COL}"
        ),
    )
    parser.add_argument(
        "--external-target-col",
        default=DEFAULT_EXTERNAL_TARGET_COL,
        help=(
            "Outcome column for external validation data. "
            f"Default: {DEFAULT_EXTERNAL_TARGET_COL}"
        ),
    )
    parser.add_argument(
        "--group-col",
        default="adm_ICU_id",
        help="ICU-admission-level grouping column for long-format inputs.",
    )
    parser.add_argument(
        "--time-col",
        default="date",
        help="Time column used to sort long-format rows before aggregation.",
    )
    parser.add_argument(
        "--outdir",
        default=f"outputs/{OUTPUT_SUBDIR}",
        help="Output directory.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated probability thresholds for final evaluation.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum TabNet training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early-stopping patience.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="TabNet batch size.",
    )
    parser.add_argument(
        "--virtual-batch-size",
        type=int,
        default=32,
        help="TabNet virtual batch size.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--apache-aggregation",
        choices=["max", "first"],
        default="max",
        help=(
            "Aggregation applied to APACHE II within the 8-hour window. 'max' matches the "
            "main-analysis and calibrated scripts, so that this sensitivity analysis differs "
            "from the main analysis only by the absence of class balancing. 'first' reproduces "
            "the earlier behaviour of this file."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def read_table(path: str) -> pd.DataFrame:
    """Read a CSV or parquet table."""

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path_obj.suffix.lower()
    if suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path_obj)
    return pd.read_csv(path_obj)


def canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply public/legacy column aliases without overwriting canonical columns."""
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    rename = {
        old: new for old, new in COLUMN_ALIASES.items()
        if old in df.columns and new not in df.columns
    }
    return df.rename(columns=rename) if rename else df


def resolve_target_column(df: pd.DataFrame, requested: str, outcome_name: str) -> str:
    """Resolve the requested outcome column with safe fallbacks."""

    if requested in df.columns:
        return requested

    fallback_candidates: List[str] = []
    if outcome_name == "sepsis":
        fallback_candidates = ["label", "sepsis_label"]
    elif outcome_name == "infection":
        fallback_candidates = ["culture_report", "infection_label", "infection"]

    for col in fallback_candidates:
        if col in df.columns:
            LOGGER.warning(
                "Requested target column '%s' was not found; using '%s' instead.",
                requested, col,
            )
            return col

    raise ValueError(
        f"Could not find target column '{requested}' or valid fallbacks "
        f"for {outcome_name}: {fallback_candidates}"
    )


def create_binary_label(df: pd.DataFrame, target_col: str, label_col: str = "label") -> pd.DataFrame:
    """Create a numeric binary label column."""

    out = df.copy()
    values = pd.to_numeric(out[target_col], errors="coerce")

    # The previous version applied .fillna(0) before validating, so a missing or
    # unparsable outcome became a silent negative and passed the 0/1 check that
    # follows. Missing outcomes are now an error.
    n_missing = int(values.isna().sum())
    if n_missing:
        raise ValueError(
            f"Target column '{target_col}' contains {n_missing} missing or "
            "non-numeric value(s). These rows must not be treated as negatives; "
            "fix the upstream labelling step."
        )

    out[label_col] = values.astype(int)
    invalid_values = sorted(set(out[label_col].unique()) - {0, 1})
    if invalid_values:
        raise ValueError(f"Label column contains values outside 0/1: {invalid_values}")

    return out


def has_engineered_features(df: pd.DataFrame) -> bool:
    """Return True when the input already contains final model features."""

    return all(col in df.columns for col in MODEL_FEATURE_COLUMNS)


def first_non_missing_numeric(series: pd.Series) -> float:
    """Return the first non-missing numeric value in a series."""

    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[numeric.notna()]
    return float(numeric.iloc[0]) if len(numeric) > 0 else float("nan")


def aggregate_long_format_to_features(
    df_raw: pd.DataFrame,
    group_col: str,
    time_col: str,
    apache_aggregation: str = "max",
) -> pd.DataFrame:
    """
    Convert long-format 8-hour rows into one row per ICU admission.

    This follows the model input representation used in the manuscript:
    each ICU admission is summarized by predefined minimum or maximum values
    within the 8-hour feature window.

    `apache_aggregation` defaults to "max", which is what the main-analysis and
    calibrated scripts use. An earlier version of this file took the first
    non-missing APACHE II value instead, which meant this sensitivity analysis
    differed from the main analysis in two ways at once -- class balancing and
    APACHE II aggregation -- and so could not isolate the effect of balancing,
    which is the whole point of the comparison. Pass "first" to reproduce that
    earlier behaviour.
    """

    if group_col not in df_raw.columns:
        raise ValueError(
            f"Long-format input requires group column '{group_col}'. "
            "If your file is already engineered, it must contain all final "
            "MODEL_FEATURE_COLUMNS."
        )

    work = df_raw.copy()
    if time_col in work.columns:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")

    rows = []
    for _, group in tqdm(work.groupby(group_col), desc="Aggregating 8-hour features"):
        if time_col in group.columns:
            group = group.sort_values(time_col).copy()
        else:
            group = group.copy()

        # Missing-data handling is completed upstream. This no-balancing
        # sensitivity analysis only aggregates the fixed 8-hour rows; it does not
        # perform a second forward-fill step that could make it differ from the
        # main analysis for reasons other than class balancing.
        group = group.infer_objects(copy=False)

        available = [c for c in RAW_FEATURE_COLUMNS if c in group.columns]
        if not available:
            continue

        selected = group[available].copy()
        for col in available:
            selected[col] = pd.to_numeric(selected[col], errors="coerce")

        # Build a plain dict per admission. The previous version created a
        # one-row DataFrame per admission and concatenated all of them at the
        # end, which is quadratic and does not scale to a cohort of this size.
        summary: Dict[str, object] = {}
        summary.update({f"mean_{k}": v for k, v in selected.mean().items()})
        summary.update({f"max_{k}": v for k, v in selected.max().items()})
        summary.update({f"min_{k}": v for k, v in selected.min().items()})

        # APACHE II is treated as an admission/severity score rather than a
        # time-varying min/max feature in the final full institutional models.
        if "APACHEII_score" in MODEL_FEATURE_COLUMNS and "APACHEII_score" in group.columns:
            apache_values = pd.to_numeric(group["APACHEII_score"], errors="coerce")
            summary["APACHEII_score"] = (
                float(apache_values.max()) if apache_aggregation == "max"
                else first_non_missing_numeric(group["APACHEII_score"])
            )

        for col in KEEP_COLUMNS:
            if col in group.columns and col not in summary:
                summary[col] = group[col].iloc[0]

        rows.append(summary)

    if not rows:
        raise ValueError("No feature rows were generated from the input data.")

    return pd.DataFrame(rows)


def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    time_col: str,
    dataset_name: str,
    apache_aggregation: str = "max",
) -> DatasetBundle:
    """Prepare model features, labels, and metadata for one dataset."""

    df = canonicalise_columns(df)
    target_col = resolve_target_column(df, target_col, OUTCOME_NAME)
    labeled = create_binary_label(df, target_col=target_col, label_col="label")

    if has_engineered_features(labeled):
        feature_df = labeled.copy()
    else:
        feature_df = aggregate_long_format_to_features(
            labeled,
            group_col=group_col,
            time_col=time_col,
            apache_aggregation=apache_aggregation,
        )

        # Rebuild admission-level label after aggregation. An admission that
        # fails to map back is an error: the previous version let it through and
        # a later .fillna(0) turned it into a negative case.
        if group_col in labeled.columns and group_col in feature_df.columns:
            label_map = labeled.groupby(group_col)["label"].max()
            mapped = feature_df[group_col].map(label_map)
            n_unmapped = int(mapped.isna().sum())
            if n_unmapped:
                raise ValueError(
                    f"{dataset_name}: {n_unmapped} ICU admission(s) could not be matched "
                    f"back to a label through '{group_col}' after aggregation. These must "
                    "not be treated as negatives; check the grouping column for type "
                    "mismatches between the feature and label tables."
                )
            feature_df["label"] = mapped.astype(int)
        elif "label" not in feature_df.columns:
            raise ValueError(
                f"Could not map labels for {dataset_name}. "
                f"Expected grouping column '{group_col}'."
            )

    missing_cols = [c for c in MODEL_FEATURE_COLUMNS if c not in feature_df.columns]
    if missing_cols:
        raise ValueError(
            f"{dataset_name} is missing required model features: {missing_cols}"
        )

    for col in MODEL_FEATURE_COLUMNS:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

    # Final validation of the label. This used to be a second `.fillna(0)`,
    # which re-opened the silent-negative path that create_binary_label closes:
    # anything that had become missing between there and here was quietly
    # relabelled as a non-case, inflating specificity and NPV without warning.
    label_values = pd.to_numeric(feature_df["label"], errors="coerce")
    n_bad_labels = int(label_values.isna().sum())
    if n_bad_labels:
        raise ValueError(
            f"{dataset_name} has {n_bad_labels} admission(s) with a missing or "
            "non-numeric outcome after aggregation. Fix the upstream labelling "
            "step rather than defaulting them to 0."
        )
    observed_labels = sorted(set(label_values.astype(int).unique()))
    if not set(observed_labels).issubset({0, 1}):
        raise ValueError(
            f"{dataset_name} outcome must be 0/1; observed {observed_labels}."
        )
    feature_df["label"] = label_values.astype(int)

    missing_counts = feature_df[MODEL_FEATURE_COLUMNS].isna().sum()
    if missing_counts.sum() > 0:
        raise ValueError(
            f"{dataset_name} still contains missing model features after the "
            "completed preprocessing pipeline. Missing counts:\n"
            f"{missing_counts[missing_counts > 0]}"
        )

    # KEEP_COLUMNS is retained because the aggregation step needs those columns
    # while the table is still in memory, but nothing identifying is carried
    # into the object that later gets written to disk.
    metadata_cols = [
        c for c in KEEP_COLUMNS
        if c in feature_df.columns and c not in IDENTIFIER_COLUMNS
    ]
    metadata = feature_df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=feature_df.index)

    return DatasetBundle(
        features=feature_df[MODEL_FEATURE_COLUMNS].copy(),
        labels=feature_df["label"].astype(int).to_numpy(),
        metadata=metadata.reset_index(drop=True),
    )


def safe_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """AUROC with NaN fallback for single-class edge cases."""

    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def safe_div(numerator: float, denominator: float) -> float:
    """Safe division with NaN for zero denominator."""

    return float(numerator / denominator) if denominator > 0 else float("nan")


def calculate_metrics(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate threshold-based and probability-based metrics."""

    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)
    y_pred = (y_proba_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_arr, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "Threshold": threshold,
        "AUROC": safe_auc(y_true_arr, y_proba_arr),
        "Accuracy": accuracy_score(y_true_arr, y_pred),
        "F1 Score": f1_score(y_true_arr, y_pred, zero_division=0),
        "Sensitivity": recall_score(y_true_arr, y_pred, zero_division=0),
        "Specificity": safe_div(tn, tn + fp),
        "PPV": precision_score(y_true_arr, y_pred, zero_division=0),
        "NPV": safe_div(tn, tn + fn),
        "Brier": brier_score_loss(y_true_arr, y_proba_arr),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def create_tabnet(seed: int) -> TabNetClassifier:
    """Create a TabNet classifier with fixed, transparent hyperparameters."""

    return TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-3),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=seed,
    )


def fit_tabnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: Optional[np.ndarray],
    y_valid: Optional[np.ndarray],
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    virtual_batch_size: int,
) -> TabNetClassifier:
    """Fit TabNet with optional validation data."""

    clf = create_tabnet(seed=seed)

    if X_valid is None or y_valid is None:
        eval_set = [(X_train, y_train)]
        eval_name = ["train"]
    else:
        eval_set = [(X_train, y_train), (X_valid, y_valid)]
        eval_name = ["train", "valid"]

    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=eval_set,
        eval_name=eval_name,
        eval_metric=["auc", "logloss"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        num_workers=0,
        drop_last=False,
    )
    return clf


def run_cross_validation(
    train_bundle: DatasetBundle,
    args: argparse.Namespace,
    outdir: Path,
) -> pd.DataFrame:
    """Run stratified 5-fold CV without PSM or oversampling."""

    X_all = train_bundle.features.reset_index(drop=True)
    y_all = train_bundle.labels

    skf = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    per_fold_rows = []
    feature_importances = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_all, y_all), start=1):
        seed = args.seed + fold
        np.random.seed(seed)
        torch.manual_seed(seed)

        X_train_df = X_all.iloc[train_idx].copy()
        X_valid_df = X_all.iloc[valid_idx].copy()
        y_train = y_all[train_idx]
        y_valid = y_all[valid_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_df).astype(float)
        X_valid = scaler.transform(X_valid_df).astype(float)

        clf = fit_tabnet(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            seed=seed,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            virtual_batch_size=args.virtual_batch_size,
        )

        for set_name, X_eval, y_eval in [
            ("train_no_balancing", X_train, y_train),
            ("valid_no_balancing", X_valid, y_valid),
        ]:
            y_proba = clf.predict_proba(X_eval)[:, 1]
            row = calculate_metrics(y_eval, y_proba, threshold=0.5)
            row.update({"Fold": fold, "Set": set_name})
            per_fold_rows.append(row)

        feature_importances.append(clf.feature_importances_)

    metrics_df = pd.DataFrame(per_fold_rows)
    metrics_df.to_csv(outdir / "cv_metrics_no_balancing.csv", index=False, encoding="utf-8-sig")

    summary_df = (
        metrics_df
        .groupby("Set")[
            [
                "AUROC",
                "Accuracy",
                "F1 Score",
                "Sensitivity",
                "Specificity",
                "PPV",
                "NPV",
                "Brier",
            ]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    summary_df.to_csv(outdir / "cv_summary_no_balancing.csv", encoding="utf-8-sig")

    pd.DataFrame(
        feature_importances,
        columns=[to_public(c) for c in MODEL_FEATURE_COLUMNS],
    ).to_csv(outdir / "cv_feature_importances_no_balancing.csv", index=False, encoding="utf-8-sig")

    return metrics_df


def train_final_model(
    train_bundle: DatasetBundle,
    args: argparse.Namespace,
    outdir: Path,
) -> Tuple[TabNetClassifier, StandardScaler]:
    """Train the final no-balancing TabNet model on the full development cohort."""

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_bundle.features).astype(float)
    y_train = train_bundle.labels

    clf = fit_tabnet(
        X_train=X_train,
        y_train=y_train,
        X_valid=None,
        y_valid=None,
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        virtual_batch_size=args.virtual_batch_size,
    )

    joblib.dump(scaler, outdir / "zscore_scaler_no_balancing.joblib")
    clf.save_model(str(outdir / "tabnet_model_no_balancing"))

    pd.DataFrame(
        {
            "feature": [to_public(c) for c in MODEL_FEATURE_COLUMNS],
            "feature_internal_name": MODEL_FEATURE_COLUMNS,
            "importance": clf.feature_importances_,
        }
    ).to_csv(outdir / "final_feature_importance_no_balancing.csv", index=False, encoding="utf-8-sig")

    return clf, scaler


def evaluate_dataset(
    name: str,
    bundle: DatasetBundle,
    clf: TabNetClassifier,
    scaler: StandardScaler,
    thresholds: Sequence[float],
    outdir: Path,
) -> pd.DataFrame:
    """Evaluate a fitted model on an untouched validation/test dataset."""

    X_eval = scaler.transform(bundle.features).astype(float)
    y_eval = bundle.labels
    y_proba = clf.predict_proba(X_eval)[:, 1]

    # Prediction files are release artefacts, so they carry no patient-level
    # attributes at all: row number, true label and score only.
    pred_df = pd.DataFrame(
        {
            "row_number": np.arange(len(y_eval)),
            "true_label": y_eval,
            "raw_probability": y_proba,
            "raw_prediction_at_0_5": (y_proba >= 0.5).astype(int),
        }
    )
    pred_df.to_csv(outdir / f"{name}_predictions_no_balancing.csv", index=False, encoding="utf-8-sig")

    metric_rows = []
    for threshold in thresholds:
        row = calculate_metrics(y_eval, y_proba, threshold=threshold)
        row["Set"] = name
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(outdir / f"{name}_threshold_metrics_no_balancing.csv", index=False, encoding="utf-8-sig")
    return metrics_df


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    if TabNetClassifier is None:
        raise ImportError(
            "pytorch-tabnet is required for model training. "
            "Install the project dependencies before running this analysis."
        )
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2")
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    if not thresholds or any(value < 0 or value > 1 for value in thresholds):
        raise ValueError("--thresholds must contain values between 0 and 1")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_df = read_table(args.train_input)
    test_df = read_table(args.test_input)

    train_bundle = prepare_features(
        train_df,
        target_col=args.target_col,
        group_col=args.group_col,
        time_col=args.time_col,
        dataset_name="development/train data",
        apache_aggregation=args.apache_aggregation,
    )
    test_bundle = prepare_features(
        test_df,
        target_col=args.target_col,
        group_col=args.group_col,
        time_col=args.time_col,
        dataset_name="internal test data",
        apache_aggregation=args.apache_aggregation,
    )

    run_config = {
        "outcome_name": OUTCOME_NAME,
        "default_target_col": DEFAULT_TARGET_COL,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "model_feature_columns_public": [to_public(c) for c in MODEL_FEATURE_COLUMNS],
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "apache_aggregation": args.apache_aggregation,
        "balancing": "none",
        "psm": False,
        "borderline_smote": False,
        "calibration": False,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "thresholds": thresholds,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "virtual_batch_size": args.virtual_batch_size,
    }
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    run_cross_validation(train_bundle, args, outdir)
    clf, scaler = train_final_model(train_bundle, args, outdir)

    all_metrics = []
    all_metrics.append(
        evaluate_dataset(
            name="internal_test",
            bundle=test_bundle,
            clf=clf,
            scaler=scaler,
            thresholds=thresholds,
            outdir=outdir,
        )
    )

    external_df = read_table(args.external_input)
    external_bundle = prepare_features(
        external_df,
        target_col=args.external_target_col,
        group_col=args.group_col,
        time_col=args.time_col,
        dataset_name="external validation data",
        apache_aggregation=args.apache_aggregation,
    )
    all_metrics.append(
        evaluate_dataset(
            name="external_validation",
            bundle=external_bundle,
            clf=clf,
            scaler=scaler,
            thresholds=thresholds,
            outdir=outdir,
        )
    )

    pd.concat(all_metrics, ignore_index=True).to_csv(
        outdir / "all_final_threshold_metrics_no_balancing.csv",
        index=False,
        encoding="utf-8-sig",
    )

    LOGGER.info("Finished. Outputs written to %s", outdir)


if __name__ == "__main__":
    main()
