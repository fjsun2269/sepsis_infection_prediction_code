# Real-time clinical decision support system for early identification of infection and sepsis in the intensive care unit: a retrospective development and prospective deployment study

Study code and downstream analytical workflows associated with the article:

> **Real-time clinical decision support system for early identification of infection and sepsis in the intensive care unit: a retrospective development and prospective deployment study**  
> **First author:** Fang-Ju Sun  
> **Authors:** Fang-Ju Sun, Yen-Yu Liu, Li-Kuo Kuo, Ting-Yu Hu, Kuang-Hua Cheng, Hung-Ting Chen, Po-Jen Chang, Min-Ching Wu, Hung-I Yeh, and Kun-Pin Wu  
> *BMJ Quality & Safety* (2026). <https://doi.org/10.1136/bmjqs-2026-020104>

Additional supplemental material is published online only and can be viewed
through the journal.

This repository contains the data-preparation pipeline (`01_*`) and the publicly
released `04_*` modelling track reported in the manuscript. The `04_*` scripts
contain the executable model-construction and training logic, including the
TabNet architecture setup, the optimizer and learning rate, learning-rate
scheduler, cross-validation and split settings, random seeds, epoch and
early-stopping limits, batch settings, and classification thresholds. TabNet
constructor parameters not overridden in the scripts use the defaults of the
installed `pytorch-tabnet` version. Each run writes a `run_config.json`
containing the resolved command-line settings and pipeline metadata; no separate
private hyperparameter configuration file is required. The repository contains
**no** patient data, database credentials, institutional schema definitions, or
patient-identifiable information.

---

## What is in this repository

| Stage | Script | Purpose |
|---|---|---|
| 1 | `01_01_sepsis_onset_auto_label.py` | Sepsis-3 labelling: suspected infection from culture/antibiotic order pairs, then organ dysfunction from SOFA |
| 2 | `01_02_infection_onset_auto_label.py` | Culture-confirmed infection labelling within the infection-confirmation window |
| 3 | `01_03_cohort_selection_flowchart.py` | Cohort exclusion criteria and the shared patient-level 80/20 train/test split |
| 4 | `01_04_feature_preprocessing.py` | Physiological clipping and the imputation rules of the online supplemental table |
| 5 | `01_05_build_feature_windows.py` | Construction of the 8-hour feature-window samples |
| — | `04_0*_without_balance_*.py` | Modelling track released here: TabNet training without class balancing (no PSM, no oversampling, no calibration) |

The released modelling track is provided in four variants covering the two
outcomes (sepsis, culture-confirmed infection) and the two validation settings
(internal only; reduced feature set with external MIMIC-IV validation).

Each `04_*` modelling script is a self-contained command-line program with
`--help` and writes a `run_config.json` recording the resolved command-line
settings and pipeline metadata. The repository scripts read their inputs either
from delimited files or from a local `.sql` query file executed against a
database named by the `DB_URL` environment variable. Institution-specific table
and column names therefore stay in local `.sql` files that are not part of this
repository; column names that differ from the canonical ones can be remapped at
run time with `--column-map RAW=CANONICAL`.

---

## What is not included

The following components are **not** published here:

- **The main model (`02_*`)** — the balanced, deployed dual-model configuration,
  together with its trained weights.
- **The probability-calibration track (`03_*`)** — the isotonic-regression
  calibration workflow and the calibration-cohort split.
- **The real-time dashboard and deployment stack** — the FastAPI/WebSocket
  service, container and reverse-proxy configuration, and the relational
  database layer described in the manuscript.
- **Institution-internal `.sql` query files**, schema definitions, table and
  column names.

These components are covered by a pending patent application (Taiwan invention
patent application no. 113134453, *Systems for Predicting Onset of Sepsis*), and
we are currently preparing an application to the Taiwan Food and Drug
Administration (TFDA) for approval of the model as software as a medical device.
We are therefore unable to release the final model, its trained weights, the
calibration workflow, or the deployment code at this time.

The methodological content of the withheld components is fully described in the
manuscript and its online supplemental material: the propensity-score matching
and Borderline-SMOTE balancing procedure, the isotonic-regression calibration,
the temporal prediction framework, the final feature sets, and the reported
performance and threshold analyses. The `04_*` scripts released here share the
same data pipeline, feature definitions, TabNet configuration and training
procedure, and differ from the main analysis by the absence of class balancing
and probability calibration.

Researchers with an academic interest in the withheld components are invited to
contact the first author (see [Contact](#contact)). Requests will be considered
on a case-by-case basis and may require a data-use or material transfer
agreement.

---

## Data availability statement

Data are available upon reasonable request. The patient-level datasets used in
this study are not publicly available because of privacy restrictions and
institutional data governance policies. Data requests may be directed to the
corresponding author; however, any data request will require independent
institutional review and approval and may not always be granted. Relevant
summary statistics are provided in the article and in online supplemental
file 1. The study code and downstream analytical workflows are publicly
available on GitHub
(<https://github.com/fjsun2269/sepsis_infection_prediction_code>). The external
validation dataset used in this study, the MIMIC-IV database, is publicly
available to qualified researchers who complete the required data use agreement
and training and can be accessed at <https://physionet.org/content/mimiciv/>.

Given the performance of the algorithm, we are currently in the process of
applying to the Taiwan Food and Drug Administration for approval of the model as
a software medical device, and are unable to share the final algorithm, its
trained parameters, the calibration workflow, or the deployment code.

---

## Requirements

Python 3.10 or later.

The minimum package versions declared in `requirements.txt` are:

```text
numpy>=1.21.0
pandas>=1.5.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
pytorch-tabnet>=4.0
torch>=2.0.0
scipy>=1.9.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.2.0
tqdm>=4.64.0
openpyxl>=3.1.0
pyarrow
sqlalchemy
oracledb
```

`pyarrow`, `sqlalchemy`, and `oracledb` are retained without minimum-version
constraints because they are optional input backends used only for Parquet,
SQL-based, or Oracle data sources, respectively.

Install with:

```bash
pip install -r requirements.txt
```

The `>=` specifiers define minimum supported versions rather than an exact
environment lock. For exact reproduction of a completed run, retain the generated
`run_config.json` together with the package versions recorded from the execution
environment.

---

## Running the pipeline

The five preparation scripts run in order; each consumes the previous stage's
output.

```bash
# 1. Sepsis-3 labelling
python 01_01_sepsis_onset_auto_label.py \
    --si-codes data/si_codes.csv \
    --orders   data/orders.csv \
    --sofa     data/sofa.csv \
    --outdir   output/step1

# 2. Culture-confirmed infection labelling
python 01_02_infection_onset_auto_label.py \
    --cohort   output/step1/sepsis_onset.csv \
    --cultures data/culture_report.csv \
    --outdir   output/step2

# 3. Cohort selection and the shared patient-level split
python 01_03_cohort_selection_flowchart.py \
    --cohort output/step2/infection_onset.csv \
    --outdir output/step3

# 4. Clipping and imputation
python 01_04_feature_preprocessing.py \
    --vital-input  data/vital_features.csv \
    --lab-input    data/lab_features.csv \
    --cohort-input output/step3/study_cohort.csv \
    --outdir       output/step4

# 5. 8-hour feature windows
#    Step 4 writes the vital and laboratory tables separately, both keyed on
#    adm_ICU_id with the timestamp in a column named `date`. Joining them onto a
#    single hourly grid is a local step; the merged table is what --input expects.
python 01_05_build_feature_windows.py \
    --input hourly_icu.csv \
    --time-col date \
    --task both
```

Modelling then runs on the engineered feature tables:

```bash
# Released modelling track, internal sepsis
python 04_01_without_balance_internal_sepsis.py \
    --train-input output/train_features.csv \
    --test-input  output/test_features.csv \
    --outdir      outputs/no_balance_internal_sepsis
```

The remaining three variants (internal infection; reduced-feature sepsis and
infection with external MIMIC-IV validation) follow the same call signature.

`--self-test` is available on `01_05_build_feature_windows.py` and runs the full
window-construction logic against synthetic data, with no institutional input
required.

---

## Reproducibility notes

Several methodological choices are exposed as command-line options rather than
being fixed silently. The defaults below are the ones used for the published
analysis unless stated otherwise.

**Hyperparameter availability.** The `04_*` scripts publicly expose the
executable model-construction and training workflow used throughout the study.
Author-selected settings — including Adam with a learning rate of 0.001, the
StepLR scheduler, cross-validation and split settings, random seeds, maximum
epochs, early-stopping patience, batch and virtual-batch sizes, and the
classification threshold of 0.5 — are defined directly in the source code or
command-line defaults, and are the same settings used for the main analysis.
TabNet constructor parameters that are not explicitly overridden use the
defaults of the installed `pytorch-tabnet` version. Each run writes the resolved
command-line settings and pipeline metadata to `run_config.json`; optimizer and
scheduler values remain visible in the source code. No separate private
hyperparameter file is required. For exact reproduction of library-default TabNet
architecture parameters, dependency versions should be pinned in
`requirements.txt`.

**ICU episode definition.** An ICU discharge followed by readmission within
48 hours is treated as one episode (`--merge-stay-gap-hours`, default 48). The
interval is not standardised in the literature — MIMIC-III uses 24 hours,
MIMIC-IV declines to merge non-consecutive stays at all — so it is exposed for
sensitivity analysis.

**Two distinct 24 h / 72 h windows.** Step 1 pairs a culture with an antibiotic
(culture first → antibiotic within 72 h; antibiotic first → culture within 24 h).
Step 2 opens a window of −24 h to +72 h around the labelled sepsis onset and asks
whether any positive culture falls inside it. These are different rules with
different anchors; see the module docstrings before comparing against the
manuscript.

**Class balancing.** The main analysis applied 1:1 greedy nearest-neighbour
propensity-score matching on sex, age and Charlson Comorbidity Index
(calliper = 0.05), followed by Borderline-SMOTE, applied only to the internal
development cohort and never to the test or external cohorts; for the
cross-validation report, matching was performed inside each training fold so
that the validation fold retained the cohort's natural outcome prevalence. The
`04_*` scripts released here deliberately omit this step and train on the
unbalanced development cohort. The balancing implementation is part of the
withheld `02_*` track.

**Calibration.** The main analysis fitted isotonic regression on an independent
calibration cohort split at the patient level, as described in the manuscript
and online supplemental figures 6 and 7. The `04_*` scripts report uncalibrated
probabilities. The calibration implementation is part of the withheld `03_*`
track.

**APACHE II aggregation.** Aggregated as the maximum within the 8-hour window,
consistent with the main analysis, so that the released track differs from the
main analysis only by the absence of class balancing and calibration.
`--apache-aggregation first` is available as an explicit sensitivity option.

**Randomness.** All splits and resampling are seeded. Control index times in
step 1 are derived from the seed and the stay identifier, so adding or
reordering other stays does not change existing index times.

---

## Outputs and de-identification

No script writes patient-identifiable data. Prediction files contain a row
number, the true label and model probabilities only. Direct and quasi
identifiers — patient and encounter numbers, bed numbers, admission and discharge
timestamps, onset timestamps — are stripped before anything is written to disk.

Patient-level feature tables are written only when explicitly requested with
`--write-feature-tables`. They are analysis intermediates and must not be
committed to a public repository. The supplied `.gitignore` excludes them, along
with `sql/*.sql`, which holds institution-internal schema, table and column
names.

---

## License and permitted use

This repository is released under the **PolyForm Noncommercial License 1.0.0**.
The full text is in [`LICENSE`](LICENSE).

Any noncommercial purpose is permitted. Use by educational institutions, public
research organisations, health organisations and government institutions is
permitted regardless of funding source. Personal use for research, experiment
and testing for the benefit of public knowledge is permitted.

**Commercial use is not licensed.** Integration into proprietary or commercial
software, commercial deployment, and redistribution for commercial profit are
outside the scope of this license. The authors additionally reserve all patent
rights in the underlying method, including those claimed in Taiwan invention
patent application no. 113134453; the patent license granted by these terms
extends only to permitted noncommercial purposes.

For commercial licensing, patent enquiries, regulatory questions, or
institutional partnerships, please contact the first author before proceeding.

> Required Notice: Copyright 2026 Fang-Ju Sun and contributors

---

## Citation

If you use this code, please cite:

```bibtex
@article{sun2026sepsis,
  author  = {Sun, Fang-Ju and Liu, Yen-Yu and Kuo, Li-Kuo and Hu, Ting-Yu and Cheng, Kuang-Hua and Chen, Hung-Ting and Chang, Po-Jen and Wu, Min-Ching and Yeh, Hung-I and Wu, Kun-Pin},
  title   = {Real-time clinical decision support system for early identification of infection and sepsis in the intensive care unit: a retrospective development and prospective deployment study},
  journal = {BMJ Quality \& Safety},
  year    = {2026},
  doi     = {10.1136/bmjqs-2026-020104},
  url     = {https://doi.org/10.1136/bmjqs-2026-020104}
}
```

---

## Contact

**Fang-Ju Sun** — first author and repository contact
✉️ fjsun.b612@mmh.org.tw

Please use this address for requests concerning the withheld main model,
calibration workflow and dashboard, data access, commercial licensing, or patent
matters.
