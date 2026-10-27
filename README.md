# HRV stress detection reanalysis

This repository contains the code, fixed configurations, and compact result tables supporting the revised Comment on Mortensen et al., *Multi-Class Stress Detection Through Heart Rate Variability: A Deep Neural Network Based Study*.

The central analysis compares record-wise random-window evaluation with participant-wise leave-one-subject-out (LOSO) evaluation. The repository also contains the participant-information diagnostics and the secondary baseline-calibrated six-feature tree analyses reported in the manuscript and supplementary material.

## Repository structure

```text
.
├── README.md
├── DATA.md
├── requirements.txt
├── reported_results/
│   ├── Table_1_main_results.csv
│   ├── Table_S1_method_alignment.csv
│   ├── Table_S2_calibrated_results.csv
│   └── README.md
├── round1_loso/
│   ├── configs/full.json
│   ├── src/hrv_reanalysis/
│   ├── tests/test_protocol.py
│   └── README.md
└── round2_diagnostics/
    ├── configs/round2.json
    ├── src/hrv_reanalysis/
    ├── tests/test_followup.py
    └── README.md
```

- `round1_loso/` contains the participant-wise LOSO workflow, including the CNN evaluation and the within-dataset baseline-calibrated tree analyses.
- `round2_diagnostics/` contains the random-window CNN evaluation, random participant-category control, frozen-representation identity probe, and bidirectional cross-dataset calibrated-tree analyses.
- `reported_results/` contains the exact compact tables used in the main manuscript and supplementary material.

Only files needed to understand and reproduce the reported analyses are included. Training checkpoints, generated logs, temporary outputs, and unrelated exploratory analyses are not part of this repository.

## Data preparation

The analyses use processed HRV feature tables from the Kaggle dataset [Biometrics for stress monitoring](https://www.kaggle.com/datasets/qiriro/stress). The source CSV files are not stored in this repository because each exceeds GitHub's standard file-size limit and remains subject to the source dataset's distribution terms.

Download the two files and place a copy under both workflow directories:

```text
round1_loso/data/swell/all_subject.csv
round1_loso/data/wesad/all_subjects.csv
round2_diagnostics/data/swell/all_subject.csv
round2_diagnostics/data/wesad/all_subjects.csv
```

Exact archive paths, row counts, and SHA-256 checksums are provided in [DATA.md](DATA.md). Each workflow verifies the input checksum before a full run.

## Environment

The recorded Python dependencies are listed in each workflow directory and in the root `requirements.txt`. GPU CNN runs used PyTorch 2.7.1 with CUDA 12.8. The supplied `launch.py` scripts prepare the workflow environment and install the matching PyTorch build.

## Running the analyses

Run the small protocol check first:

```bash
cd round1_loso
python launch.py --smoke

cd ../round2_diagnostics
python launch.py --smoke
```

Run the full predefined workflows from their respective directories:

```bash
python launch.py
```

Full CNN training is computationally intensive and requires a compatible CUDA GPU. Outputs are written to the `results/` directory within each workflow. The workflow-specific README files describe the scope of each run.

## Principal reported results

- SWELL-KW three-class CNN: 99.75% subject-equal accuracy with a random-window split and 44.72% with LOSO.
- Random participant-category control: 99.49% accuracy.
- Frozen-representation identity probe: 99.36% accuracy for 22 participants.
- Baseline-calibrated six-feature tree, within-dataset LOSO: 85.47% on SWELL-KW and 90.84% on WESAD.
- Cross-dataset calibrated tree: 77.31% for SWELL-KW to WESAD and 62.15% for WESAD to SWELL-KW.

The CNN task is three-class and does not use target-user calibration. The tree task is binary and uses the evaluated participant's resting baseline for feature calibration while excluding that participant's stress-labelled observations from classifier training. The two analyses therefore represent different evaluation settings and are not a direct algorithm comparison.
