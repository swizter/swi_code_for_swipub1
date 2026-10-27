# Data acquisition and verification

The repository expects the processed HRV feature tables distributed in the Kaggle dataset [Biometrics for stress monitoring](https://www.kaggle.com/datasets/qiriro/stress). These files contain derived HRV features rather than raw ECG recordings.

| Dataset | Path inside the downloaded archive | Local filename | Rows | SHA-256 |
|---|---|---|---:|---|
| SWELL-KW | `dataset/2. final/datasets/hrv/swell/combined/regression/combined-swell-regression-hrv-dataset.csv` | `data/swell/all_subject.csv` | 391,638 | `eea0f994dd12d5422ad14259d46a3758c6ced6e0fe1b5392efcc162e10d5f887` |
| WESAD | `dataset/2. final/datasets/hrv/wesad/combined/classification/wesad-chest-combined-classification-hrv.csv` | `data/wesad/all_subjects.csv` | 135,650 | `b5a74442f5f6ba30a8842ab4e7c04049b2fb346acc07fd25dd90f4bab0bc7aa8` |

Place the files in the following locations before running either workflow:

```text
round1_loso/data/swell/all_subject.csv
round1_loso/data/wesad/all_subjects.csv
round2_diagnostics/data/swell/all_subject.csv
round2_diagnostics/data/wesad/all_subjects.csv
```

The loaders check the complete file hash, row count, expected participant identifiers, required feature columns, and condition labels. A checksum mismatch stops the run instead of silently accepting a different file.

SWELL-KW provides the 34 HRV features used in the three-class CNN analysis. The WESAD table provides 32 of those features and is used only for the binary calibrated-tree analyses described in the manuscript. The analysis code excludes identifier and condition fields from model inputs.
