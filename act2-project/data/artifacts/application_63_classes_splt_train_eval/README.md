# SPLT 63-Class Model Artifacts

This directory documents the trained Random Forest run for the 63-class SPLT application classifier.

Tracked files:

- `classification_report.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `run_metadata.json`

Not tracked:

- `model.joblib`

The trained model binary is intentionally excluded from Git because it is a large generated artifact. Store it in external object storage such as Google Drive, S3, GitHub Releases, or an internal artifact registry, then restore it locally with the repository-level `setup_data.py` workflow.
