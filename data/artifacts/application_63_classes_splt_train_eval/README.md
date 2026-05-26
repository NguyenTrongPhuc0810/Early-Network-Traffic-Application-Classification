# SPLT 63-Class Reference Artifacts

These files document the reference Random Forest run for the bundled 63-class
SPLT dataset.

Tracked:

- `classification_report.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `run_metadata.json`

Not tracked:

- `model.joblib`

The trained model is about 4 GB. Recreate it locally with:

```bash
python -m ml_pipeline.train --data data/raw/final_dataset_63_classes_splt.parquet --out-dir data/artifacts/application_63_classes_splt_train_eval
```
