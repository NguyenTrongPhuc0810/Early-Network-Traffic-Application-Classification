from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from act2_project.utils.io import write_json, write_text


def generate_classification_reports(
    y_true,
    y_pred,
    *,
    digits: int = 2,
) -> tuple[str, dict[str, Any]]:
    text_report = classification_report(y_true, y_pred, digits=digits, zero_division=0)
    json_report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        digits=digits,
        zero_division=0,
    )
    return text_report, json_report


def save_classification_reports(
    y_true,
    y_pred,
    *,
    text_path: Path,
    json_path: Path,
    digits: int = 2,
) -> tuple[str, dict[str, Any]]:
    text_report, json_report = generate_classification_reports(y_true, y_pred, digits=digits)
    write_text(text_path, text_report)
    write_json(json_path, json_report)
    return text_report, json_report


def save_confusion_matrix(
    y_true,
    y_pred,
    *,
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        ax=ax,
        xticks_rotation="vertical",
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
