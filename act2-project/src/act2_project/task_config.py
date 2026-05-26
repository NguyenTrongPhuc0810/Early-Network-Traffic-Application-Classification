from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from act2_project.paths import project_root, resolve_project_path
from act2_project.utils.io import read_yaml


@dataclass(frozen=True)
class ForegroundRule:
    category_names: tuple[str, ...]
    application_keywords: tuple[str, ...]


@dataclass(frozen=True)
class BackgroundRule:
    application_prefixes: tuple[str, ...]
    category_names: tuple[str, ...]


@dataclass(frozen=True)
class TaskDatasetRule:
    min_packets: int
    foreground_weight: float
    support_weight: float
    background_weight: float
    capture_total_weight: float
    vote_min_confidence: float
    vote_top_flows: int
    packet_weight_power: float
    auxiliary_weight_ratio: float
    auxiliary_purity_threshold: float


@dataclass(frozen=True)
class TaskConfig:
    label_priority: tuple[str, ...]
    label_patterns: dict[str, tuple[str, ...]]
    foreground: dict[str, ForegroundRule]
    background: BackgroundRule
    dataset: TaskDatasetRule

    @property
    def final_classes(self) -> tuple[str, ...]:
        return tuple(label for label in self.label_priority if label in self.label_patterns)


def load_task_config(task_config_path: Path | None = None) -> TaskConfig:
    root = project_root()
    config_path = resolve_project_path(task_config_path or "configs/task_labels.yaml", root)
    raw = read_yaml(config_path)

    task_labels = raw.get("task_labels", {})
    foreground = raw.get("foreground", {})
    background = raw.get("background", {})
    dataset = raw.get("dataset", {})

    return TaskConfig(
        label_priority=tuple(task_labels.get("label_priority", ())),
        label_patterns={
            str(label): tuple(patterns or ())
            for label, patterns in (task_labels.get("label_patterns", {}) or {}).items()
        },
        foreground={
            str(label): ForegroundRule(
                category_names=tuple((rule or {}).get("category_names", ())),
                application_keywords=tuple((rule or {}).get("application_keywords", ())),
            )
            for label, rule in (foreground or {}).items()
        },
        background=BackgroundRule(
            application_prefixes=tuple(background.get("application_prefixes", ())),
            category_names=tuple(background.get("category_names", ())),
        ),
        dataset=TaskDatasetRule(
            min_packets=int(dataset.get("min_packets", 5)),
            foreground_weight=float(dataset.get("foreground_weight", 1.0)),
            support_weight=float(dataset.get("support_weight", 0.25)),
            background_weight=float(dataset.get("background_weight", 0.0)),
            capture_total_weight=float(dataset.get("capture_total_weight", 100.0)),
            vote_min_confidence=float(dataset.get("vote_min_confidence", 0.45)),
            vote_top_flows=int(dataset.get("vote_top_flows", 40)),
            packet_weight_power=float(dataset.get("packet_weight_power", 0.5)),
            auxiliary_weight_ratio=float(dataset.get("auxiliary_weight_ratio", 0.5)),
            auxiliary_purity_threshold=float(dataset.get("auxiliary_purity_threshold", 0.98)),
        ),
    )
