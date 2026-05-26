"""Unified command line interface for the SPLT classification pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml_pipeline.evaluate import evaluate_model_bundle
from ml_pipeline.nfstream_ingest import NFStreamConfig, extract_pcap_to_parquet
from ml_pipeline.predict import predict_parquet
from ml_pipeline.prepare import prepare_splt_dataset
from ml_pipeline.train import TrainingConfig, train_random_forest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SPLT 63-class network traffic classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("extract-pcap", help="PCAP -> NFStream flows parquet.")
    ingest.add_argument("--pcap", required=True, type=Path)
    ingest.add_argument("--out", required=True, type=Path)
    ingest.add_argument("--n-dissections", type=int, default=20)
    ingest.add_argument("--splt-analysis", type=int, default=25)
    ingest.add_argument("--statistical-analysis", action=argparse.BooleanOptionalAction, default=False)

    prepare = subparsers.add_parser("prepare", help="Raw flows parquet -> curated SPLT parquet.")
    prepare.add_argument("--input", required=True, type=Path)
    prepare.add_argument("--out", required=True, type=Path)
    prepare.add_argument("--target-column", default="application_name")
    prepare.add_argument("--min-packets", type=int, default=10)
    prepare.add_argument("--min-samples-per-application", type=int, default=1000)
    prepare.add_argument("--feature-width", type=int, default=25)

    train = subparsers.add_parser("train", help="Train/evaluate Random Forest SPLT model.")
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--out-dir", required=True, type=Path)
    train.add_argument("--target-column", default="application_name")
    train.add_argument("--n-estimators", type=int, default=100)
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument("--n-jobs", type=int, default=-1)
    train.add_argument("--class-weight", default="balanced")
    train.add_argument("--feature-width", type=int, default=25)
    train.add_argument("--min-packets", type=int, default=10)
    train.add_argument("--min-samples-per-application", type=int, default=1000)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a trained bundle on labeled parquet.")
    evaluate.add_argument("--model", required=True, type=Path)
    evaluate.add_argument("--data", required=True, type=Path)
    evaluate.add_argument("--out-dir", required=True, type=Path)
    evaluate.add_argument("--target-column", default="application_name")
    evaluate.add_argument("--feature-width", type=int, default=None)

    predict = subparsers.add_parser("predict", help="Predict labels for an SPLT parquet.")
    predict.add_argument("--model", required=True, type=Path)
    predict.add_argument("--data", required=True, type=Path)
    predict.add_argument("--out", required=True, type=Path)
    predict.add_argument("--target-column", default=None)
    predict.add_argument("--min-packets", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "extract-pcap":
        extract_pcap_to_parquet(
            pcap_path=args.pcap,
            out_path=args.out,
            config=NFStreamConfig(
                n_dissections=args.n_dissections,
                splt_analysis=args.splt_analysis,
                statistical_analysis=args.statistical_analysis,
            ),
        )
        return 0

    if args.command == "prepare":
        from ml_pipeline.data_loader import DatasetConfig

        prepare_splt_dataset(
            input_path=args.input,
            output_path=args.out,
            config=DatasetConfig(
                target_column=args.target_column,
                min_packets=args.min_packets,
                min_samples_per_application=args.min_samples_per_application,
                feature_width=args.feature_width,
            ),
        )
        return 0

    if args.command == "train":
        class_weight = None if args.class_weight.lower() == "none" else args.class_weight
        train_random_forest(
            data_path=args.data,
            out_dir=args.out_dir,
            config=TrainingConfig(
                n_estimators=args.n_estimators,
                test_size=args.test_size,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                class_weight=class_weight,
                target_column=args.target_column,
                feature_width=args.feature_width,
                min_packets=args.min_packets,
                min_samples_per_application=args.min_samples_per_application,
            ),
        )
        return 0

    if args.command == "evaluate":
        evaluate_model_bundle(
            model_path=args.model,
            data_path=args.data,
            out_dir=args.out_dir,
            target_column=args.target_column,
            feature_width=args.feature_width,
        )
        return 0

    if args.command == "predict":
        predict_parquet(
            model_path=args.model,
            data_path=args.data,
            out_path=args.out,
            target_column=args.target_column,
            min_packets=args.min_packets,
        )
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
