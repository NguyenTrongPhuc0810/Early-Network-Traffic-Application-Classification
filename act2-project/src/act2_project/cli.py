from __future__ import annotations

import argparse
from pathlib import Path

from act2_project.config import load_app_config
from act2_project.pipeline.act2_prepare import (
    build_training_dataset,
    build_pcap_inference_dataset,
)
from act2_project.pipeline.cesnet_convert import convert_cesnet_to_act2_parquet
from act2_project.pipeline.parquet_eval import run_parquet_eval_pretrained
from act2_project.pipeline.per_label_eval import run_parquet_eval_pretrained_per_label
from act2_project.pipeline.pcap_ingest import run_pcap_ingest
from act2_project.pipeline.predict import run_pcap_predict, run_pcap_predict_pretrained
from act2_project.pipeline.task_dataset import build_auxiliary_task_dataset, build_task_dataset
from act2_project.pipeline.task_model import run_task_pcap_predict, run_task_train_eval
from act2_project.pipeline.task_taxonomy import build_task_taxonomy_dataset
from act2_project.pipeline.vnat_task_taxonomy import (
    build_combined_task_dataset,
    build_vnat_task_taxonomy_dataset,
)
from act2_project.pipeline.train import run_train_eval
from act2_project.task_config import load_task_config
from act2_project.utils.logging import configure_logging


def _absolute_cli_path(path: Path | None, default: Path | None = None) -> Path:
    if path is not None:
        return path.resolve()
    if default is None:
        raise ValueError("A path argument or default path is required.")
    return default


def _resolve_n_dissections(args) -> int | None:
    if getattr(args, "no_dpi", False):
        return 0
    return getattr(args, "n_dissections", None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone SPLT early-classification pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to configs/default.yaml",
    )
    parser.add_argument(
        "--category-config",
        "--core-category-config",
        "--class-subset-config",
        dest="category_config",
        type=Path,
        default=None,
        help="Optional path to configs/class_subset.yaml. Leave category_subset empty for all categories.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO)",
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=None,
        help="Optional path to configs/task_labels.yaml",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_eval = subparsers.add_parser(
        "train-eval",
        help="Train/evaluate the SPLT RandomForest on categories derived from df_final_model_data.",
    )
    train_eval.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Raw or prepared parquet. Defaults to 02-app-classification/data/data.parquet.",
    )
    train_eval.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for model, reports, and plots.",
    )
    train_eval.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Override the target column. Defaults to application_category_name.",
    )

    prepare_train_data = subparsers.add_parser(
        "prepare-train-data",
        help="Prepare data.parquet into df_final_model_data then keep only SPLT columns + labels.",
    )
    prepare_train_data.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Input raw parquet. Defaults to 02-app-classification/data/data.parquet.",
    )
    prepare_train_data.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet for the prepared training dataset.",
    )

    cesnet_convert = subparsers.add_parser(
        "convert-cesnet",
        help="Convert CESNET-QUIC22/CESNET-TLS-Year22 CSV or ZIP data into Act2 SPLT parquet.",
    )
    cesnet_convert.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CESNET .zip, .csv, or .csv.gz file.",
    )
    cesnet_convert.add_argument(
        "--dataset",
        choices=("quic", "tls"),
        required=True,
        help="CESNET dataset kind: quic for CESNET-QUIC22, tls for CESNET-TLS-Year22.",
    )
    cesnet_convert.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output Act2-compatible parquet.",
    )
    cesnet_convert.add_argument(
        "--width",
        type=int,
        default=25,
        help="Number of first packets to keep from PPI. Default: 25.",
    )
    cesnet_convert.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Rows per CSV chunk. Lower this if RAM is tight.",
    )
    cesnet_convert.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on output rows for a quick pilot conversion.",
    )
    cesnet_convert.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional balanced reservoir sample size per mapped class. Scans the full input.",
    )
    cesnet_convert.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for --max-per-class reservoir sampling.",
    )
    cesnet_convert.add_argument(
        "--keep-unmapped",
        action="store_true",
        help="Keep labels that cannot be mapped to the current 63-class taxonomy.",
    )

    parquet_eval = subparsers.add_parser(
        "eval-parquet-pretrained",
        help="Evaluate an Act2-compatible SPLT parquet using a pre-trained model bundle.",
    )
    parquet_eval.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Input parquet with splt_direction/splt_ps/splt_piat_ms and target labels.",
    )
    parquet_eval.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to model.joblib produced by train-eval.",
    )
    parquet_eval.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for predictions, report, confusion matrix, and metadata.",
    )
    parquet_eval.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Override target column. Defaults to target stored in model bundle.",
    )

    parquet_eval_split = subparsers.add_parser(
        "eval-parquet-pretrained-per-label",
        help=(
            "Evaluate an Act2-compatible labeled parquet one application/label at a time "
            "using a pre-trained model bundle. Writes one sub-folder per label."
        ),
    )
    parquet_eval_split.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Input parquet with splt_direction/splt_ps/splt_piat_ms and target labels.",
    )
    parquet_eval_split.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to model.joblib produced by train-eval.",
    )
    parquet_eval_split.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output base directory. Each label gets a subdirectory.",
    )
    parquet_eval_split.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Override target column. Defaults to target stored in model bundle.",
    )
    parquet_eval_split.add_argument(
        "--label",
        action="append",
        default=None,
        help=(
            "Optional label to evaluate. Can be repeated. "
            "If omitted, evaluates all labels present in the parquet."
        ),
    )
    parquet_eval_split.add_argument(
        "--max-per-label",
        type=int,
        default=None,
        help="Optional cap on rows per label (random sample).",
    )
    parquet_eval_split.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used with --max-per-label.",
    )
    parquet_eval_split.add_argument(
        "--save-subset",
        action="store_true",
        help="Also write eval_subset.parquet for each label.",
    )

    pcap_to_flows = subparsers.add_parser(
        "pcap-to-flows",
        help="Extract SPLT-oriented NFStream flows from a PCAP.",
    )
    pcap_to_flows.add_argument("--pcap", type=Path, required=True, help="Input .pcap/.pcapng file")
    pcap_to_flows.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet for extracted flows.",
    )
    pcap_to_flows.add_argument(
        "--n-dissections",
        type=int,
        default=None,
        help="Override NFStream n_dissections. Use 0 to disable DPI labels.",
    )
    pcap_to_flows.add_argument(
        "--no-dpi",
        action="store_true",
        help="Shortcut for --n-dissections 0.",
    )

    prepare_pcap_data = subparsers.add_parser(
        "prepare-pcap-data",
        help="Prepare an NFStream flows parquet for SPLT category inference.",
    )
    prepare_pcap_data.add_argument(
        "--flows",
        type=Path,
        required=True,
        help="Input flows parquet generated by pcap-to-flows.",
    )
    prepare_pcap_data.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet for the prepared inference dataset.",
    )

    pcap_predict = subparsers.add_parser(
        "pcap-predict",
        help="Fit on the SPLT training dataset, then predict categories on a PCAP.",
    )
    pcap_predict.add_argument("--pcap", type=Path, required=True, help="Input .pcap/.pcapng file")
    pcap_predict.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help="Training parquet. Defaults to 02-app-classification/data/data.parquet.",
    )
    pcap_predict.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Base directory for interim, processed, and artifact outputs.",
    )
    pcap_predict.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Override the target column. Defaults to application_category_name.",
    )
    pcap_predict.add_argument(
        "--n-dissections",
        type=int,
        default=None,
        help="Override NFStream n_dissections. Use 0 to disable DPI labels during inference.",
    )
    pcap_predict.add_argument(
        "--no-dpi",
        action="store_true",
        help="Shortcut for --n-dissections 0.",
    )

    pcap_predict_pretrained = subparsers.add_parser(
        "pcap-predict-pretrained",
        help="Predict categories on a PCAP using a pre-trained model bundle.",
    )
    pcap_predict_pretrained.add_argument("--pcap", type=Path, required=True, help="Input .pcap/.pcapng file")
    pcap_predict_pretrained.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a previously trained model.joblib bundle.",
    )
    pcap_predict_pretrained.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Base directory for interim, processed, and artifact outputs.",
    )
    pcap_predict_pretrained.add_argument(
        "--n-dissections",
        type=int,
        default=None,
        help="Override NFStream n_dissections. Use 0 to disable DPI labels during inference.",
    )
    pcap_predict_pretrained.add_argument(
        "--no-dpi",
        action="store_true",
        help="Shortcut for --n-dissections 0.",
    )
    pcap_predict_pretrained.add_argument(
        "--include-flow-metadata",
        action="store_true",
        help="Also store src/dst ip:port and protocol in interim flows.parquet for traceability.",
    )

    build_task_data = subparsers.add_parser(
        "build-task-dataset",
        help="Build a task-labeled SPLT dataset from a folder of labeled PCAP files.",
    )
    build_task_data.add_argument(
        "--pcap-dir",
        type=Path,
        required=True,
        help="Folder containing labeled .pcap/.pcapng files.",
    )
    build_task_data.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet for the task-labeled dataset.",
    )
    build_task_data.add_argument(
        "--n-dissections",
        type=int,
        default=None,
        help="Override NFStream n_dissections while building the task dataset.",
    )

    build_aux_task = subparsers.add_parser(
        "build-aux-task-data",
        help="Create my_df_act3_splt.parquet-style auxiliary SPLT data from data.parquet.",
    )
    build_aux_task.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Raw parquet. Defaults to 02-app-classification/data/data.parquet.",
    )
    build_aux_task.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet for the auxiliary task dataset.",
    )

    build_task_taxonomy_data = subparsers.add_parser(
        "build-task-taxonomy-data",
        help="Remap application_name into a clean task_label taxonomy for RF training.",
    )
    build_task_taxonomy_data.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Prepared SPLT parquet. Defaults to data/processed/df_final_model_data_splt_real.parquet.",
    )
    build_task_taxonomy_data.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet for the remapped task taxonomy dataset.",
    )
    build_task_taxonomy_data.add_argument(
        "--taxonomy-config",
        type=Path,
        default=None,
        help="Optional path to configs/task_taxonomy.yaml",
    )

    build_vnat_task_taxonomy_data = subparsers.add_parser(
        "build-vnat-task-taxonomy-data",
        help="Convert VNAT H5 into the shared task_label taxonomy schema.",
    )
    build_vnat_task_taxonomy_data.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to VNAT_Dataframe_release_1.h5",
    )
    build_vnat_task_taxonomy_data.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet for the VNAT remapped dataset.",
    )
    build_vnat_task_taxonomy_data.add_argument(
        "--taxonomy-config",
        type=Path,
        default=None,
        help="Optional path to configs/vnat_task_taxonomy.yaml",
    )

    build_combined_task_data = subparsers.add_parser(
        "build-combined-task-data",
        help="Combine Act2 task dataset and VNAT task dataset into one training parquet.",
    )
    build_combined_task_data.add_argument(
        "--act2-data",
        type=Path,
        required=True,
        help="Path to Act2 remapped task parquet.",
    )
    build_combined_task_data.add_argument(
        "--vnat-data",
        type=Path,
        required=True,
        help="Path to VNAT remapped task parquet.",
    )
    build_combined_task_data.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet for the combined task dataset.",
    )

    task_train_eval = subparsers.add_parser(
        "task-train-eval",
        help="Train/evaluate the task-level early classifier with capture-level voting.",
    )
    task_train_eval.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Task-labeled parquet produced by build-task-dataset.",
    )
    task_train_eval.add_argument(
        "--aux-data",
        type=Path,
        default=None,
        help="Optional auxiliary task parquet such as my_df_act3_splt.parquet.",
    )
    task_train_eval.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for task train/eval artifacts.",
    )

    task_pcap_predict = subparsers.add_parser(
        "task-pcap-predict",
        help="Train the task-level classifier then predict the task for one PCAP.",
    )
    task_pcap_predict.add_argument("--pcap", type=Path, required=True, help="Input .pcap/.pcapng file")
    task_pcap_predict.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Task-labeled parquet produced by build-task-dataset.",
    )
    task_pcap_predict.add_argument(
        "--aux-data",
        type=Path,
        default=None,
        help="Optional auxiliary task parquet such as my_df_act3_splt.parquet.",
    )
    task_pcap_predict.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for task prediction artifacts.",
    )
    task_pcap_predict.add_argument(
        "--n-dissections",
        type=int,
        default=None,
        help="Override NFStream n_dissections during prediction.",
    )
    task_pcap_predict.add_argument(
        "--no-dpi",
        action="store_true",
        help="Shortcut for --n-dissections 0.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    app_config = load_app_config(
        config_path=args.config,
        class_subset_path=args.category_config,
    )
    task_config = load_task_config(args.task_config)

    if args.command == "train-eval":
        data_path = _absolute_cli_path(args.data_path, app_config.dataset.train_data_path)
        out_dir = _absolute_cli_path(
            args.out_dir,
            app_config.paths.artifacts_dir / "df_final_model_train_eval",
        )
        run_train_eval(
            data_path=data_path,
            out_dir=out_dir,
            app_config=app_config,
            target_column=args.target_column,
        )
        return 0

    if args.command == "prepare-train-data":
        data_path = _absolute_cli_path(args.data_path, app_config.dataset.train_data_path)
        out_path = _absolute_cli_path(
            args.out,
            app_config.paths.processed_dir / "df_final_model_data_splt.parquet",
        )
        build_training_dataset(
            data_path=data_path,
            out_path=out_path,
            app_config=app_config,
        )
        return 0

    if args.command == "convert-cesnet":
        metadata = convert_cesnet_to_act2_parquet(
            input_path=_absolute_cli_path(args.input),
            out_path=_absolute_cli_path(args.out),
            dataset=args.dataset,
            width=args.width,
            chunksize=args.chunksize,
            max_rows=args.max_rows,
            max_per_class=args.max_per_class,
            random_seed=args.random_seed,
            keep_unmapped=args.keep_unmapped,
        )
        print(
            "Converted CESNET rows: "
            f"{metadata['rows_written']} written, "
            f"{metadata['rows_seen']} seen, "
            f"{metadata['rows_unmapped']} unmapped. "
            f"Metadata: {Path(metadata['out_path']).with_suffix('.metadata.json')}"
        )
        return 0

    if args.command == "eval-parquet-pretrained":
        run_parquet_eval_pretrained(
            data_path=_absolute_cli_path(args.data_path),
            model_path=_absolute_cli_path(args.model_path),
            out_dir=_absolute_cli_path(args.out_dir),
            app_config=app_config,
            target_column=args.target_column,
        )
        return 0

    if args.command == "eval-parquet-pretrained-per-label":
        run_parquet_eval_pretrained_per_label(
            data_path=_absolute_cli_path(args.data_path),
            model_path=_absolute_cli_path(args.model_path),
            out_dir=_absolute_cli_path(args.out_dir),
            app_config=app_config,
            target_column=args.target_column,
            labels=args.label,
            max_per_label=args.max_per_label,
            random_seed=args.random_seed,
            save_subset=bool(args.save_subset),
        )
        return 0

    if args.command == "pcap-to-flows":
        pcap_path = _absolute_cli_path(args.pcap)
        out_path = _absolute_cli_path(
            args.out,
            app_config.paths.interim_dir / f"{pcap_path.stem}_flows.parquet",
        )
        run_pcap_ingest(
            pcap_path=pcap_path,
            out_path=out_path,
            app_config=app_config,
            n_dissections_override=_resolve_n_dissections(args),
        )
        return 0

    if args.command == "prepare-pcap-data":
        flows_path = _absolute_cli_path(args.flows)
        out_path = _absolute_cli_path(
            args.out,
            app_config.paths.processed_dir / f"{flows_path.stem}_pcap_inference_splt.parquet",
        )
        build_pcap_inference_dataset(
            flows_path=flows_path,
            out_path=out_path,
            app_config=app_config,
        )
        return 0

    if args.command == "pcap-predict":
        pcap_path = _absolute_cli_path(args.pcap)
        train_data_path = _absolute_cli_path(args.train_data, app_config.dataset.train_data_path)
        out_dir = _absolute_cli_path(
            args.out_dir,
            app_config.paths.artifacts_dir / f"pcap_predict_{pcap_path.stem}",
        )
        run_pcap_predict(
            pcap_path=pcap_path,
            train_data_path=train_data_path,
            out_dir=out_dir,
            app_config=app_config,
            target_column=args.target_column,
            n_dissections_override=_resolve_n_dissections(args),
        )
        return 0

    if args.command == "pcap-predict-pretrained":
        pcap_path = _absolute_cli_path(args.pcap)
        model_path = _absolute_cli_path(args.model_path)
        out_dir = _absolute_cli_path(
            args.out_dir,
            app_config.paths.artifacts_dir / f"pcap_predict_pretrained_{pcap_path.stem}",
        )
        run_pcap_predict_pretrained(
            pcap_path=pcap_path,
            model_path=model_path,
            out_dir=out_dir,
            app_config=app_config,
            n_dissections_override=_resolve_n_dissections(args),
            ingest_extra_columns=(
                ("src_ip", "src_port", "dst_ip", "dst_port", "protocol")
                if args.include_flow_metadata
                else ()
            ),
        )
        return 0

    if args.command == "build-task-dataset":
        pcap_dir = _absolute_cli_path(args.pcap_dir)
        out_path = _absolute_cli_path(
            args.out,
            app_config.paths.processed_dir / "task_labeled_splt.parquet",
        )
        build_task_dataset(
            pcap_dir=pcap_dir,
            out_path=out_path,
            app_config=app_config,
            task_config=task_config,
            n_dissections_override=args.n_dissections,
        )
        return 0

    if args.command == "build-aux-task-data":
        data_path = _absolute_cli_path(args.data_path, app_config.dataset.train_data_path)
        out_path = _absolute_cli_path(
            args.out,
            app_config.paths.processed_dir / "my_df_act3_splt.parquet",
        )
        build_auxiliary_task_dataset(
            raw_data_path=data_path,
            out_path=out_path,
            app_config=app_config,
            task_config=task_config,
        )
        return 0

    if args.command == "build-task-taxonomy-data":
        data_path = _absolute_cli_path(
            args.data_path,
            app_config.paths.processed_dir / "df_final_model_data_splt_real.parquet",
        )
        out_path = _absolute_cli_path(
            args.out,
            app_config.paths.processed_dir / "task_taxonomy_act2.parquet",
        )
        build_task_taxonomy_dataset(
            data_path=data_path,
            out_path=out_path,
            taxonomy_config_path=args.taxonomy_config,
        )
        return 0

    if args.command == "build-vnat-task-taxonomy-data":
        data_path = _absolute_cli_path(args.data_path)
        out_path = _absolute_cli_path(
            args.out,
            app_config.paths.processed_dir / "task_taxonomy_vnat.parquet",
        )
        build_vnat_task_taxonomy_dataset(
            data_path=data_path,
            out_path=out_path,
            taxonomy_config_path=args.taxonomy_config,
        )
        return 0

    if args.command == "build-combined-task-data":
        act2_path = _absolute_cli_path(args.act2_data)
        vnat_path = _absolute_cli_path(args.vnat_data)
        out_path = _absolute_cli_path(
            args.out,
            app_config.paths.processed_dir / "task_taxonomy_combined.parquet",
        )
        build_combined_task_dataset(
            act2_path=act2_path,
            vnat_path=vnat_path,
            out_path=out_path,
        )
        return 0

    if args.command == "task-train-eval":
        dataset_path = _absolute_cli_path(args.dataset)
        aux_data_path = _absolute_cli_path(args.aux_data) if args.aux_data is not None else None
        out_dir = _absolute_cli_path(
            args.out_dir,
            app_config.paths.artifacts_dir / "task_train_eval",
        )
        run_task_train_eval(
            task_data_path=dataset_path,
            out_dir=out_dir,
            app_config=app_config,
            task_config=task_config,
            auxiliary_data_path=aux_data_path,
        )
        return 0

    if args.command == "task-pcap-predict":
        pcap_path = _absolute_cli_path(args.pcap)
        dataset_path = _absolute_cli_path(args.dataset)
        aux_data_path = _absolute_cli_path(args.aux_data) if args.aux_data is not None else None
        out_dir = _absolute_cli_path(
            args.out_dir,
            app_config.paths.artifacts_dir / f"task_predict_{pcap_path.stem}",
        )
        run_task_pcap_predict(
            pcap_path=pcap_path,
            task_data_path=dataset_path,
            out_dir=out_dir,
            app_config=app_config,
            task_config=task_config,
            auxiliary_data_path=aux_data_path,
            n_dissections_override=_resolve_n_dissections(args),
        )
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
