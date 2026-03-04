from __future__ import annotations

import argparse
import json
from pathlib import Path

from neutcellselector.pipeline import (
    DatasetConfig,
    DegradeConfig,
    TrainConfig,
    build_feature_matrix,
    create_simulated_low_quality_neutrophils,
    load_single_cell_data,
    save_model,
    train_logistic_regression,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train logistic regression model for low-quality neutrophil identification."
    )
    parser.add_argument("--input", required=True, help="Input CSV or H5AD file")
    parser.add_argument("--model-out", default="artifacts/logreg_lowq_neut.joblib", help="Output model path")
    parser.add_argument("--metrics-out", default="artifacts/train_metrics.json", help="Output metrics JSON")
    parser.add_argument("--cell-id-col", default="cell_id", help="Cell ID column name (obs col for H5AD)")
    parser.add_argument("--umi-col", default="umi_count", help="UMI count column name (obs col for H5AD)")
    parser.add_argument("--cell-type-col", default="cell_type", help="Cell type annotation column name")
    parser.add_argument("--neutrophil-label", default="neutrophil", help="Label value for neutrophils")
    parser.add_argument("--expression-layer", default=None, help="H5AD layer for expression matrix (default uses X)")
    parser.add_argument("--target-umi-upper", type=int, default=500)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_cfg = DatasetConfig(
        cell_id_col=args.cell_id_col,
        umi_col=args.umi_col,
        cell_type_col=args.cell_type_col,
        neutrophil_label=args.neutrophil_label,
    )
    degrade_cfg = DegradeConfig(target_umi_upper=args.target_umi_upper, random_seed=args.seed)
    train_cfg = TrainConfig(test_size=args.test_size, random_seed=args.seed)

    df = load_single_cell_data(args.input, dataset_cfg=data_cfg, expression_layer=args.expression_layer)
    train_df = create_simulated_low_quality_neutrophils(df, data_cfg, degrade_cfg)
    X, y = build_feature_matrix(train_df, data_cfg)

    model, metrics = train_logistic_regression(X, y, train_cfg)

    out_model = Path(args.model_out)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, X.columns.tolist(), out_model)

    out_metrics = Path(args.metrics_out)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Model saved to: {out_model}")
    print(f"Metrics saved to: {out_metrics}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
