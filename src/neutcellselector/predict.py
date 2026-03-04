from __future__ import annotations

import argparse
from pathlib import Path

from neutcellselector.pipeline import (
    DatasetConfig,
    load_model,
    load_single_cell_data,
    predict_low_quality_neutrophils,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict low-quality neutrophils (UMI<500 pattern) from count matrix."
    )
    parser.add_argument("--input", required=True, help="Input CSV or H5AD with same features as train")
    parser.add_argument("--model", required=True, help="Trained model artifact path")
    parser.add_argument("--output", default="artifacts/predictions.csv", help="Output CSV")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction probability threshold")
    parser.add_argument("--cell-id-col", default="cell_id", help="Cell ID column name")
    parser.add_argument("--umi-col", default="umi_count", help="UMI count column name")
    parser.add_argument("--cell-type-col", default="cell_type", help="Cell type annotation column name")
    parser.add_argument("--neutrophil-label", default="neutrophil", help="Label value for neutrophils")
    parser.add_argument("--expression-layer", default=None, help="H5AD layer for expression matrix (default uses X)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_cfg = DatasetConfig(
        cell_id_col=args.cell_id_col,
        umi_col=args.umi_col,
        cell_type_col=args.cell_type_col,
        neutrophil_label=args.neutrophil_label,
    )

    df = load_single_cell_data(args.input, dataset_cfg=data_cfg, expression_layer=args.expression_layer)
    artifact = load_model(args.model)
    result = predict_low_quality_neutrophils(
        model_artifact=artifact,
        df=df,
        dataset_cfg=data_cfg,
        threshold=args.threshold,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    print(f"Prediction saved to: {out}")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
