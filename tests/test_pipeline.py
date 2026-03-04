from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neutcellselector.pipeline import (
    DatasetConfig,
    DegradeConfig,
    TrainConfig,
    build_feature_matrix,
    create_simulated_low_quality_neutrophils,
    load_count_matrix,
    load_single_cell_data,
    predict_low_quality_neutrophils,
    save_model,
    train_logistic_regression,
    load_model,
)


def test_end_to_end_training_and_prediction(tmp_path: Path) -> None:
    rows = []
    for i in range(40):
        rows.append(
            {
                "cell_id": f"n_{i}",
                "umi_count": 800 + i * 5,
                "cell_type": "neutrophil",
                "gene_a": 200 + i,
                "gene_b": 120 + i,
                "gene_c": 80 + i,
            }
        )
    for i in range(40):
        rows.append(
            {
                "cell_id": f"t_{i}",
                "umi_count": 500 + i * 2,
                "cell_type": "t_cell",
                "gene_a": 50 + i,
                "gene_b": 220 + i,
                "gene_c": 150 + i,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "mini.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_count_matrix(csv_path)
    merged = create_simulated_low_quality_neutrophils(
        loaded,
        DatasetConfig(),
        DegradeConfig(target_umi_upper=500, random_seed=7),
    )

    X, y = build_feature_matrix(merged, DatasetConfig())
    model, metrics = train_logistic_regression(X, y, TrainConfig(test_size=0.25, random_seed=7))
    assert metrics["roc_auc"] > 0.85

    model_path = tmp_path / "model.joblib"
    save_model(model, X.columns.tolist(), model_path)
    artifact = load_model(model_path)

    pred_df = predict_low_quality_neutrophils(artifact, merged, DatasetConfig(), threshold=0.5)
    assert {"lowq_neutrophil_probability", "predicted_lowq_neutrophil"}.issubset(pred_df.columns)


def test_load_h5ad_with_celltype_annotations(tmp_path: Path) -> None:
    ad = pytest.importorskip("anndata")

    x = pd.DataFrame(
        {
            "gene_a": [10, 5, 0],
            "gene_b": [3, 7, 2],
        }
    )
    obs = pd.DataFrame(
        {
            "cell_type": ["neutrophil", "t_cell", "neutrophil"],
        },
        index=["c1", "c2", "c3"],
    )
    adata = ad.AnnData(X=x.to_numpy(), obs=obs, var=pd.DataFrame(index=x.columns))
    h5ad_path = tmp_path / "mini.h5ad"
    adata.write_h5ad(h5ad_path)

    loaded = load_single_cell_data(h5ad_path, dataset_cfg=DatasetConfig())
    assert {"cell_id", "umi_count", "cell_type", "gene_a", "gene_b"}.issubset(loaded.columns)
    assert loaded["umi_count"].tolist() == [13, 12, 2]
