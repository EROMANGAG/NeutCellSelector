from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetConfig:
    cell_id_col: str = "cell_id"
    umi_col: str = "umi_count"
    cell_type_col: str = "cell_type"
    neutrophil_label: str = "neutrophil"


@dataclass
class DegradeConfig:
    target_umi_upper: int = 500
    random_seed: int = 42
    min_drop_fraction: float = 0.35
    max_drop_fraction: float = 0.9


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_seed: int = 42
    c: float = 1.0
    max_iter: int = 1000


REQUIRED_META_COLUMNS = ("cell_id", "umi_count", "cell_type")


def _extract_dense_matrix(matrix: object) -> np.ndarray:
    """Convert dense/sparse matrix-like object to 2D numpy array."""
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def _load_h5ad(path: str | Path, dataset_cfg: DatasetConfig, expression_layer: str | None = None) -> pd.DataFrame:
    """Load single-cell data from H5AD and return a flat DataFrame."""
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("Reading H5AD requires `anndata`. Please install project dependencies.") from exc

    adata = ad.read_h5ad(path)
    matrix = adata.layers[expression_layer] if expression_layer else adata.X
    expr = _extract_dense_matrix(matrix)
    if expr.ndim != 2:
        raise ValueError("Expression matrix must be 2D.")

    gene_cols = [str(g) for g in adata.var_names.tolist()]
    gene_df = pd.DataFrame(expr, columns=gene_cols)

    obs = adata.obs.copy()
    if dataset_cfg.cell_id_col not in obs.columns:
        obs[dataset_cfg.cell_id_col] = adata.obs_names.astype(str)

    if dataset_cfg.cell_type_col not in obs.columns:
        raise ValueError(
            f"Cell type annotation column '{dataset_cfg.cell_type_col}' not found in H5AD obs."
        )

    if dataset_cfg.umi_col not in obs.columns:
        obs[dataset_cfg.umi_col] = np.asarray(expr.sum(axis=1)).astype(int)

    meta = obs[[dataset_cfg.cell_id_col, dataset_cfg.umi_col, dataset_cfg.cell_type_col]].reset_index(drop=True)
    df = pd.concat([meta, gene_df], axis=1)
    return df


def load_count_matrix(path: str | Path) -> pd.DataFrame:
    """
    Load matrix from CSV.

    Expected format:
    - metadata columns: cell_id, umi_count, cell_type
    - remaining columns are gene expression counts (integer)
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_META_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def load_single_cell_data(
    path: str | Path,
    dataset_cfg: DatasetConfig,
    expression_layer: str | None = None,
) -> pd.DataFrame:
    """Load CSV or H5AD single-cell data into unified tabular format."""
    suffix = Path(path).suffix.lower()
    if suffix in {".h5ad", ".h5", ".h5data"}:
        df = _load_h5ad(path=path, dataset_cfg=dataset_cfg, expression_layer=expression_layer)
    else:
        df = load_count_matrix(path)

    missing = [
        c
        for c in (
            dataset_cfg.cell_id_col,
            dataset_cfg.umi_col,
            dataset_cfg.cell_type_col,
        )
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def _randomly_drop_umi(counts: np.ndarray, target_umi: int, rng: np.random.Generator) -> np.ndarray:
    """Downsample UMI counts to a target depth while preserving count distribution."""
    total = int(counts.sum())
    if total <= target_umi:
        return counts.copy()

    gene_idx = np.repeat(np.arange(counts.size), counts.astype(int))
    keep = rng.choice(gene_idx, size=target_umi, replace=False)
    return np.bincount(keep, minlength=counts.size)


def create_simulated_low_quality_neutrophils(
    df: pd.DataFrame,
    dataset_cfg: DatasetConfig,
    degrade_cfg: DegradeConfig,
) -> pd.DataFrame:
    """
    Build simulated low-quality neutrophils from high-quality neutrophils by UMI downsampling.

    Output contains same feature columns plus labels:
    - is_low_quality_neutrophil: 1 for simulated low-quality neutrophils
    - is_low_quality_neutrophil: 0 for all original cells
    """
    rng = np.random.default_rng(degrade_cfg.random_seed)

    gene_cols = [
        c
        for c in df.columns
        if c
        not in {
            dataset_cfg.cell_id_col,
            dataset_cfg.umi_col,
            dataset_cfg.cell_type_col,
        }
    ]
    if not gene_cols:
        raise ValueError("No gene feature columns found.")

    neut_mask = (
        (df[dataset_cfg.cell_type_col] == dataset_cfg.neutrophil_label)
        & (df[dataset_cfg.umi_col] > degrade_cfg.target_umi_upper)
    )

    highq_neutrophils = df.loc[neut_mask].copy()
    if highq_neutrophils.empty:
        raise ValueError("No high-quality neutrophils found for simulation.")

    simulated_records = []
    for _, row in highq_neutrophils.iterrows():
        counts = row[gene_cols].to_numpy(dtype=int)
        total_umi = int(row[dataset_cfg.umi_col])

        lower = max(int(total_umi * (1 - degrade_cfg.max_drop_fraction)), 50)
        upper = min(
            degrade_cfg.target_umi_upper - 1,
            int(total_umi * (1 - degrade_cfg.min_drop_fraction)),
        )
        if upper < lower:
            target_umi = min(degrade_cfg.target_umi_upper - 1, max(50, total_umi // 3))
        else:
            target_umi = int(rng.integers(lower, upper + 1))

        degraded = _randomly_drop_umi(counts, target_umi=target_umi, rng=rng)
        new_row = row.copy()
        new_row[gene_cols] = degraded
        new_row[dataset_cfg.umi_col] = int(degraded.sum())
        new_row[dataset_cfg.cell_id_col] = f"{row[dataset_cfg.cell_id_col]}_sim_lowq"
        simulated_records.append(new_row)

    simulated_df = pd.DataFrame(simulated_records)

    original = df.copy()
    original["is_low_quality_neutrophil"] = 0

    simulated_df["is_low_quality_neutrophil"] = 1

    merged = pd.concat([original, simulated_df], ignore_index=True)
    return merged


def build_feature_matrix(df: pd.DataFrame, dataset_cfg: DatasetConfig) -> tuple[pd.DataFrame, pd.Series]:
    gene_cols = [
        c
        for c in df.columns
        if c
        not in {
            dataset_cfg.cell_id_col,
            dataset_cfg.umi_col,
            dataset_cfg.cell_type_col,
            "is_low_quality_neutrophil",
        }
    ]

    X = df[gene_cols].copy()
    y = df["is_low_quality_neutrophil"].astype(int)

    # Add total UMI as explicit feature.
    X[dataset_cfg.umi_col] = df[dataset_cfg.umi_col].astype(float)

    return X, y


def train_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: TrainConfig,
) -> tuple[Pipeline, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_seed,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=cfg.c,
                    max_iter=cfg.max_iter,
                    class_weight="balanced",
                    random_state=cfg.random_seed,
                    n_jobs=None,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "classification_report": classification_report(y_test, pred, output_dict=True),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }
    return model, metrics


def save_model(model: Pipeline, feature_columns: Sequence[str], path: str | Path) -> None:
    artifact = {
        "model": model,
        "feature_columns": list(feature_columns),
    }
    joblib.dump(artifact, path)


def load_model(path: str | Path) -> dict:
    return joblib.load(path)


def predict_low_quality_neutrophils(
    model_artifact: dict,
    df: pd.DataFrame,
    dataset_cfg: DatasetConfig,
    threshold: float = 0.5,
) -> pd.DataFrame:
    model: Pipeline = model_artifact["model"]
    feature_columns: Iterable[str] = model_artifact["feature_columns"]

    features = df[list(feature_columns)].copy()
    proba = model.predict_proba(features)[:, 1]

    out = df[[dataset_cfg.cell_id_col, dataset_cfg.umi_col, dataset_cfg.cell_type_col]].copy()
    out["lowq_neutrophil_probability"] = proba
    out["predicted_lowq_neutrophil"] = (proba >= threshold).astype(int)
    return out
