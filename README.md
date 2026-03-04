# NeutCellSelector（Python + Logistic Regression）

这个项目提供一套可运行的 Python 机器学习环境，用于：

1. 读取高质量单细胞测序数据（包含中性粒细胞）。
2. 自动将高质量中性粒细胞做 UMI 下采样，生成“模拟低质量”中性粒细胞（UMI < 500）。
3. 训练 logistic regression，学习低质量中性粒细胞表达模式。
4. 对实际 UMI < 500（或任意细胞）进行低质量中性粒细胞识别。

## 1) 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

> 如果你只想安装依赖，也可以使用：`pip install -r requirements.txt`

## 2) 输入数据格式

支持两种输入：

1. **CSV**
   - 必须包含：`cell_id`、`umi_count`、`cell_type`
   - 其余列视为基因表达计数（整数）

2. **H5AD / H5data（推荐用于你的场景）**
   - 表达矩阵来自 `adata.X`（或你通过 `--expression-layer` 指定的 layer）
   - 细胞注释来自 `adata.obs`
   - 需要 `obs` 中有细胞类型注释列（默认列名 `cell_type`）
   - `cell_id` 默认使用 `obs_names`（如果你在 `obs` 中有单独列，也可通过参数指定）
   - `umi_count` 如果在 `obs` 中不存在，会自动按每个细胞的表达总和计算

## 3) 生成示例数据（可选）

```bash
python data/generate_demo_data.py
```

会生成：`data/demo_counts.csv`

## 4) 训练模型

```bash
neut-train \
  --input data/demo_counts.csv \
  --model-out artifacts/logreg_lowq_neut.joblib \
  --metrics-out artifacts/train_metrics.json \
  --target-umi-upper 500

# 或者直接输入 H5AD（含细胞类型注释）
neut-train \
  --input data/your_sc_data.h5ad \
  --cell-type-col cell_type \
  --neutrophil-label neutrophil \
  --expression-layer counts
```

输出：
- `artifacts/logreg_lowq_neut.joblib`
- `artifacts/train_metrics.json`

## 5) 预测

```bash
neut-predict \
  --input data/demo_counts.csv \
  --model artifacts/logreg_lowq_neut.joblib \
  --output artifacts/predictions.csv \
  --threshold 0.5

# H5AD 推理
neut-predict \
  --input data/your_sc_data.h5ad \
  --model artifacts/logreg_lowq_neut.joblib \
  --output artifacts/predictions.csv \
  --cell-type-col cell_type \
  --expression-layer counts
```

输出：`artifacts/predictions.csv`

包含列：
- `cell_id`
- `umi_count`
- `cell_type`
- `lowq_neutrophil_probability`
- `predicted_lowq_neutrophil`

## 6) 方法说明

- 使用高质量中性粒细胞（默认 `cell_type == neutrophil` 且 `umi_count > 500`）作为来源。
- 对每个来源细胞做随机 UMI 下采样，使其总 UMI 降到 `<500`，保留相对表达模式。
- 将这些模拟样本标记为正类（1），原始样本标记为负类（0）。
- 使用标准化 + Logistic Regression（`class_weight='balanced'`）进行训练。

## 7) 可调参数

训练脚本参数：
- `--cell-id-col`: 细胞 ID 列名（默认 `cell_id`）
- `--umi-col`: UMI 列名（默认 `umi_count`）
- `--cell-type-col`: 细胞类型注释列名（默认 `cell_type`）
- `--neutrophil-label`: 中性粒细胞标签值（默认 `neutrophil`）
- `--expression-layer`: H5AD layer 名称（默认 `None`，即 `adata.X`）
- `--target-umi-upper`: 低质量阈值（默认 500）
- `--test-size`: 测试集比例（默认 0.2）
- `--seed`: 随机种子（默认 42）

预测脚本参数：
- `--threshold`: 分类阈值（默认 0.5）
- 同样支持 `--cell-id-col / --umi-col / --cell-type-col / --expression-layer`

## 8) 注意事项（真实项目建议）

- 真实数据中建议先做严格 QC（双细胞、线粒体比例、批次校正等）。
- 可增加特征：线粒体比例、核糖体比例、基因复杂度（nFeature）。
- 若类别不平衡严重，可进一步做 sample weighting 或阈值调优。
- 可结合交叉验证与外部数据集验证泛化能力。
