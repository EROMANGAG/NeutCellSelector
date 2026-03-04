from __future__ import annotations

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    genes = [f"gene_{i:03d}" for i in range(60)]
    rows = []

    for i in range(1200):
        if i < 450:
            cell_type = "neutrophil"
            umi = int(rng.integers(900, 4500))
            alpha = np.concatenate([np.full(20, 2.5), np.full(40, 0.8)])
        elif i < 850:
            cell_type = "t_cell"
            umi = int(rng.integers(700, 3200))
            alpha = np.concatenate([np.full(20, 0.9), np.full(40, 2.0)])
        else:
            cell_type = "mono"
            umi = int(rng.integers(600, 3600))
            alpha = np.concatenate([np.full(30, 1.4), np.full(30, 1.2)])

        probs = rng.dirichlet(alpha)
        counts = rng.multinomial(umi, probs)
        record = {
            "cell_id": f"cell_{i:05d}",
            "umi_count": umi,
            "cell_type": cell_type,
        }
        record.update({g: int(c) for g, c in zip(genes, counts)})
        rows.append(record)

    df = pd.DataFrame(rows)
    df.to_csv("data/demo_counts.csv", index=False)
    print("Generated: data/demo_counts.csv", df.shape)


if __name__ == "__main__":
    main()
