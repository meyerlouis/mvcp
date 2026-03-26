import json
import glob
import numpy as np
import pandas as pd

# ── CONFIG ──
EXPERIMENTS = {
    "House": {
        2: "realdata_experiments/experiment_results/results_house_2d_v3/*.json",
        3: "realdata_experiments/experiment_results/results_house_3d_v3/*.json",
        4: "realdata_experiments/experiment_results/results_house_4d_v3/*.json",
    },
    "Bio": {
        2: "realdata_experiments/experiment_results/results_bio_2d_v3/*.json",
        3: "realdata_experiments/experiment_results/results_bio_3d_v3/*.json",
        4: "realdata_experiments/experiment_results/results_bio_4d_v3/*.json",
    },
    "Blog": {
        2: "realdata_experiments/experiment_results/results_blog_data_2d_v3/*.json",
        3: "realdata_experiments/experiment_results/results_blog_data_3d_v3/*.json",
        4: "realdata_experiments/experiment_results/results_blog_data_4d_v3_mcfixed/*.json",
    },
}

ALPHAS_DISPLAY = ["0.1", "0.05", "0.02", "0.01"]


def build_df(experiments):
    all_rows = []
    for dataset, dims in experiments.items():
        for d, pattern in dims.items():
            for path in sorted(glob.glob(pattern)):
                with open(path) as f:
                    data = json.load(f)
                seed = data["seed"]
                model = data["model"]
                results = data["result"]
                for method, method_data in results.items():
                    if method in ["seed", "model", "model_rmse", "meta"]:
                        continue
                    for alpha, metrics in method_data.items():
                        if alpha == "params":
                            continue
                        all_rows.append({
                            "dataset": dataset,
                            "d": d,
                            "seed": seed,
                            "model": model,
                            "method": method,
                            "alpha": str(float(alpha))[:4],
                            "coverage": metrics["coverage"],
                            "volume": metrics["volume"],
                            "wsc": metrics["wsc"],
                        })
    return pd.DataFrame(all_rows)


# ──────────────────────────────────────────────────────
# STYLED TABLE — your original format, uses std
# ──────────────────────────────────────────────────────
def styled_table(df, dataset, d=None, model=None):
    """
    Original-style styled table for notebook display.
    Uses std (not SE). Green highlights on best vol/WSC.

    Usage:
        styled_table(df, "Bio", d=2, model="MLP")
        styled_table(df, "House", model="MLP")
    """
    sub = df[df["dataset"] == dataset].copy()
    if d is not None:
        sub = sub[sub["d"] == d]
    if model is not None:
        sub = sub[sub["model"] == model]

    idx_cols = []
    if d is None:
        idx_cols.append("d")
    if model is None:
        idx_cols.append("model")
    idx_cols += ["method", "alpha"]

    df_stats = sub.groupby(idx_cols).agg(["mean", "std"]).sort_index()
    metrics = ["coverage", "volume", "wsc"]
    means = df_stats.xs("mean", axis=1, level=1)[metrics]
    stds  = df_stats.xs("std", axis=1, level=1)[metrics]

    means_wide = means.unstack("alpha").swaplevel(axis=1).sort_index(axis=1, level=[0, 1])
    stds_wide  = stds.unstack("alpha").swaplevel(axis=1).sort_index(axis=1, level=[0, 1])

    formatted = means_wide.copy().astype(str)
    for col in formatted.columns:
        formatted[col] = [
            f"{m:.5f} \u00b1 {s:.5f}"
            for m, s in zip(means_wide[col], stds_wide[col])
        ]

    if model is None and d is None:
        group_level = ["d", "model"]
    elif model is None:
        group_level = "model"
    elif d is None:
        group_level = "d"
    else:
        group_level = None

    def highlight_best(data):
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        for alpha in [a for a in ALPHAS_DISPLAY if a in data.columns.get_level_values(0)]:
            sub_m = means_wide[alpha]
            if group_level is not None:
                groups = sub_m.groupby(level=group_level)
            else:
                groups = [("all", sub_m)]
            for _, group in groups:
                styles.loc[group["volume"].idxmin(), (alpha, "volume")] = "background-color: #12b53e"
                styles.loc[group["wsc"].idxmax(), (alpha, "wsc")] = "background-color: #12b53e"
        return styles

    def highlight_kernel_row(data):
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        styles.loc[data.index.get_level_values("method") == "Kernel", :] = "background-color: #bbedc5"
        return styles

    def add_borders(data):
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        for level_name in ["model", "d"]:
            if level_name in data.index.names:
                vals = data.index.get_level_values(level_name)
                for i in range(1, len(data)):
                    if vals[i] != vals[i-1]:
                        styles.iloc[i, :] = "border-top: 3px solid black;"
        for alpha in ALPHAS_DISPLAY:
            positions = [i for i, c in enumerate(data.columns) if c[0] == alpha]
            if positions:
                styles.iloc[:, positions[0]] += "border-left: 3px solid black;"
        return styles

    avail = [a for a in ALPHAS_DISPLAY if a in formatted.columns.get_level_values(0)]
    styled = (
        formatted[avail]
        .style
        .apply(highlight_kernel_row, axis=None)
        .apply(highlight_best, axis=None)
        .apply(add_borders, axis=None)
    )
    title = dataset
    if d is not None: title += f" (d={d})"
    if model is not None: title += f" \u2014 {model}"
    styled.set_caption(title)
    return styled


# ──────────────────────────────────────────────
# VOLUME RATIOS (per seed, report mean ± std)
# ──────────────────────────────────────────────
def compute_volume_ratios(df, model="MLP", numerator="Kernel", denominator="Mahal"):
    sub = df[df["model"] == model]
    num = sub[sub["method"] == numerator][["dataset", "d", "seed", "alpha", "volume"]].rename(columns={"volume": "vol_num"})
    den = sub[sub["method"] == denominator][["dataset", "d", "seed", "alpha", "volume"]].rename(columns={"volume": "vol_den"})
    merged = num.merge(den, on=["dataset", "d", "seed", "alpha"])
    merged["ratio"] = merged["vol_num"] / merged["vol_den"]
    return merged.groupby(["dataset", "d", "alpha"]).agg(
        ratio_mean=("ratio", "mean"),
        ratio_std=("ratio", "std"),
        n_seeds=("ratio", "count"),
    ).reset_index()


def compute_kernel_coverage(df, model="MLP", method="Kernel"):
    sub = df[(df["model"] == model) & (df["method"] == method)]
    return sub.groupby(["dataset", "d", "alpha"]).agg(
        cov_mean=("coverage", "mean"),
        cov_std=("coverage", "std"),
    ).reset_index()
