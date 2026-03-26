"""
LaTeX table generation for real data experiment results.
"""

import json
import glob
import pandas as pd


def print_latex_table(seed_paths, caption="", label=""):
    """
    Produce LaTeX table from seed JSONs for a single dataset/dimension.
    Methods ordered: Bonferroni, Mahalanobis, Density, Kernel.
    Bold on best (lowest) volume per model and alpha.

    :param seed_paths: glob pattern, e.g. "realdata_experiments/experiment_results/results_bio_2d_v3/*.json"
    :param caption: LaTeX table caption string
    :param label: LaTeX table label string
    """
    all_rows = []
    files = sorted(glob.glob(seed_paths))
    for file_path in files:
        with open(file_path, "r") as f:
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
                    "seed": seed,
                    "model": model,
                    "method": method,
                    "alpha": str(float(alpha))[:4],
                    "coverage": metrics["coverage"],
                    "volume": metrics["volume"],
                    "wsc": metrics["wsc"],
                })
    df_all = pd.DataFrame(all_rows)

    df_stats = df_all.groupby(["model", "method", "alpha"]).agg(["mean", "std"]).sort_index()
    metrics = ["coverage", "volume", "wsc"]
    means = df_stats.xs("mean", axis=1, level=1)[metrics]
    stds = df_stats.xs("std", axis=1, level=1)[metrics]

    best_vol = means.groupby(["model", "alpha"])["volume"].idxmin()

    alphas = ["0.1", "0.05", "0.02", "0.01"]
    method_order = ["Bonferroni", "Mahal", "Density", "Kernel"]
    models = ["Ridge", "MLP"]

    def fmt(m, s):
        return f"{m:.5f}\\tiny{{$\\pm${s:.5f}}}"

    def fmt_bold(m, s):
        return f"\\textbf{{{m:.5f}\\tiny{{$\\pm${s:.5f}}}}}"

    lines = []
    for ai, alpha in enumerate(alphas):
        if ai > 0:
            lines.append("    \\midrule")
        first_in_alpha = True
        for method in method_order:
            cells = []
            for model in models:
                idx = (model, method, alpha)
                if idx not in means.index:
                    cells.extend(["---", "---", "---"])
                    continue
                m_cov, m_vol, m_wsc = means.loc[idx]
                s_cov, s_vol, s_wsc = stds.loc[idx]
                is_best_vol = (best_vol[(model, alpha)] == idx)
                cells.append(fmt(m_cov, s_cov))
                cells.append(fmt_bold(m_vol, s_vol) if is_best_vol else fmt(m_vol, s_vol))
                cells.append(fmt(m_wsc, s_wsc))

            if first_in_alpha:
                prefix = f"    \\multirow{{4}}{{*}}{{{alpha}}}"
                first_in_alpha = False
            else:
                prefix = "   "

            method_str = f"\\textbf{{{method}}}" if method == "Kernel" else method
            lines.append(f"{prefix} & {method_str} & {' & '.join(cells)} \\\\")

    n_models = len(models)
    model_header = " & ".join([f"\\multicolumn{{3}}{{c}}{{{m}}}" for m in models])
    cmidrules = " ".join([f"\\cmidrule(lr){{{3 + i*3}-{5 + i*3}}}" for i in range(n_models)])
    metric_header = " & ".join(["Coverage & Volume & WSC"] * n_models)
    n_cols = 2 + 3 * n_models
    col_spec = "ll " + " ".join(["ccc"] * n_models)

    body = "\n".join(lines)

    print(f"""\\begin{{table}}[t]
  \\centering
  \\resizebox{{\\textwidth}}{{!}}{{%
  \\begin{{tabular}}{{{col_spec}}}
    \\toprule
    & & {model_header} \\\\
    {cmidrules}
    $\\alpha$ & Method & {metric_header} \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
  }}
  \\vspace{{0.5em}}
  \\caption{{{caption}}}
  \\label{{{label}}}
  \\vspace{{-1em}}
\\end{{table}}""")


def print_latex_real_data(paths, model="Ridge", datasets=None, dims=None):
    """
    Generate LaTeX for the combined real data summary table:
    volume ratio (Kernel/Mahalanobis) and Kernel empirical coverage.

    :param paths: list of 9 glob patterns in order bio2d, bio3d, bio4d, house2d, ..., blog4d
    :param model: which regression model to show (e.g. "Ridge")
    :param datasets: list of dataset labels (default: ["Bio"]*3 + ["House"]*3 + ["Blog"]*3)
    :param dims: list of dimensions (default: [2,3,4,2,3,4,2,3,4])
    """
    if datasets is None:
        datasets = ["Bio", "Bio", "Bio", "House", "House", "House", "Blog", "Blog", "Blog"]
    if dims is None:
        dims = [2, 3, 4, 2, 3, 4, 2, 3, 4]

    alphas = ["0.1", "0.05", "0.02", "0.01"]

    all_rows = []
    for path, dataset, d in zip(paths, datasets, dims):
        files = sorted(glob.glob(path))
        for file_path in files:
            with open(file_path, "r") as f:
                data = json.load(f)
            seed = data["seed"]
            mdl = data["model"]
            if mdl != model:
                continue
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
                        "method": method,
                        "alpha": str(float(alpha))[:4],
                        "coverage": metrics["coverage"],
                        "volume": metrics["volume"],
                    })

    df = pd.DataFrame(all_rows)

    kernel = df[df["method"] == "Kernel"][["dataset", "d", "seed", "alpha", "volume"]].rename(columns={"volume": "vol_k"})
    mahal = df[df["method"] == "Mahal"][["dataset", "d", "seed", "alpha", "volume"]].rename(columns={"volume": "vol_m"})
    merged = kernel.merge(mahal, on=["dataset", "d", "seed", "alpha"])
    merged["ratio"] = merged["vol_k"] / merged["vol_m"]

    ratio_stats = merged.groupby(["dataset", "d", "alpha"]).agg(
        ratio_mean=("ratio", "mean"),
        ratio_std=("ratio", "std"),
    ).reset_index()

    kcov = df[df["method"] == "Kernel"].groupby(["dataset", "d", "alpha"]).agg(
        cov_mean=("coverage", "mean"),
        cov_std=("coverage", "std"),
    ).reset_index()

    dataset_order = ["House", "Bio", "Blog"]

    lines = []
    for i, dataset in enumerate(dataset_order):
        if i > 0:
            lines.append("    \\midrule")
        for j, d in enumerate([2, 3, 4]):
            ratio_cells = []
            cov_cells = []
            for alpha in alphas:
                r = ratio_stats[(ratio_stats["dataset"] == dataset) & (ratio_stats["d"] == d) & (ratio_stats["alpha"] == alpha)]
                c = kcov[(kcov["dataset"] == dataset) & (kcov["d"] == d) & (kcov["alpha"] == alpha)]
                if len(r) == 1:
                    ratio_cells.append(f"{r['ratio_mean'].values[0]:.5f}\\tiny{{$\\pm${r['ratio_std'].values[0]:.3f}}}")
                else:
                    ratio_cells.append("---")
                if len(c) == 1:
                    cov_cells.append(f"{c['cov_mean'].values[0]:.5f}\\tiny{{$\\pm${c['cov_std'].values[0]:.5f}}}")
                else:
                    cov_cells.append("---")

            all_cells = ratio_cells + cov_cells
            prefix = f"    \\multirow{{3}}{{*}}{{{dataset}}}\n    & {d}" if j == 0 else f"    & {d}"
            lines.append(f"{prefix} & {' & '.join(all_cells)} \\\\")

    body = "\n".join(lines)
    n_seeds = df["seed"].nunique()

    print(f"""\\begin{{table}}[t]
  \\centering
  \\resizebox{{\\textwidth}}{{!}}{{%
  \\begin{{tabular}}{{ll cccc cccc}}
    \\toprule
    & & \\multicolumn{{4}}{{c}}{{Volume Ratio (Kernel\\,/\\,Mahalanobis)}} & \\multicolumn{{4}}{{c}}{{Kernel Coverage}} \\\\
    \\cmidrule(lr){{3-6}} \\cmidrule(lr){{7-10}}
    Dataset & $d$ & $1{{-}}\\alpha{{=}}.90$ & $.95$ & $.98$ & $.99$ & $1{{-}}\\alpha{{=}}.90$ & $.95$ & $.98$ & $.99$ \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
  }}
  \\vspace{{0.5em}}
  \\caption{{Real data results under {model} regression residuals, averaged over {n_seeds} seeds. Left: volume ratio of the kernel score to Mahalanobis (computed per seed, values below 1 indicate smaller kernel regions). Right: empirical coverage of the kernel score (target is $1{{-}}\\alpha$).}}
  \\label{{tab:real_data}}
  \\vspace{{-1em}}
\\end{{table}}""")
