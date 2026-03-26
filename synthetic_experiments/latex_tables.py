"""
LaTeX table generation for synthetic experiment results.
"""

import json
import glob
import pandas as pd


def print_latex_synthetic(seed_paths):
    """
    Generate LaTeX for main synthetic results table (Table 1).
    Methods ordered: Bonferroni, Mahalanobis, Density, Kernel.
    Bold on best (lowest) volume per model and alpha.
    """
    all_rows = []
    files = sorted(glob.glob(seed_paths))

    for file_path in files:
        seed = int(file_path.split("_")[-1].split(".")[0])
        with open(file_path, "r") as f:
            data = json.load(f)
        results = data["results"]
        for model, methods in results.items():
            for method, alphas in methods.items():
                for alpha, metrics in alphas.items():
                    all_rows.append({
                        "seed": seed,
                        "model": model,
                        "method": method,
                        "alpha": str(float(alpha))[:4],
                        "coverage": metrics["coverage"],
                        "volume": metrics["volume"],
                        "wsc": metrics["wsc"],
                    })

    df = pd.DataFrame(all_rows)
    stats = df.groupby(["model", "method", "alpha"]).agg(["mean", "std"]).sort_index()
    means = stats.xs("mean", axis=1, level=1)[["coverage", "volume", "wsc"]]
    stds = stats.xs("std", axis=1, level=1)[["coverage", "volume", "wsc"]]

    best_vol = means.groupby(["model", "alpha"])["volume"].idxmin()

    alphas = ["0.1", "0.05", "0.02", "0.01"]
    method_order = ["Bonferroni", "Mahal", "Density", "Kernel"]
    models = ["Linear", "NN"]

    def fmt(m, s):
        return f"{m:.5f}\\tiny{{$\\pm${s:.5f}}}"

    def fmt_bold(m, s):
        return f"\\textbf{{{m:.5f}\\tiny{{$\\pm${s:.5f}}}}}"

    lines = []
    for ai, alpha in enumerate(alphas):
        if ai > 0:
            lines.append("    \\midrule")
        first = True
        for method in method_order:
            cells = []
            for model in models:
                idx = (model, method, alpha)
                if idx not in means.index:
                    cells.extend(["---"] * 3)
                    continue
                m_c, m_v, m_w = means.loc[idx]
                s_c, s_v, s_w = stds.loc[idx]
                is_best = (best_vol.get((model, alpha)) == idx)
                cells.append(fmt(m_c, s_c))
                cells.append(fmt_bold(m_v, s_v) if is_best else fmt(m_v, s_v))
                cells.append(fmt(m_w, s_w))

            prefix = f"    \\multirow{{4}}{{*}}{{{alpha}}}" if first else "   "
            first = False
            method_str = f"\\textbf{{{method}}}" if method == "Kernel" else method
            lines.append(f"{prefix} & {method_str}  & {' & '.join(cells)} \\\\")

    body = "\n".join(lines)
    n_seeds = df["seed"].nunique()

    print(f"""\\begin{{table}}[t]
  \\centering
  \\resizebox{{\\textwidth}}{{!}}{{%
  \\begin{{tabular}}{{ll ccc ccc}}
    \\toprule
    & & \\multicolumn{{3}}{{c}}{{Linear}} & \\multicolumn{{3}}{{c}}{{NN}} \\\\
    \\cmidrule(lr){{3-5}} \\cmidrule(lr){{6-8}}
    $\\alpha$ & Method & Coverage & Volume & WSC & Coverage & Volume & WSC \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
  }}
  \\vspace{{0.5em}}
  \\caption{{Results on synthetic data across {n_seeds} seeds. All values reported as mean $\\pm$ std. Bold indicates best volume per model and $\\alpha$.}}
  \\label{{tab:main_results_C}}
  \\vspace{{-1em}}
\\end{{table}}""")


def print_latex_lengthscale(seed_paths):
    """
    Generate LaTeX for lengthscale ablation table (sidewaystable).
    Density vs Kernel across lengthscale values.
    """
    all_rows = []
    files = sorted(glob.glob(seed_paths))

    for file_path in files:
        seed = int(file_path.split("_")[-1].split(".")[0])
        with open(file_path, "r") as f:
            data = json.load(f)
        results = data["results"]
        for model, methods in results.items():
            for method, alphas in methods.items():
                if '_ls' in method:
                    base_method, ls_str = method.split('_ls')
                    lengthscale = float(ls_str)
                else:
                    continue
                for alpha, metrics in alphas.items():
                    all_rows.append({
                        "seed": seed,
                        "model": model,
                        "lengthscale": lengthscale,
                        "method": base_method,
                        "alpha": str(float(alpha))[:4],
                        "coverage": metrics["coverage"],
                        "volume": metrics["volume"],
                        "wsc": metrics["wsc"],
                    })

    df = pd.DataFrame(all_rows)
    stats = df.groupby(["model", "lengthscale", "method", "alpha"]).agg(["mean", "std"]).sort_index()
    means = stats.xs("mean", axis=1, level=1)[["coverage", "volume", "wsc"]]
    stds = stats.xs("std", axis=1, level=1)[["coverage", "volume", "wsc"]]

    best_vol = means.groupby(["model", "lengthscale", "alpha"])["volume"].idxmin()

    alphas = ["0.1", "0.05", "0.02", "0.01"]
    lengthscales = sorted(df["lengthscale"].unique())
    models = ["Linear", "NN"]
    method_order = ["Density", "Kernel"]

    def fmt(m, s):
        return f"{m:.5f}\\tiny{{$\\pm${s:.5f}}}"

    def fmt_bold(m, s):
        return f"\\textbf{{{m:.5f}\\tiny{{$\\pm${s:.5f}}}}}"

    lines = []
    for mi, model in enumerate(models):
        if mi > 0:
            lines.append("    \\midrule")
            lines.append("    \\midrule")
        for li, ls in enumerate(lengthscales):
            if li > 0:
                lines.append("    \\cmidrule(lr){2-15}")
            for method in method_order:
                cells = []
                for alpha in alphas:
                    idx = (model, ls, method, alpha)
                    if idx not in means.index:
                        cells.extend(["---"] * 3)
                        continue
                    m_c, m_v, m_w = means.loc[idx]
                    s_c, s_v, s_w = stds.loc[idx]
                    is_best = (best_vol.get((model, ls, alpha)) == idx)
                    cells.append(fmt(m_c, s_c))
                    cells.append(fmt_bold(m_v, s_v) if is_best else fmt(m_v, s_v))
                    cells.append(fmt(m_w, s_w))

                if method == method_order[0]:
                    if li == 0:
                        n_ls = len(lengthscales)
                        prefix = f"    \\multirow{{{n_ls * 2}}}{{*}}{{{model}}} & \\multirow{{2}}{{*}}{{{ls:.2f}}}"
                    else:
                        prefix = f"    & \\multirow{{2}}{{*}}{{{ls:.2f}}}"
                else:
                    prefix = f"    & "

                method_str = f"\\textbf{{{method}}}" if method == "Kernel" else method
                lines.append(f"{prefix} & {method_str} & {' & '.join(cells)} \\\\[4pt]")

    body = "\n".join(lines)
    n_seeds = df["seed"].nunique()

    print(f"""\\begin{{sidewaystable}}[p]
  \\centering
  \\resizebox{{\\textwidth}}{{!}}{{%
  \\begin{{tabular}}{{lll ccc ccc ccc ccc}}
    \\toprule
    & & & \\multicolumn{{3}}{{c}}{{$\\alpha=0.10$}} & \\multicolumn{{3}}{{c}}{{$\\alpha=0.05$}} & \\multicolumn{{3}}{{c}}{{$\\alpha=0.02$}} & \\multicolumn{{3}}{{c}}{{$\\alpha=0.01$}} \\\\
    \\cmidrule(lr){{4-6}} \\cmidrule(lr){{7-9}} \\cmidrule(lr){{10-12}} \\cmidrule(lr){{13-15}}
    Model & $\\ell$ & Method & Coverage & Volume & WSC & Coverage & Volume & WSC & Coverage & Volume & WSC & Coverage & Volume & WSC \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
  }}
  \\vspace{{0.5em}}
  \\caption{{Density vs Kernel ablation across lengthscale values on 2D synthetic data ({n_seeds} seeds). Bold indicates lowest volume.}}
  \\label{{tab:main_results_lengthscale}}
\\end{{sidewaystable}}""")
