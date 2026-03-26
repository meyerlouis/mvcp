import json
import glob
import pandas as pd
import numpy as np



def print_table_synthetic(seed_paths):
    all_rows = []

    # Adjust path pattern if needed
    files = sorted(glob.glob(seed_paths))

    for file_path in files:
        # extract seed number from filename
        seed = int(file_path.split("_")[-1].split(".")[0])

        with open(file_path, "r") as f:
            data = json.load(f)

        results = data["results"]

        for model, methods in results.items():
            for method, alphas in methods.items():
                for alpha, metrics in alphas.items():
                    row = {
                        "seed": seed,
                        "model": model,
                        "method": method,
                        "alpha": str(float(alpha))[:4],
                        **metrics
                    }
                    all_rows.append(row)

    df_all = pd.DataFrame(all_rows)

    df_stats = (
        df_all
        .groupby(["model", "method", "alpha"])
        .agg(["mean", "std"])
        .sort_index()
    )

    metrics = ["coverage", "volume", "wsc"]

    # Assume df_stats exists (multiindex columns: metrics x ["mean","std"])
    means = df_stats.xs("mean", axis=1, level=1)[metrics]
    stds  = df_stats.xs("std",  axis=1, level=1)[metrics]

    # Build wide tables
    means_wide = means.unstack("alpha").swaplevel(axis=1).sort_index(axis=1, level=[0,1])
    stds_wide  = stds.unstack("alpha").swaplevel(axis=1).sort_index(axis=1, level=[0,1])

    # Create formatted table: mean ± std strings
    formatted = means_wide.copy().astype(str)
    for col in formatted.columns:
        formatted[col] = [
            f"{m:.5f} ± {s:.5f}"
            for m, s in zip(means_wide[col], stds_wide[col])
        ]

    def highlight_best(data):
        # Same shape as data, start empty
        styles = pd.DataFrame("", index=data.index, columns=data.columns)

        for alpha in data.columns.levels[0]:
            sub = means_wide[alpha]  # numeric subtable for this alpha

            for model, group in sub.groupby(level="model"):
                # best coverage / lowest volume / best WSC
                # idx_cov = group["coverage"].idxmax()
                idx_vol = group["volume"].idxmin()
                idx_wsc = group["wsc"].idxmax()

                # apply style on the string table
                # styles.loc[idx_cov, (alpha, "coverage")] = "background-color: #d3d3d3"
                styles.loc[idx_vol, (alpha, "volume")]   = "background-color: #12b53e"
                styles.loc[idx_wsc, (alpha, "wsc")]      = "background-color: #12b53e"

        return styles

    def add_borders(df):
        """
        Add thicker separators:
        - Horizontal: between models
        - Vertical: only before each alpha block
        """
        styles = pd.DataFrame("", index=df.index, columns=df.columns)

        # ---- Horizontal thick line between models ----
        models = df.index.get_level_values("model")
        for i in range(1, len(df)):
            if models[i] != models[i-1]:
                styles.iloc[i, :] = "border-top: 3px solid black;"

        # ---- Vertical thick line before each alpha block ----
        alpha_levels = df.columns.levels[0]
        for alpha in alpha_levels:
            # get all positions of this alpha
            positions = [i for i, c in enumerate(df.columns) if c[0] == alpha]
            first_pos = positions[0]  # only first metric column
            # add left border for this column
            styles.iloc[:, first_pos] += "border-left: 3px solid black;"

        return styles

    def highlight_kernel_row(data):
        """
        Darker background for all Kernel rows, keeping existing cell highlights (like green).
        """
        styles = pd.DataFrame("", index=data.index, columns=data.columns)

        # Assuming 'method' is the second level of your index
        kernel_rows = data.index.get_level_values("method") == "Kernel"

        # Set subtle gray for the entire row
        styles.loc[kernel_rows, :] = "background-color: #bbedc5"  # light gray

        return styles


    styled = (
        formatted[['0.1', '0.05', '0.02', '0.01']]
        .style
        .apply(highlight_kernel_row, axis=None)  # base row shading for Kernel
        .apply(highlight_best, axis=None)        # green highlights for best volume/WSC
        .apply(add_borders, axis=None)           # thick separators
    )
    return styled


def print_table_real(seed_paths):

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

                row = {
                    "seed": seed,
                    "model": model,
                    "method": method,
                    "alpha": str(float(alpha))[:4],
                    "coverage": metrics["coverage"],
                    "volume": metrics["volume"],
                    "wsc": metrics["wsc"],
                }

                all_rows.append(row)

    df_all = pd.DataFrame(all_rows)

    df_stats = (
        df_all
        .groupby(["model", "method", "alpha"])
        .agg(["mean", "std"])
        .sort_index()
    )

    metrics = ["coverage", "volume", "wsc"]

    means = df_stats.xs("mean", axis=1, level=1)[metrics]
    stds  = df_stats.xs("std", axis=1, level=1)[metrics]

    means_wide = (
        means.unstack("alpha")
        .swaplevel(axis=1)
        .sort_index(axis=1, level=[0, 1])
    )

    stds_wide = (
        stds.unstack("alpha")
        .swaplevel(axis=1)
        .sort_index(axis=1, level=[0, 1])
    )

    formatted = means_wide.copy().astype(str)

    for col in formatted.columns:
        formatted[col] = [
            f"{m:.5f} ± {s:.5f}"
            for m, s in zip(means_wide[col], stds_wide[col])
        ]

    def highlight_best(data):

        styles = pd.DataFrame("", index=data.index, columns=data.columns)

        for alpha in data.columns.levels[0]:

            sub = means_wide[alpha]

            for model, group in sub.groupby(level="model"):

                idx_vol = group["volume"].idxmin()
                idx_wsc = group["wsc"].idxmax()

                styles.loc[idx_vol, (alpha, "volume")] = "background-color: #12b53e"
                styles.loc[idx_wsc, (alpha, "wsc")] = "background-color: #12b53e"

        return styles

    def add_borders(df):

        styles = pd.DataFrame("", index=df.index, columns=df.columns)

        models = df.index.get_level_values("model")

        for i in range(1, len(df)):
            if models[i] != models[i-1]:
                styles.iloc[i, :] = "border-top: 3px solid black;"

        alpha_levels = df.columns.levels[0]

        for alpha in alpha_levels:
            positions = [i for i, c in enumerate(df.columns) if c[0] == alpha]
            first_pos = positions[0]
            styles.iloc[:, first_pos] += "border-left: 3px solid black;"

        return styles

    def highlight_kernel_row(data):

        styles = pd.DataFrame("", index=data.index, columns=data.columns)

        kernel_rows = data.index.get_level_values("method") == "Kernel"

        styles.loc[kernel_rows, :] = "background-color: #bbedc5"

        return styles

    styled = (
        formatted[['0.1', '0.05', '0.02', '0.01']]
        .style
        .apply(highlight_kernel_row, axis=None)
        .apply(highlight_best, axis=None)
        .apply(add_borders, axis=None)
    )

    return styled


def print_table_lengthscale(seed_paths):
    all_rows = []

    # Adjust path pattern if needed
    files = sorted(glob.glob(seed_paths))

    if not files:
        print(f"No files found matching pattern: {seed_paths}")
        return None

    for file_path in files:
        # extract seed number from filename
        seed = int(file_path.split("_")[-1].split(".")[0])

        with open(file_path, "r") as f:
            data = json.load(f)

        results = data["results"]

        for model, methods in results.items():
            for method, alphas in methods.items():
                # Extract lengthscale from method name if present
                if '_ls' in method:
                    base_method, ls_str = method.split('_ls')
                    lengthscale = float(ls_str)
                else:
                    base_method = method
                    lengthscale = np.nan  # Use NaN for methods without lengthscale

                for alpha, metrics in alphas.items():
                    # Format alpha consistently
                    alpha_str = f"{float(alpha):.2f}"

                    row = {
                        "seed": seed,
                        "model": model,
                        "lengthscale": lengthscale,
                        "method": base_method,
                        "alpha": alpha_str,
                        "coverage": metrics.get('coverage', None),
                        "volume": metrics.get('volume', None),
                        "wsc": metrics.get('wsc', None),
                    }
                    all_rows.append(row)

    if not all_rows:
        print("No data found in files")
        return None

    df_all = pd.DataFrame(all_rows)

    # Group by model, lengthscale, method, alpha
    grouped = df_all.groupby(['model', 'lengthscale', 'method', 'alpha'])[['coverage', 'volume', 'wsc']].agg(['mean', 'std']).round(5)

    # Reset index to make it flat
    grouped = grouped.reset_index()

    # Flatten the column MultiIndex
    grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in grouped.columns.values]

    # Create the formatted table with MultiIndex rows
    alphas = sorted(grouped['alpha'].unique())

    # Build the formatted data
    formatted_data = []

    for (model, lengthscale, method), group in grouped.groupby(['model', 'lengthscale', 'method']):
        row = {
            'model': model,
            'lengthscale': lengthscale,
            'method': method
        }

        for alpha in alphas:
            alpha_data = group[group['alpha'] == alpha]
            if len(alpha_data) > 0:
                for metric in ['coverage', 'volume', 'wsc']:
                    mean_val = alpha_data[f'{metric}_mean'].values[0]
                    std_val = alpha_data[f'{metric}_std'].values[0]
                    row[(alpha, metric)] = f"{mean_val:.5f} ± {std_val:.5f}"
            else:
                for metric in ['coverage', 'volume', 'wsc']:
                    row[(alpha, metric)] = "N/A"

        formatted_data.append(row)

    # Create DataFrame
    formatted = pd.DataFrame(formatted_data)

    # Set MultiIndex for rows: model, lengthscale, method
    formatted = formatted.set_index(['model', 'lengthscale', 'method'])

    # Sort the index: model first, then lengthscale (with NaN first), then method
    formatted = formatted.sort_index(level=['model', 'lengthscale', 'method'],
                                    ascending=[True, True, True])

    # Create MultiIndex columns
    col_tuples = []
    for alpha in alphas:
        for metric in ['coverage', 'volume', 'wsc']:
            col_tuples.append((alpha, metric))

    formatted.columns = pd.MultiIndex.from_tuples(col_tuples, names=['alpha', 'metric'])

    def highlight_best(data):
        styles = pd.DataFrame("", index=data.index, columns=data.columns)

        # For each alpha
        for alpha in data.columns.levels[0]:
            # For each model AND lengthscale combination
            for (model, lengthscale) in data.index.droplevel('method').unique():
                # Skip NaN lengthscale (Mahal/Bonferroni) - they don't compete
                if pd.isna(lengthscale):
                    continue

                # Get all methods for this model and lengthscale
                mask = (data.index.get_level_values('model') == model) & \
                       (data.index.get_level_values('lengthscale') == lengthscale)
                group_data = data[mask]

                if len(group_data) == 0:
                    continue

                # Find best volume (lowest) and best wsc (highest) within this lengthscale group
                best_volume = float('inf')
                best_volume_idx = None
                best_wsc = -float('inf')
                best_wsc_idx = None

                for idx in group_data.index:
                    vol_str = data.loc[idx, (alpha, 'volume')]
                    wsc_str = data.loc[idx, (alpha, 'wsc')]

                    if vol_str != "N/A" and wsc_str != "N/A":
                        try:
                            vol_val = float(vol_str.split(' ± ')[0])
                            wsc_val = float(wsc_str.split(' ± ')[0])

                            if vol_val < best_volume:
                                best_volume = vol_val
                                best_volume_idx = idx

                            if wsc_val > best_wsc:
                                best_wsc = wsc_val
                                best_wsc_idx = idx
                        except (ValueError, IndexError):
                            continue

                if best_volume_idx:
                    styles.loc[best_volume_idx, (alpha, 'volume')] = "background-color: #12b53e"
                if best_wsc_idx:
                    styles.loc[best_wsc_idx, (alpha, 'wsc')] = "background-color: #12b53e"

        return styles

    def add_borders(df):
        """
        Add thicker separators:
        - Horizontal thick lines between models
        - Horizontal thinner lines between lengthscales within same model
        - Vertical lines before each alpha block
        """
        styles = pd.DataFrame("", index=df.index, columns=df.columns)

        # ---- Horizontal lines ----
        prev_model = None
        prev_lengthscale = None

        for i, (model, lengthscale, method) in enumerate(df.index):
            # Thick black line between different models
            if prev_model is not None and model != prev_model:
                styles.iloc[i, :] = "border-top: 3px solid black;"
            # Gray line between different lengthscales within same model
            elif prev_lengthscale is not None and lengthscale != prev_lengthscale and not (pd.isna(lengthscale) and pd.isna(prev_lengthscale)):
                styles.iloc[i, :] = "border-top: 2px solid #999;"

            prev_model = model
            prev_lengthscale = lengthscale

        # ---- Vertical thick line before each alpha block ----
        alpha_levels = df.columns.levels[0]
        for alpha in alpha_levels:
            # Find first column for this alpha
            for j, (alpha_col, metric) in enumerate(df.columns):
                if alpha_col == alpha:
                    styles.iloc[:, j] += "border-left: 3px solid black;"
                    break

        return styles

    def highlight_method_rows(data):
        """
        Color-code different method types:
        - Kernel: light green
        - Density: light blue
        - Others (Mahal, Bonferroni): no color
        """
        styles = pd.DataFrame("", index=data.index, columns=data.columns)

        for i, (model, lengthscale, method) in enumerate(data.index):
            if 'Kernel' in method:
                styles.iloc[i, :] = "background-color: #bbedc5"  # light green
            # elif 'Density' in method:
            #     styles.iloc[i, :] = "background-color: #c5e0ff"  # light blue

        return styles

    def format_lengthscale(val):
        """Format lengthscale for display"""
        if pd.isna(val):
            return "—"
        return f"{val:.1f}"

    # Apply formatting to the index for display
    formatted.index = pd.MultiIndex.from_tuples(
        [(model, format_lengthscale(ls), method)
         for model, ls, method in formatted.index.values],
        names=['model', 'lengthscale', 'method']
    )

    return formatted
