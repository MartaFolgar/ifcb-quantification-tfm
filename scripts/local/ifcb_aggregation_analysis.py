'''
Granularity analysis of MAE and MRAE across hierarchy levels

Input: 
    - IFCB.csv
    - Prediction files:
        - quantification_results_FG/ifcb_quantification_results_*.csv        
        - quantification_results_AC/ifcb_quantification_results_*.csv        
        - quantification_results_OC/ifcb_quantification_results_*.csv
        
    - True prevalences:
        - quantification_results_FG/test_prevalences.csv
        - quantification_results_AC/test_prevalences.csv
        - quantification_results_OC/test_prevalences.csv
Output: 
    - ifcb_quantification_results/MAE_granularity_comparison.csv
    - ifcb_quantification_results/MRAE_granularity_comparison.csv
    - ifcb_quantification_results/MAE_heatmap.png
'''

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Compute MAE and MRAE per Functional Group and hierarchy level
def compute_errors(pred, true, data, mode, epsilon=1e-3):

    mae_dict = {}
    mrae_dict = {}

    # Case 1: Functional Group level (no aggregation needed)
    if mode == "FG":

        pred_wide = pred.pivot_table(
            index="sample",
            columns="class",
            values="y_pred"
        ).sort_index()

        true = true.set_index("sample").sort_index()
        fgs = true.columns

        # Compute errors
        for fg in fgs:
            abs_err = np.abs(pred_wide[fg] - true[fg])
            mae_dict[fg] = np.mean(abs_err)
            mrae_dict[fg] = np.mean(abs_err / np.maximum(true[fg], epsilon)) 

    # Case 2: Higher hierarchy levels that require aggregation (AutoClass / OriginalClass)
    else:

        # Map AC/OCs to FG based on IFCB.csv
        col = "AutoClass" if mode == "AC" else "OriginalClass"
        class_to_fg = dict(zip(data[col], data["FunctionalGroup"]))

        # Group hierarchy classes by FG
        fg_groups = {}
        for cls in true.columns:
            fg = class_to_fg.get(cls)
            if fg is not None:
                fg_groups.setdefault(fg, []).append(cls)

        true = true.set_index("sample").sort_index()

        mae_vals = {fg: [] for fg in fg_groups}
        mrae_vals = {fg: [] for fg in fg_groups}

        # Compute errors per sample
        for sample in true.index:
            for fg, cls_list in fg_groups.items():

                y_true = true.loc[sample, cls_list].sum()

                y_pred = pred[
                    (pred["sample"] == sample) &
                    (pred["class"].isin(cls_list))
                ]["y_pred"].sum()

                abs_err = abs(y_true - y_pred)

                mae_vals[fg].append(abs_err)
                mrae_vals[fg].append(abs_err / max(y_true, epsilon))

        # Average errors per FG
        mae_dict = {fg: np.mean(v) for fg, v in mae_vals.items()}
        mrae_dict = {fg: np.mean(v) for fg, v in mrae_vals.items()}

    return mae_dict, mrae_dict


# Convert results to final table format
def build_matrix(results_dict, metric_name, method):

    df = pd.DataFrame(results_dict)
    df.index.name = "FG"

    df_pivot = df.T.reset_index()
    df_pivot.columns = ["Level"] + list(df.index)

    df_pivot["METHOD"] = method

    df_pivot[f"MEAN_{metric_name}"] = df_pivot.drop(columns=["Level", "METHOD"]).mean(axis=1)

    df_pivot = df_pivot[["Level", "METHOD"] + sorted(df.index) + [f"MEAN_{metric_name}"]]

    return df_pivot

# Quantification methods tested
methods = ["CC", "ACC", "PCC", "PACC", "EMQ", "DMy"]

# Hierarchy levels
levels = ["FG", "AC", "OC"]

# Output folder
output_dir = "ifcb_quantification_results"
os.makedirs(output_dir, exist_ok=True)

# Load initial dataset (needed for AutoClass/OriginalClass mapping to FunctionalGroup)
data = pd.read_csv("IFCB.csv")

all_mae = []
all_mrae = []

# Iterate over quantification methods
for method in methods:

    results_mae = {}
    results_mrae = {}

    # Iterate over taxonomy levels
    for level in levels:

        # Load predictions and true prevalences
        pred_path = f"quantification_results_{level}/ifcb_quantification_results_{method}.csv"
        true_path = f"quantification_results_{level}/test_prevalences.csv"

        pred = pd.read_csv(pred_path)
        true = pd.read_csv(true_path)

        # Compute MAE and MRAE per FG
        mae, mrae = compute_errors(pred, true, data if level != "FG" else None, level)

        results_mae[level] = mae
        results_mrae[level] = mrae

    # Build matrix per quantification method
    df_mae = build_matrix(results_mae, "MAE", method)
    df_mrae = build_matrix(results_mrae, "MRAE", method)

    all_mae.append(df_mae)
    all_mrae.append(df_mrae)

# Concatenate final results
final_mae = pd.concat(all_mae, ignore_index=True)
final_mrae = pd.concat(all_mrae, ignore_index=True)

print("\n=== MAE ===\n")
print(final_mae)

print("\n=== MRAE ===\n")
print(final_mrae)

# Save results
final_mae.to_csv(
    os.path.join(output_dir, "MAE_granularity_comparison.csv"),
    index=False,
    float_format="%.5f"
)

final_mrae.to_csv(
    os.path.join(output_dir, "MRAE_granularity_comparison.csv"),
    index=False,
    float_format="%.5f"
)

print("\nAll results saved.")

# -- Plot figure --
input_dir = "ifcb_quantification_results"

functional_groups = [
    "Ciliate",
    "Diatom",
    "Dinoflagellata",
    "Flagellate",
    "Other"
]

def plot_delta_vs_fg_heatmaps(df, metric_name, output_file):

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(13, 6.5),
        sharey=True,
        constrained_layout=True
    )

    method_matrices = {}

    # Compute delta matrices per method
    for method in methods:
        df_method = df[df["METHOD"] == method].copy()

        df_method["Level"] = pd.Categorical(
            df_method["Level"],
            categories=levels,
            ordered=True
        )
        df_method = df_method.sort_values("Level")

        fg_cols = [fg for fg in functional_groups if fg in df_method.columns]
        matrix = df_method.set_index("Level")[fg_cols]

        fg_baseline = matrix.loc["FG"]

        delta_matrix = matrix.subtract(fg_baseline, axis=1)
        delta_matrix = delta_matrix.drop(index="FG")

        method_matrices[method] = delta_matrix

    axes_flat = axes.flatten()

    # Plot heatmaps
    for idx, method in enumerate(methods):

        ax = axes_flat[idx]
        delta_matrix = method_matrices[method]

        vmax = np.nanmax(np.abs(delta_matrix.values))
        if vmax == 0 or np.isnan(vmax):
            vmax = 1e-12

        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.imshow(
            delta_matrix.values,
            cmap="coolwarm",
            aspect="auto",
            norm=norm
        )

        ax.set_title(method, fontsize=12)

        # X axis labels
        ax.set_xticks(np.arange(len(functional_groups)))

        if idx >= 3:  # solo la fila inferior
            ax.set_xticklabels(
                functional_groups,
                rotation=30,
                ha="right",
                fontsize=10
            )
            ax.tick_params(axis="x", bottom=True)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", bottom=False)

        # Y axis labels
        ax.set_yticks([0, 1])

        if idx % 3 == 0:  # columna izquierda
            ax.set_yticklabels(["AC → FG", "OC → FG"], fontsize=10)
            ax.tick_params(axis="y", left=True, labelleft=True)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

        # Grid styling
        ax.set_xticks(np.arange(-0.5, len(functional_groups), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)

        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)

    cbar.set_label(
        f"Δ{metric_name} respecto a FG",
        fontsize=10
    )

    # Save figure
    plt.savefig(os.path.join(input_dir, "MAE_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()


# MAE heatmaps
plot_delta_vs_fg_heatmaps(
    final_mae,
    "MAE",
    os.path.join(input_dir, "MAE_heatmap.png")
)

print("\nHeatmaps saved.")



'''
input_dir = "ifcb_quantification_results"
output_dir = os.path.join(input_dir, "Aggr")
os.makedirs(output_dir, exist_ok=True)

functional_groups = [
    "Ciliate",
    "Diatom",
    "Dinoflagellata",
    "Flagellate",
    "Other"
]


def plot_delta_vs_fg_heatmaps(df, metric_name, output_file):

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(13, 6.5),
        sharey=True,
        constrained_layout=True
    )

    method_matrices = {}

    # Compute delta matrices per method
    for method in methods:
        df_method = df[df["METHOD"] == method].copy()

        df_method["Level"] = pd.Categorical(
            df_method["Level"],
            categories=levels,
            ordered=True
        )
        df_method = df_method.sort_values("Level")

        fg_cols = [fg for fg in functional_groups if fg in df_method.columns]
        matrix = df_method.set_index("Level")[fg_cols]

        fg_baseline = matrix.loc["FG"]

        delta_matrix = matrix.subtract(fg_baseline, axis=1)
        delta_matrix = delta_matrix.drop(index="FG")

        method_matrices[method] = delta_matrix

    axes_flat = axes.flatten()

    # Plot heatmaps
    for idx, method in enumerate(methods):

        ax = axes_flat[idx]
        delta_matrix = method_matrices[method]

        vmax = np.nanmax(np.abs(delta_matrix.values))
        if vmax == 0 or np.isnan(vmax):
            vmax = 1e-12

        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.imshow(
            delta_matrix.values,
            cmap="coolwarm",
            aspect="auto",
            norm=norm
        )

        ax.set_title(method, fontsize=12)

        # X axis labels
        ax.set_xticks(np.arange(len(functional_groups)))

        if idx >= 3:  # solo la fila inferior
            ax.set_xticklabels(
                functional_groups,
                rotation=30,
                ha="right",
                fontsize=10
            )
            ax.tick_params(axis="x", bottom=True)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", bottom=False)

        # Y axis labels
        ax.set_yticks([0, 1])

        if idx % 3 == 0:  # columna izquierda
            ax.set_yticklabels(["AC → FG", "OC → FG"], fontsize=10)
            ax.tick_params(axis="y", left=True, labelleft=True)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

        # Grid styling
        ax.set_xticks(np.arange(-0.5, len(functional_groups), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)

        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)

    cbar.set_label(
        f"Δ{metric_name} respecto a FG",
        fontsize=10
    )

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


# MAE heatmaps
plot_delta_vs_fg_heatmaps(
    final_mae,
    "MAE",
    os.path.join(output_dir, "MAE_heatmap.png")
)

print("\nHeatmaps saved.")'''