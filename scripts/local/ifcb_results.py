'''
Computation of MAE and MRAE per method and hierarchy level

Input: 
    - Prediction files:
        - quantification_results_FG/ifcb_quantification_results_*.csv        
        - quantification_results_AC/ifcb_quantification_results_*.csv        
        - quantification_results_OC/ifcb_quantification_results_*.csv
        
    - True prevalences:
        - quantification_results_FG/test_prevalences.csv
        - quantification_results_AC/test_prevalences.csv
        - quantification_results_OC/test_prevalences.csv
Output: 
    - Summary metrics:
        - ifcb_quantification_results/summary_MAE.csv
        - ifcb_quantification_results/summary_MRAE.csv
        
    - Figures:
        - ifcb_quantification_results/boxplot_MAE_per_sample.png
        - ifcb_quantification_results/boxplot_MRAE_per_sample.png
'''
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# General configuration
levels = ["FG", "AC", "OC"]
methods_order = ["CC", "ACC", "PCC", "PACC", "EMQ", "DMy"]

# Output folder
output_dir = "ifcb_quantification_results"
os.makedirs(output_dir, exist_ok=True)

data_levels = {}

# Iterate over taxonomy levels
for level in levels:

    # Load predictions
    files = glob.glob(f"quantification_results_{level}/ifcb_quantification_results_*.csv")

    if len(files) == 0:
        print(f"WARNING: No files found for level {level}")
        continue

    df_pred = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df_pred["method"] = df_pred["method"].replace({"DMy_b4": "DMy"})

    # Load true prevalences
    df_true = pd.read_csv(f"quantification_results_{level}/test_prevalences.csv")

    df_true_long = df_true.melt(
        id_vars="sample",
        var_name="class",
        value_name="y_true"
    )

    # Merge predictions with true prevalences
    df = df_pred.merge(
        df_true_long,
        on=["sample", "class"],
        how="inner"
    )

    if df.empty:
        print(f"WARNING: Empty merge for level {level}")
        continue

    # Compute MAE and MRAE
    epsilon = 1e-3

    df["abs_error"] = np.abs(df["y_pred"] - df["y_true"])
    df["rel_error"] = df["abs_error"] / np.maximum(df["y_true"], epsilon)

    data_levels[level] = df

# ========================= SAMPLE-LEVEL METRICS ====================================
def compute_sample_metrics(df):
    return (
        df.groupby(["method", "sample"])
        .agg(
            MAE=("abs_error", "mean"),
            MRAE=("rel_error", "mean")
        )
        .reset_index()
    )

def format_mean_std(mean, std, decimals=4):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

# ========================= SUMMARY METRICS ====================================
# Summary tables
mae_table = pd.DataFrame({"method": methods_order})
mrae_table = pd.DataFrame({"method": methods_order})   

mae_mean_std_table = pd.DataFrame({"method": methods_order})
mrae_mean_std_table = pd.DataFrame({"method": methods_order})

sample_metrics_levels = {}

for level in levels:

    if level not in data_levels:
        continue

    df = data_levels[level]

    # Compute MAE and MRAE at sample level
    sample_metrics = compute_sample_metrics(df)
    sample_metrics["level"] = level
    sample_metrics_levels[level] = sample_metrics

    # Compute mean and standard deviation across samples
    summary = (
        sample_metrics
        .groupby("method")
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            MRAE_mean=("MRAE", "mean"),
            MRAE_std=("MRAE", "std")
        )
        .reindex(methods_order)
    )

    # Numeric mean tables for figures
    mae_table[level] = summary["MAE_mean"].values
    mrae_table[level] = summary["MRAE_mean"].values

    # Formatted mean ± std tables
    mae_mean_std_table[level] = [
        format_mean_std(mean, std, decimals=4)
        for mean, std in zip(summary["MAE_mean"], summary["MAE_std"])
    ]

    mrae_mean_std_table[level] = [
        format_mean_std(mean, std, decimals=4)
        for mean, std in zip(summary["MRAE_mean"], summary["MRAE_std"])
    ]

# Save formatted tables
mae_mean_std_table.to_csv(
    os.path.join(output_dir, "summary_MAE.csv"),
    index=False
)

mrae_mean_std_table.to_csv(
    os.path.join(output_dir, "summary_MRAE.csv"),
    index=False
)

print("\nSaved MAE table:")
print(mae_mean_std_table)

print("\nSaved MRAE table:")
print(mrae_mean_std_table)

# ========================= SAMPLE-LEVEL BOXPLOTS ====================================
# Colors per taxonomy level
color_map = {
    "FG": "#e3984d",
    "AC": "#a1d99b",
    "OC": "#80b1d3"
}

# Labels
legend_labels = {
    "FG": "FunctionalGroup",
    "AC": "AutoClass",
    "OC": "OriginalClass"
}

def plot_boxplots_per_level(metric_name, output_path):

    # Create one subplot for each taxonomy level
    fig, axes = plt.subplots(1, len(levels), figsize=(15, 5), sharey=False)

    if len(levels) == 1:
        axes = [axes]

    # Iterate over taxonomy levels
    for i, level in enumerate(levels):

        ax = axes[i]

        if level not in sample_metrics_levels:
            ax.set_visible(False)
            continue

        sample_metrics = sample_metrics_levels[level]

        data_to_plot = []
        method_labels = []

        # Collect metric values for each quantification method
        for method in methods_order:

            values = sample_metrics.loc[
                sample_metrics["method"] == method,
                metric_name
            ].values

            if len(values) > 0:
                data_to_plot.append(values)
                method_labels.append(method)

        # Boxplots
        box = ax.boxplot(
            data_to_plot,
            labels=method_labels,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1)
        )

        # Apply colors according to taxonomy level
        for patch in box["boxes"]:
            patch.set_facecolor(color_map[level])
            patch.set_alpha(0.9)

        ax.set_title(legend_labels[level])
        ax.set_xlabel("Método")
        ax.grid(axis="y", alpha=0.3)

        if metric_name == "MRAE":
            ax.set_ylim(0, 2)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# Generate boxplots
plot_boxplots_per_level(
    metric_name="MAE",
    output_path="ifcb_quantification_results/boxplot_MAE_per_sample.png"
)

plot_boxplots_per_level(
    metric_name="MRAE",
    output_path="ifcb_quantification_results/boxplot_MRAE_per_sample.png"
)

print("\nBoxplots saved.")
