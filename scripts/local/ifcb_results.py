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
        - ifcb_quantification_results/MAE_all_levels.png
        - ifcb_quantification_results/MRAE_all_levels.png
        - ifcb_quantification_results/MAE_PACC_EMQ.png
        - ifcb_quantification_results/MRAE_PACC_EMQ.png
'''

import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# General settings
levels = ["FG", "AC", "OC"] # Hierarchy levels to evaluate
methods_order = ["CC", "ACC", "PCC", "PACC", "EMQ"]  # Order

# Output folder
os.makedirs("ifcb_quantification_results", exist_ok=True)

data_levels = {}

# Loop over each hierarchy level
for level in levels:

    # Load predictions
    files = glob.glob(f"quantification_results_{level}/ifcb_quantification_results_*.csv")

    if len(files) == 0:
        print(f"WARNING: No files found for level {level}")
        continue

    df_pred = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Load true prevalences
    df_true = pd.read_csv(f"quantification_results_{level}/test_prevalences.csv")

    df_true_long = df_true.melt(
        id_vars="sample",
        var_name="class",
        value_name="y_true"
    )

    # Merge both predictions and prevalences
    df = df_pred.merge(
        df_true_long,
        on=["sample", "class"],
        how="inner"
    )

    if df.empty:
        print(f"WARNING: Empty merge for level {level}")
        continue

    # Compute MAE and MRAE metrics
    epsilon = 1e-3

    df["abs_error"] = np.abs(df["y_pred"] - df["y_true"])

    df["rel_error"] = (
        df["abs_error"] /
        np.maximum(df["y_true"], epsilon)
    )

    data_levels[level] = df


# ========================= SUMMARY METRICS ====================================
# Summary metrics tableS
mae_table = pd.DataFrame()
mrae_table = pd.DataFrame()

for level in levels:

    if level not in data_levels:
        continue

    df = data_levels[level]

    summary = df.groupby("method").agg(
        MAE=("abs_error", "mean"),
        MRAE=("rel_error", "mean")
    )

    mae_table[level] = summary["MAE"]
    mrae_table[level] = summary["MRAE"]

# Final formatting
mae_table = mae_table.reindex(methods_order).reset_index().round(6)
mrae_table = mrae_table.reindex(methods_order).reset_index().round(6)

# Save results
mae_table.to_csv("ifcb_quantification_results/summary_MAE.csv", index=False)
mrae_table.to_csv("ifcb_quantification_results/summary_MRAE.csv", index=False)

print("\nSaved MAE:")
print(mae_table)

print("\nSaved MRAE:")
print(mrae_table)


# ========================= SUMMARY FIGURES ====================================
def summary_metrics(df):
    return df.groupby(["method", "sample"]).agg(
        MAE=("abs_error", "mean"),
        MRAE=("rel_error", "mean")
    ).reset_index()

# --- Line plots of MAE and MRAE per sample for all quantification methods across hierarchy levels ---
for metric in ["MAE", "MRAE"]:

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for i, level in enumerate(levels):

        df = data_levels[level]
        summary = summary_metrics(df)

        ax = axes[i]

        for method in summary["method"].unique():

            df_m = summary[summary["method"] == method].sort_values("sample")

            ax.plot(df_m["sample"], df_m[metric], label=method)

        ax.set_title(f"{metric} - {level}")
        ax.legend(fontsize=8)

    plt.xticks([])
    plt.tight_layout()

    plt.savefig(f"ifcb_quantification_results/{metric}_all_levels.png", dpi=300)
    plt.close()


# --- PACC and EMQ line plots per sample across hierarchy levels ---
for metric in ["MAE", "MRAE"]:

    y_min, y_max = float("inf"), float("-inf")
    summaries = {}

    for level in levels:

        df = data_levels[level]
        summary = summary_metrics(df)

        summaries[level] = summary

        for method in ["PACC", "EMQ"]:
            df_m = summary[summary["method"] == method]
            y_min = min(y_min, df_m[metric].min())
            y_max = max(y_max, df_m[metric].max())

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)

    for i, method in enumerate(["PACC", "EMQ"]):

        ax = axes[i]

        for level in levels:

            df = summaries[level]
            df_m = df[df["method"] == method].sort_values("sample")

            ax.plot(df_m["sample"], df_m[metric], label=level)

        ax.set_title(method)
        ax.set_ylim(y_min, y_max)
        ax.legend()

    plt.xticks([])
    plt.tight_layout()

    plt.savefig(f"ifcb_quantification_results/{metric}_PACC_EMQ.png", dpi=300)
    plt.close()

print("\nAll results saved.")
