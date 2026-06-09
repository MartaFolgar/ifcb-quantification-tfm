'''
Correlation between MAE, MRAE and JSD per sample and FG
across hierarchy levels and quantification methods

Input: 
    - IFCB.csv
    - Prediction files:       
        - quantification_results_AC/ifcb_quantification_results_*.csv        
        - quantification_results_OC/ifcb_quantification_results_*.csv        
    - True prevalences:
        - quantification_results_AC/test_prevalences.csv
        - quantification_results_OC/test_prevalences.csv
    - JSD/JSD_AutoClass_FG.csv
    - JSD/JSD_OriginalClass_FG.csv
Output: 
    - correlation_JSD/jsd_mae_mrae.csv
    - correlation_JSD/AutoClass_MAE.png
    - correlation_JSD/AutoClass_MRAE.png
    - correlation_JSD/OriginalClass_MAE.png
    - correlation_JSD/OriginalClass_MRAE.png
'''

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Compute MAE and MRAE per sample and FG by aggregating classes into FunctionalGroups
def compute_sample_fg_errors(pred, true, data, col_class, method):

    # Map each AutoClass/Original class to its corresponding FunctionalGroup
    class_to_fg = dict(zip(data[col_class], data["FunctionalGroup"]))

    fg_groups = {}
    for cls in true.columns:
        fg = class_to_fg.get(cls)
        if fg is not None:
            fg_groups.setdefault(fg, []).append(cls)

    true = true.set_index("sample")

    rows = []

    # Loop over samples and Functional Groups to compute MAE and MRAE
    for sample in true.index:
        for fg, cls_list in fg_groups.items():

            y_true = true.loc[sample, cls_list].sum()

            y_pred = pred[
                (pred["sample"] == sample) &
                (pred["class"].isin(cls_list))
            ]["y_pred"].sum()

            abs_err = abs(y_true - y_pred)

            rows.append({
                "Sample": sample,
                "FunctionalGroup": fg,
                "METHOD": method,
                "MAE": abs_err,
                "MRAE": abs_err / (y_true + epsilon)
            })

    return pd.DataFrame(rows)

# Plot JSD vs MAE/MRAE per FunctionalGroup and method for hierarchy level
def plot_level(level_name):

    df = df_all[df_all["LEVEL"] == level_name]

    methods_list = sorted(df["METHOD"].unique())
    fgs = sorted(df["FunctionalGroup"].unique())

    for metric in metrics:

        fig, axes = plt.subplots(
            len(methods_list),
            len(fgs),
            figsize=(4 * len(fgs), 3 * len(methods_list)),
            sharex=True,
            sharey=True
        )

        for i, method in enumerate(methods_list):
            for j, fg in enumerate(fgs):

                ax = axes[i, j]

                df_sub = df[
                    (df["METHOD"] == method) &
                    (df["FunctionalGroup"] == fg)
                ]

                sns.scatterplot(
                    data=df_sub,
                    x="JSD",
                    y=metric,
                    ax=ax,
                    s=20
                )

                if len(df_sub) > 1:
                    corr = df_sub["JSD"].corr(df_sub[metric])
                    ax.text(0.05, 0.9, f"r={corr:.2f}",
                            transform=ax.transAxes, fontsize=8)

                if i == 0:
                    ax.set_title(fg)

                if j == 0:
                    ax.set_ylabel(f"{method}\n{metric}")

        plt.tight_layout()

        plt.savefig(
            f"correlation_JSD/{level_name}_{metric}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

# General settings
levels = {
    "AutoClass": "AC",
    "OriginalClass": "OC"
}

methods = ["CC", "PCC", "PACC", "EMQ"]
metrics = ["MAE", "MRAE"]

epsilon = 1e-3

os.makedirs("correlation_JSD", exist_ok=True)

# Load initial dataset (needed for AC/OC mapping to FG)
data = pd.read_csv("IFCB.csv")

all_data = []

# Loop over hierarchy levels (AutoClass/OriginalClass)
for level_name, folder in levels.items():

    # Loop over quantification methods
    for method in methods:

        # Load predictions and true prevalences
        pred = pd.read_csv(f"quantification_results_{folder}/ifcb_quantification_results_{method}.csv")
        true = pd.read_csv(f"quantification_results_{folder}/test_prevalences.csv")

        # Obtain MAE, MRAE and JSD per sample
        df = compute_sample_fg_errors(pred, true, data, level_name, method)

        jsd = pd.read_csv(f"JSD/JSD_{level_name}_FG.csv")

        df = df.merge(jsd, on=["Sample", "FunctionalGroup"], how="left")

        df["LEVEL"] = level_name

        all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

# Save results
df_all.to_csv("correlation_JSD/jsd_mae_mrae.csv", index=False)

# Plot correlation per hierarchy level
for level in levels:
    plot_level(level)

print("All results saved.")
