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
'''

import numpy as np
import pandas as pd
import os

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
            mrae_dict[fg] = np.mean(abs_err / (true[fg] + epsilon))

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
                mrae_vals[fg].append(abs_err / (y_true + epsilon))

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
methods = ["CC", "ACC", "PCC", "PACC", "EMQ"]

# Hierarchy levels
levels = ["FG", "AC", "OC"]

# Output folder
output_dir = "ifcb_quantification_results"
os.makedirs(output_dir, exist_ok=True)

# Load initial dataset (needed for AutoClass/OriginalClass mapping to FunctionalGroup)
data = pd.read_csv("IFCB.csv")

all_mae = []
all_mrae = []

# Loop over quantification methods
for method in methods:

    results_mae = {}
    results_mrae = {}

    # Loop over hierarchy levels
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
