'''
Evaluation of quantification methods on the IFCB dataset

Input: 
    - IFCB dataset in /quapy_data/ifcb
    - Folder structure:
        /ifcb/
            - train/
            - test/
            - test_prevalences.csv
Output: 
    - /results/quantification_results_{level}/ifcb_quantification_results_{method}.csv
'''

import os
import numpy as np
import pandas as pd
import quapy as qp

from quapy.method.aggregative import CC, ACC, PCC, PACC, EMQ, DMy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone

qp.environ['N_JOBS'] = 4  # number of parallel processes

# Load IFCB dataset 
# for_model_selection=False: Uses a standard train/test split without internal validation
# single_sample_train=True: Trains using a single aggregated sample
train, _ = qp.datasets.fetch_IFCB(for_model_selection=False, single_sample_train=True)

# Load test dataset
test_path = os.path.expanduser("~/quapy_data/ifcb/test")

# Extract class labels from training set
class_names = train.classes_ if hasattr(train, "classes_") else np.unique(train.y)

# Classification pipeline using standardized features and logistic regression
base_clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=5000,
        solver='lbfgs'
    )
)

# Calibrated classifier using Platt scaling (sigmoid) with 4-fold cross-validation
calibrated_clf = CalibratedClassifierCV(
    estimator=base_clf,
    method='sigmoid',
    cv=4
)

# Quantification methods to test
quantifiers = {

    # Hard prediction methods
    "CC": CC(clone(base_clf)),
    "ACC": ACC(clone(base_clf)),

    # Probabilistic methods
    "PCC": PCC(clone(base_clf)),
    "PACC": PACC(clone(base_clf)),
    "EMQ": EMQ(clone(calibrated_clf)),

    # Distribution matching
    "DMy_b4": DMy(clone(calibrated_clf), nbins=4),
    "DMy_b8": DMy(clone(calibrated_clf), nbins=8),
    "DMy_b16": DMy(clone(calibrated_clf), nbins=16)
}   

# Create output directory
os.makedirs("results", exist_ok=True)

# Loop over quantification methods
for method_name, quantifier in quantifiers.items():

    # Train quantification model
    print(f"\n\nTraining {method_name}...")
    model = quantifier.fit(train.X, train.y)
    print(f"{method_name} training completed.")

    results = []

    # Iterate over test samples
    for file in os.listdir(test_path):

        if not file.endswith(".csv"):
            continue

        sample_id = file.replace(".csv", "")

        df = pd.read_csv(os.path.join(test_path, file))

        X_i = df.select_dtypes(include=[np.number]).values

        # Estimate class prevalences
        y_pred = np.asarray(model.quantify(X_i))

        # Store predicted prevalences
        for class_id in range(len(class_names)):

            class_label = class_names[class_id]

            results.append({
                "method": method_name,
                "sample": sample_id,                
                "class": class_label,
                "y_pred": y_pred[class_id]
            })

    # Convert to DataFrame for storage
    df_results = pd.DataFrame(results)

    # Store results per method
    output_path = f"results/ifcb_quantification_results_{method_name}.csv"
    df_results.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

print("All quantification methods processed.")
