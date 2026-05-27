'''
Compute Jensen-Shannon divergence (JSD) for AutoClass and OriginalClass 
hierarchy levels after aggregating classes into FunctionalGroups

Input: 
    - IFCB.csv
Output: 
    - JSD/JSD_{level}_FG.csv
'''

import os,sys
import pandas as pd
import numpy as np
import random
from scipy.spatial.distance import jensenshannon

# Check missing test classes in train
def check_missing_classes(data, samplestraining, level, classes):
    datatraining = data[data['Sample'].isin(samplestraining)]

    missing = [
        c for c in classes
        if c not in np.unique(datatraining[level])
    ]

    print("\nClasses not present in training samples:", missing)

    for c in missing:
        count = len(data[data[level] == c])
        print(f"Class '{c}' has {count} examples.")

    return missing

# Training prevalence matrix: P(class | FunctionalGroup)
def build_train_matrix(df, level):
    counts = (
        df.groupby(['FunctionalGroup', level])
        .size()
        .reset_index(name='count')
    )

    totals = df.groupby('FunctionalGroup').size().reset_index(name='total')

    counts = counts.merge(totals, on='FunctionalGroup')
    counts['prevalence'] = counts['count'] / counts['total']

    matrix = counts.pivot(
        index='FunctionalGroup',
        columns=level,
        values='prevalence'
    ).fillna(0)

    return matrix

# JSD between train and test distributions per FG and sample
def compute_jsd(data_test, level, pxy_matrix, min_rois=30):
    results = []

    for sample in data_test['Sample'].unique():

        df_sample = data_test[data_test['Sample'] == sample]
        fg_counts = df_sample.groupby('FunctionalGroup').size()

        # ROIs per FG
        summary = (
            df_sample.groupby(['FunctionalGroup', level])
            .size()
            .reset_index(name='count')
        )

        # Compute class prevalence per FG
        summary = summary.merge(
            fg_counts.rename_axis('FunctionalGroup').reset_index(name='total'),
            on='FunctionalGroup'
        )

        summary['prevalence'] = summary['count'] / summary['total']

        # Compute JSD per FG
        for fg in pxy_matrix.index:

            train_vec = pxy_matrix.loc[fg].values

            if fg not in fg_counts.index or fg_counts[fg] < min_rois:
                jsd = np.nan
            else:
                test_vec = (
                    summary[summary['FunctionalGroup'] == fg]
                    .set_index(level)['prevalence']
                    .reindex(pxy_matrix.columns)
                    .fillna(0)
                    .values
                )

                jsd = jensenshannon(train_vec, test_vec, base=2)

            results.append({
                'Sample': sample,
                'FunctionalGroup': fg,
                'JSD': jsd
            })

    return pd.DataFrame(results)

# Hierarchy levels to evaluate
levels = ["AutoClass", "OriginalClass"]

# For reproducibility
random.seed(2032)

# Load IFCB dataset
data = pd.read_csv('IFCB.csv')

# Extract year from sample
data['year'] = data['Sample'].apply(lambda s: int(s.split('_')[1]))
samples = data.groupby('Sample').first()[["year"]]

# Split train/test
train_years = [2006, 2007, 2008]
test_years = [2009, 2010, 2011, 2012, 2013, 2014]
trainval = list(samples[samples['year'].isin(train_years)].index)
random.shuffle(trainval)

# Split 70% train / 30% validation
split = int(len(trainval) * 0.7)
samples_train = trainval[:split]
samples_val = trainval[split:]
samples_test = list(samples[samples['year'].isin(test_years)].index)

# Output folder
os.makedirs("JSD", exist_ok=True)

# Loop over hierarchy levels
for level in levels:

    print(f"\nProcessing {level}")

    # Remove classes not present in training to avoid unseen-label issues
    classes = np.sort(data[level].unique())
    missing = check_missing_classes(data, samples_train, level, classes)
    data_clean = data[~data[level].isin(missing)]

    # Training, validation and test split at sample level
    data_train = data_clean[data_clean['Sample'].isin(samples_train)]
    data_val = data_clean[data_clean['Sample'].isin(samples_val)]
    data_test = data_clean[data_clean['Sample'].isin(samples_test)]

    print("\nTrain matrix:")
    pxy_matrix = build_train_matrix(data_train, level)
    print(pxy_matrix)

    # Compute JSD
    df_jsd = compute_jsd(data_test, level, pxy_matrix)

    # Save results
    out_file = f"JSD/JSD_{level}_FG.csv"
    df_jsd.to_csv(out_file, index=False)

    print("Saved:", out_file)
