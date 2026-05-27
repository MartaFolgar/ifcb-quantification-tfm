# TFM

## Scripts

- compute_jsd.py
  
  Compute Jensen–Shannon divergence (JSD) for AutoClass and OriginalClass hierarchical levels after aggregating classes into Functional Groups.

- run_quantifiers.py
  
  Evaluate quantification methods on the IFCB dataset.

- ifcb_results.py
  
  Compute MAE and MRAE per method and hierarchy level.

- ifcb_aggregation_analysis.py
  
  Compute MAE and MRAE across hierarchical levels (FG, AutoClass, OriginalClass) for granularity analysis.

- correlation_jsd.py
  
  Analyze correlations between MAE, MRAE, and JSD per sample and Functional Group across hierarchical levels and quantification methods.

## Execution pipeline
python scripts/compute_jsd.py

python scripts/run_quantifiers.py

python scripts/ifcb_results.py

python scripts/ifcb_aggregation_analysis.py

python scripts/correlation_jsd.py
