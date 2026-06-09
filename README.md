# TFM

## Environments

Two different environments were used during the development of this project.

### Local environment
Used for:

- Distributional analysis
- Jensen-Shannon divergence computation
- Results aggregation
- Statistical analysis
- Visualization


**Python version**

```text
Python 3.11.9
```

Install dependencies:

```bash
pip install -r requirements_local.txt
```

### Server environment

Used for:

- Quantification experiments

**Python version**

```text
Python 3.10.12
```

Install dependencies:

```bash
pip install -r requirements_server.txt
```

---

## Requirements

### requirements_local.txt

```text
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
matplotlib==3.8.4
seaborn==0.13.2
```

### requirements_server.txt

```text
numpy==2.2.6
pandas==2.3.3
scikit-learn==1.7.2
quapy==0.2.0
```

---

## Scripts

### Local scripts

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

### Server scripts

- run_quantifiers.py

  Evaluate quantification methods on the IFCB dataset.

---

## Execution pipeline

### 1. Quantification experiments (server)

```bash
python scripts/server/run_quantifiers.py
```

### 2. Compute distributional shift (local)

```bash
python scripts/local/compute_jsd.py
```

### 3. Compute quantification errors (local)

```bash
python scripts/local/ifcb_results.py
```

### 4. Analyze taxonomic granularity (local)

```bash
python scripts/local/ifcb_aggregation_analysis.py
```

### 5. Analyze correlation between JSD and quantification error (local)

```bash
python scripts/local/correlation_jsd.py
```

---

## Authors

**Marta Folgar Martínez**

Master's Thesis  
Máster Universitario en Inteligencia Artificial  
Universidad Internacional Menéndez Pelayo (UIMP)

### Supervisors

- Pablo González González (University of Oviedo)
- Olaya Pérez Mon (University of Oviedo)

