# TFM

Repository containing the code developed for the Master's Thesis:

*Quantification of Marine Plankton Prevalence from Flow Cytometry Images*

*Estimación de la prevalencia de especies de plancton marino mediante cuantificación basada en imágenes de citometría de flujo*

This project evaluates several quantification methods on the IFCB (Imaging FlowCytobot) dataset and investigates the relationship between dataset shift and quantification error across different levels of the taxonomic hierarchy. The study focuses on estimating the prevalence of marine plankton taxa from flow cytometry images and analyzing how changes in class distributions affect quantification performance.

---
## Repository Structure

```text
.
├── data/
├── results/
├── scripts/
│   ├── local/
│   └── server/
├── requirements_local.txt
├── requirements_server.txt
└── README.md
```

---
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
## Dataset

The dataset used in this project is the WHOI-Plankton dataset:

https://github.com/hsosik/WHOI-Plankton

The dataset contains plankton images grouped into samples acquired over time and annotated at different taxonomic levels.

---
## Quantification methods

The following quantification methods from the QuaPy library were evaluated:

https://github.com/HLT-ISTI/QuaPy

- CC (Classify and Count)
- ACC (Adjusted Classify and Count)
- PCC (Probabilistic Classify and Count)
- PACC (Probabilistic Adjusted Classify and Count)
- EMQ (Expectation Maximization for Quantification)
- DMy (Distribution Matching)

---
## Scripts

### Local scripts

- compute_jsd.py
  
  Compute Jensen–Shannon divergence (JSD) for AutoClass and OriginalClass hierarchical levels after aggregating classes into Functional Groups.

- ifcb_results.py
  
  Compute MAE and MRAE per method and hierarchy level.

- ifcb_aggregation_analysis.py
  
  Compute MAE and MRAE across hierarchical levels (FG, AutoClass, OriginalClass) for granularity analysis.

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

---

## Authors

**Marta Folgar Martínez**

Master's Thesis  
Máster Universitario en Inteligencia Artificial  
Universidad Internacional Menéndez Pelayo (UIMP)

### Supervisors

- Pablo González González (University of Oviedo)
- Olaya Pérez Mon (University of Oviedo)

