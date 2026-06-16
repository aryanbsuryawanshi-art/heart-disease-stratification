# Multi-Model Clinical Stratification of Coronary Artery Disease via Leakage-Proof Pipelines

An end-to-end predictive modeling framework evaluating seven statistical, kernel-based, and non-parametric classifiers to identify coronary artery disease risk using the UCI Cleveland cohort. This architecture leverages strict, validation-isolated preprocessing pipelines to evaluate baseline linear models against state-of-the-art tree ensembles without risking data contamination.

---

## 🛠️ Architectural & Design Rigor

* **Data Leakage Mitigation:** Continuous variable normalization (`StandardScaler`) and multi-class nominal mapping (`OneHotEncoder`) are handled concurrently inside an isolated `ColumnTransformer`. This step is explicitly nested inside a top-level Scikit-Learn `Pipeline` to enforce strict isolation between validation folds and training subsets during evaluation cycles.
* **Validation Strategy:** Implements 5-Fold `StratifiedKFold` Cross-Validation nested within hyperparameter grid optimization loops to protect against class imbalances and guarantee high generalization stability across distinct demographics.
* **Clinical Boundary Tuning:** Rather than deploying models against a naive default 0.5 classification threshold, the pipeline evaluates the structural trade-off between baseline precision and clinical sensitivity. Priority is assigned to maximizing **Recall (Sensitivity)** to structurally mitigate the critical risks of False Negatives in diagnostic pipelines.

## 📊 Pipeline Layout & Execution Matrix

```text
heart-disease-stratification/
├── data/
│   └── heart.csv              # UCI Cleveland clinical cohort
├── MLFIRSTPROJECT.ipynb       # Main analysis, validation, and training pipeline
├── requirements.txt           # Explicit production dependency pinning
└── README.md                  # System architecture documentation
