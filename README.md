# SNP Prioritization Using Chromatin Accessibility and Machine Learning

> 📌 **Note:** The input data must be preprocessed single-cell ATAC-seq data. We recommend using [Signac](https://satijalab.org/signac/) to generate peak-level accessibility matrices and annotations and then integrated with gwas summary statistics prior to running this pipeline.

This repository implements a modular pipeline for prioritizing putative regulatory SNPs based on chromatin accessibility features using a machine learning regression model.

## 🧩 Pipeline Overview

1. **Data Preparation**
   - Load SNP-peak matrix with associated features (e.g., chromatin accessibility in PD vs. HC, p-values, TSS distance).
   - Clean and preprocess data (handle missing values, filter outliers if needed).

2. **Modeling**
   - Train an XGBoost regression model to predict SNP `effect_size` as a proxy for regulatory impact.
   - Perform hyperparameter tuning via cross-validation.

3. **Evaluation**
   - Report standard metrics: MSE, MAE, R² on train/test sets.
   - Visualize model performance and prediction distribution.

4. **Feature Interpretation**
   - Extract global feature importance from the model.
   - Compute SHAP values to assess per-feature contributions to predictions.

5. **SNP Ranking**
   - Rank all input SNPs by predicted effect size.
   - Export ranked list and model diagnostics.

## 📁 Key Outputs
- `ranked_snps_improved.csv`: Top SNPs based on predicted effect size.
- `feature_importance.csv`: Ranked feature contributions.
- `model_evaluation_plots.png`: Visual summary of performance.
- `shap_summary_plot.png`: SHAP-based interpretation (if SHAP installed).

## ✨ Notes
- The model focuses on chromatin accessibility-derived features from single-cell datasets.
- Code is modular and can be adapted to include additional annotations such as eQTL, conservation, or enhancer activity.

---
