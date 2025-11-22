# Complete Cancer Analysis ML Pipeline

## NeoHacks 2025 - TCGA HNSC Analysis

**Team NeuralBits** | Abdul Wali Khan University Mardan

This directory contains a comprehensive machine learning pipeline for analyzing Head & Neck Squamous Cell Carcinoma (HNSC) data from The Cancer Genome Atlas (TCGA).

## 📁 Directory Structure

```
ML_Pipeline/
├── data/                          # Processed data (CSV format)
│   ├── patient_features.csv       # Main patient-level dataset (260 features)
│   ├── task1_sample_labels.csv    # Sample-level labels for Task I
│   └── results_summary.csv        # Final results table
├── notebooks/
│   └── Complete_Cancer_Analysis_Pipeline.ipynb  # Main pipeline notebook
├── results/                       # Generated visualizations
│   ├── task_comparison.png        # Model performance comparison
│   ├── roc_curves_all_tasks.png   # ROC curves for all 3 tasks
│   ├── eda_distributions.png      # Feature distributions
│   └── feature_correlations.png   # Correlation analysis
├── presentation.tex               # LaTeX Beamer presentation
├── presentation.pdf               # Compiled presentation
└── README.md                      # This file
```

## 🎯 Three Clinical Tasks Implemented

### Task I: Cancer vs Normal Classification
- **Goal:** Distinguish tumor tissue from normal tissue
- **Method:** Logistic Regression with DNA methylation features
- **Data Level:** Sample-level (96 samples: 82 tumor, 14 normal)
- **Features:** Top 500 variant CpG methylation sites (selected per fold)
- **Result:** **100% accuracy, AUC = 1.0**
- **Key Insight:** DNA methylation patterns provide perfect biomarkers for cancer detection

**What the model looks for:**
- Beta values (0-1) at 500 CpG sites across the genome
- Hypermethylation patterns in tumor suppressor genes
- Hypomethylation in oncogenic regions
- Epigenetic signatures that fundamentally differ between cancer and normal cells

---

### Task II: Stage Classification (Early vs Late)
- **Goal:** Predict cancer stage (Early: I/II vs Late: III/IV)
- **Method:** Neural Network Ensemble (5 models averaged)
- **Data Level:** Patient-level (82 patients)
- **Features:** Multi-modal genomic data (~130 features selected per fold)
  - Top 30 mutation features (binary mutation status)
  - Top 50 expression features (log2-transformed RNA-seq)
  - Top 50 methylation features (beta values)
- **Result:** **94.5% ± 1.3% accuracy, F1 = 0.95, AUC = 0.98**
- **Key Insight:** Multi-modal integration captures complex molecular stage signatures

**What the model looks for:**
- **Mutations:** TP53, PIK3CA, NOTCH1, FAT1 (key driver mutations)
- **Gene Expression:** Highly variant genes showing differential expression in advanced cancer
- **DNA Methylation:** Epigenetic changes that accumulate with disease progression
- Combined patterns across all three modalities to identify late-stage molecular characteristics

---

### Task III: Clinical Risk Factor Analysis
- **Goal:** Predict poor prognosis using clinical risk factors
- **Method:** Random Forest (best performing)
- **Data Level:** Patient-level (82 patients)
- **Features:** 6 interpretable clinical risk factors
  - Age at diagnosis
  - Gender (male/female)
  - Smoking status (pack-years)
  - Is smoker (binary)
  - Tumor grade
  - Cancer stage (early/late)
- **Result:** **83.3% ± 0.8% accuracy, F1 = 0.89, AUC = 0.92**
- **Key Insight:** Smoking and cancer stage are the strongest clinical predictors

**What the model looks for:**
- **Primary risk factors:** Smoking history (strongest predictor for HNSC)
- **Disease severity:** Cancer stage (late-stage indicates poor prognosis)
- **Demographics:** Age and gender correlations with outcomes
- **Combined risk profile:** Interaction patterns between multiple clinical factors

---

## 📊 Dataset Overview

### Data Source
- **Database:** The Cancer Genome Atlas (TCGA)
- **Cancer Type:** Head & Neck Squamous Cell Carcinoma (HNSC)
- **Total Patients:** 82
- **Data Modalities:** 4 (Genomic, Transcriptomic, Epigenomic, Clinical)

### Files Used (5 out of 7)

| File | Size | Used In | Purpose |
|------|------|---------|---------|
| **clinical.txt** | 309 rows × 210 cols | Task II, III | Age, gender, vital status, cancer stage |
| **mutations.txt** | 13,180 mutations | Task II | Genomic mutations (50 top genes) |
| **transcriptomics.txt** | 20,503 genes × 92 samples | Task II | Gene expression (RNA-seq) |
| **methylation.txt** | 20,114 CpG sites × 96 samples | Task I, II | DNA methylation beta values |
| **exposure.txt** | 114 rows × 40 cols | Task III | Smoking status, pack-years |
| ~~follow_up.txt~~ | *(Loaded but unused)* | — | Follow-up data (future work) |
| ~~pathology_detail.txt~~ | *(Loaded but unused)* | — | Pathology details (future work) |

### Feature Summary

**Task I (Cancer Detection):**
- 20,114 CpG sites available → 500 most variant selected per fold
- Sample-level classification

**Task II (Stage Classification):**
- 50 mutation features → Top 30 selected per fold
- 100 expression features → Top 50 selected per fold
- 100 methylation features → Top 50 selected per fold
- Total: ~130 multi-modal features

**Task III (Risk Factors):**
- 6 clinical risk factors (all used, no selection)

---

## 🚀 How to Run

### Prerequisites

```bash
# Install required packages
pip install pandas numpy scikit-learn torch imblearn matplotlib seaborn jupyter
```

**Python version:** 3.8+ recommended
**PyTorch version:** 2.0+

### Run the Pipeline

1. **Navigate to notebooks directory:**
   ```bash
   cd NeoHacks2025Hackathon/ML_Pipeline/notebooks
   ```

2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook Complete_Cancer_Analysis_Pipeline.ipynb
   ```

3. **Execute all cells:**
   - Click "Kernel" → "Restart & Run All"
   - Or run cells sequentially from top to bottom

4. **Expected runtime:** ~5-10 minutes (depending on hardware)

5. **Expected outputs:**
   - Processed CSV files saved to `../data/`
   - Visualizations saved to `../results/`
   - Performance metrics printed in notebook output

---

## 📈 Results Summary

### Overall Performance

| Task | Method | Accuracy | F1-Score | AUC |
|------|--------|----------|----------|-----|
| **Task I: Cancer Detection** | Logistic Regression | **100.0% ± 0.0%** | 1.00 | 1.00 |
| **Task II: Stage Classification** | Neural Network Ensemble | **94.5% ± 1.3%** | 0.95 | 0.98 |
| **Task III: Risk Factor Analysis** | Random Forest | **83.3% ± 0.8%** | 0.89 | 0.92 |

*All results validated with 5-fold stratified cross-validation*

### Key Findings

1. **Perfect Cancer Detection (100%):** DNA methylation provides complete separation between tumor and normal tissue - biologically validated and scientifically sound.

2. **Excellent Stage Prediction (94.5%):** Multi-modal integration of mutations, expression, and methylation captures disease progression signatures.

3. **Strong Risk Stratification (83.3%):** Clinical factors alone can predict outcomes effectively, with smoking and stage as primary predictors.

---

## 🔬 Technical Highlights

### Methodological Rigor

✅ **No Data Leakage**
- Feature selection performed **INSIDE** cross-validation loops
- Each fold selects features only from training data
- Validation sets remain completely unseen during feature selection

✅ **Proper Cross-Validation**
- 5-fold stratified cross-validation
- Maintains class proportions in each fold
- Reports validation accuracy (NOT training accuracy)

✅ **Robust Evaluation**
- Multiple metrics: Accuracy, F1-Score, AUC-ROC
- Statistical significance: Mean ± Standard Deviation
- Confusion matrices and ROC curves generated

### Advanced Techniques

**Task I:**
- SimpleImputer for missing value handling
- Variance-based feature selection (top 500 CpG sites)
- Logistic regression with max_iter=1000

**Task II:**
- Neural Network architecture: 128 → 64 → 2 neurons
- 5-model ensemble with prediction averaging
- Dropout (0.5) and L2 regularization (weight_decay=1e-3)
- Gradient clipping (max_norm=1.0)
- Balanced class weights
- Adam optimizer (lr=0.0005)

**Task III:**
- Random Forest (n_estimators=100, max_depth=4)
- Balanced class weights
- Median imputation for missing values

### Data Processing Pipeline

1. **Patient-level aggregation:** Sample-level data aggregated to patient level
2. **Feature engineering:**
   - Mutation matrix (binary encoding)
   - Log2 transformation for expression
   - Top variant feature selection
3. **Normalization:**
   - Expression data: log2(x + 1)
   - Methylation: Already normalized (beta values 0-1)
4. **Missing value handling:** Mean/median imputation

---

## 💡 Innovation & Contributions

### 1. Multi-Modal Data Integration
Combined 4 different biological data types for comprehensive cancer analysis:
- Genomic (mutations)
- Transcriptomic (gene expression)
- Epigenomic (DNA methylation)
- Clinical (demographics, smoking)

### 2. Rigorous Methodology
- Feature selection inside CV loops (prevents overfitting)
- Stratified k-fold validation (maintains class balance)
- Ensemble methods for stability
- Statistical significance reporting

### 3. Clinical Interpretability
- Task III uses interpretable clinical features
- Identifies actionable risk factors (smoking, stage)
- Provides evidence-based insights for clinical decision-making

### 4. Biological Validation
- Results align with known cancer biology
- Methylation as cancer biomarker (well-established)
- TP53 mutations in 83% of patients (expected in HNSC)
- Smoking as primary risk factor (validated epidemiologically)

### 5. Reproducibility
- All data in accessible CSV format (not pickle)
- Fixed random seed (42) for reproducibility
- Complete Jupyter notebook with detailed documentation
- Self-contained pipeline (no external dependencies)

---

## 🧬 Biological Insights

### Top Mutated Genes (HNSC Drivers)
1. **TP53 (83%):** Tumor suppressor, most common cancer mutation
2. **PIK3CA (18%):** PI3K pathway, targetable with drugs
3. **NOTCH1 (23%):** Cell differentiation, HNSC-specific driver

### DNA Methylation Patterns
- **Hypermethylation:** Silencing of tumor suppressor genes
- **Hypomethylation:** Activation of oncogenic pathways
- **CpG Islands:** Key regulatory regions showing differential methylation

### Clinical Risk Factors
- **Smoking:** Primary environmental risk factor for HNSC
- **Late Stage (III/IV):** Strong predictor of poor outcomes
- **Age:** Older patients show worse prognosis

### Gene Expression Signatures
- Variant genes capture transcriptional changes in cancer
- Log2 transformation normalizes skewed expression distributions
- Top 100 variant genes most informative for classification

---

## 📝 Notes & Limitations

### Strengths
✅ Rigorous cross-validation methodology
✅ Multi-modal data integration
✅ Biologically interpretable results
✅ Complete reproducibility
✅ No data leakage

### Limitations
⚠️ Small dataset (82 patients) - common for TCGA studies
⚠️ No external validation cohort
⚠️ Class imbalance in some tasks (handled with balanced weights)
⚠️ Feature selection could be enhanced with biological pathway knowledge

### Future Work
- External validation on independent HNSC cohorts
- Survival analysis using follow_up data
- Integration of pathology image data
- Pathway-based feature selection
- Deep learning fusion models
- Clinical deployment as decision support tool

---

## 📚 Technical Stack

**Languages & Frameworks:**
- Python 3.8+
- PyTorch 2.0+ (deep learning)
- scikit-learn (classical ML)
- pandas (data manipulation)
- NumPy (numerical computing)

**Key Libraries:**
- `sklearn`: Logistic Regression, Random Forest, metrics, CV
- `torch`: Neural networks, optimization
- `imblearn`: Class imbalance handling (SMOTE)
- `matplotlib` & `seaborn`: Visualization

**Development Environment:**
- Jupyter Notebook
- LaTeX (Beamer) for presentation

---

## 📧 Contact & Attribution

**Team NeuralBits**
- Iftikhar Ali
- Hilal Khan

**Institution:** Abdul Wali Khan University Mardan
**Event:** NeoHacks 2025 Hackathon
**Dataset:** TCGA HNSC (The Cancer Genome Atlas)

---

## 🏆 Competition Highlights

**What Makes This Project Stand Out:**

1. ✅ **Perfect cancer detection** using DNA methylation
2. ✅ **94.5% stage prediction** with multi-modal integration
3. ✅ **Rigorous methodology** preventing data leakage
4. ✅ **Interpretable risk factors** for clinical use
5. ✅ **Complete reproducibility** with documented pipeline
6. ✅ **Biological validation** of all findings

**Generated for NeoHacks 2025 Hackathon** 🚀

---

*Last Updated: 2025-11-22*


