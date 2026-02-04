# 🚀 Quick Setup Guide - Tax Compliance Analysis

This guide will help you set up and run the complete project in under 5 minutes.

---

## ⚡ Quick Start (For the Impatient)

```bash
# 1. Navigate to project directory
cd "/Users/ememakpan/Desktop/Compliance Analysis"

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the complete pipeline
python main.py

# 5. Launch interactive dashboard
streamlit run streamlit_app.py
```

That's it! Your browser will open with the interactive dashboard.

---

## 📝 Step-by-Step Setup

### Step 1: Prerequisites
Ensure you have Python 3.8+ installed:
```bash
python --version  # Should show 3.8 or higher
```

### Step 2: Virtual Environment
Create an isolated Python environment:
```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows
```

You should see `(venv)` in your terminal prompt.

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `numpy`, `pandas` - Data manipulation
- `scikit-learn` - Machine learning
- `matplotlib`, `seaborn`, `plotly` - Visualizations
- `streamlit` - Interactive dashboard
- `joblib` - Model persistence
- `pytest` - Testing framework

### Step 4: Run the Pipeline
```bash
python main.py
```

**Expected output:**
```
======================================================================
TAX COMPLIANCE RISK ANALYSIS - PIPELINE
======================================================================

[1/6] Generating synthetic taxpayer data...
✓ Generated 10000 taxpayer records

[2/6] Cleaning and preprocessing data...
✓ Data cleaned successfully

[3/6] Creating EDA visualizations...
✓ EDA charts saved to output/eda/

[4/6] Building features...
✓ Feature matrix shape: (10000, 5)
✓ Target distribution: {0: 8000, 1: 2000}

[5/6] Training models...
✓ Model trained and saved to output/model/

[6/6] Evaluating model performance...
============================================================
MODEL EVALUATION RESULTS
============================================================
AUC-ROC Score: 0.9970
Classification Report:
  Precision (Risk=1): 0.9500
  Recall (Risk=1):    0.9700
  F1-Score (Risk=1):  0.9599
  Accuracy:           0.9680
============================================================

✓ Evaluation complete. Charts saved to output/model/
✓ Model AUC: 0.9970

======================================================================
PIPELINE COMPLETE
======================================================================

Next steps:
  • View EDA charts in output/eda/
  • View model evaluation in output/model/
  • View dashboard at output/dashboard/roc_curve.html
  • Run Streamlit app: streamlit run streamlit_app.py
======================================================================
```

**Time**: ~30 seconds

### Step 5: Explore Outputs
```bash
# View directory structure
tree output/

# Or manually navigate:
# output/eda/           - 5 EDA charts
# output/model/         - 5 model evaluation charts + saved model
# data/raw/            - Generated CSV data
```

### Step 6: Launch Dashboard
```bash
streamlit run streamlit_app.py
```

Your browser will automatically open to `http://localhost:8501` with a 6-page interactive dashboard:

1. **📊 Overview** - Project summary and key metrics
2. **🔎 Data Explorer** - Filter and browse 10K taxpayer records
3. **📈 Model Performance** - ROC curves, confusion matrices
4. **⚠️ Risk Scoring** - Real-time risk assessment tool
5. **🎯 Feature Importance** - Model interpretability
6. **🎚️ Threshold Analysis** - Precision-recall optimization

**Tip**: Use the sidebar to navigate between pages.

---

## 🧪 Run Tests (Optional)

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

**Expected output:**
```
tests/test_pipeline.py::TestDataGeneration::test_generate_data_shape PASSED
tests/test_pipeline.py::TestDataGeneration::test_generate_data_reproducibility PASSED
...
========================== 17 passed in 5.23s ==========================
```

---

## 🐙 Push to GitHub

### First-time Setup
```bash
# Initialize Git repository
git init

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: Tax compliance analysis project"

# Create GitHub repo (via GitHub website or CLI)
gh repo create tax-compliance-analysis --public --source=. --remote=origin

# Push to GitHub
git push -u origin main
```

### Update README
Before pushing, update these sections in `README.md`:

1. **Line 152**: Replace GitHub URL with your repo
2. **Line 556**: Update author information
3. **Line 557-559**: Add your GitHub, LinkedIn, email

```bash
# After updating README
git add README.md
git commit -m "Update README with personal information"
git push
```

---

## 📦 Project Structure After Setup

```
Compliance Analysis/
│
├── main.py                      # ✅ Pipeline orchestration
├── streamlit_app.py             # ✅ Interactive dashboard
├── requirements.txt             # ✅ Dependencies
├── README.md                    # ✅ Documentation
├── LICENSE                      # ✅ MIT license
├── .gitignore                   # ✅ Git ignore rules
│
├── src/                         # ✅ Source code
│   ├── __init__.py
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualizations.py
│
├── data/                        # ✅ Generated after running main.py
│   └── raw/
│       └── synthetic_taxpayers.csv  (10K records, ~700KB)
│
├── output/                      # ✅ Generated after running main.py
│   ├── eda/
│   │   ├── risk_distribution.png
│   │   ├── income_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── risk_by_property.png
│   │   └── income_by_risk.png
│   ├── model/
│   │   ├── risk_model.pkl  (~100KB)
│   │   ├── roc_curve.png
│   │   ├── precision_recall_curve.png
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   └── probability_distribution.png
│   └── dashboard/
│       └── roc_curve.html
│
├── tests/                       # ✅ Unit tests
│   └── test_pipeline.py
│
└── venv/                        # ⚠️ Virtual environment (not in Git)
```

---

## 🎤 Interview Preparation Checklist

### Before the Interview
- [ ] Run `python main.py` to ensure everything works
- [ ] Test Streamlit dashboard on all pages
- [ ] Review README talking points section
- [ ] Practice explaining threshold analysis
- [ ] Prepare 3 questions to ask interviewer (see README)

### Demo Script (5 minutes)
1. **Show GitHub repo** (30 sec)
   - Clean structure, good documentation
   
2. **Run pipeline** (30 sec)
   - `python main.py` - emphasize reproducibility
   
3. **Open Streamlit dashboard** (3 min)
   - Overview page: Show AUC ~0.997
   - Threshold Analysis: Demonstrate precision-recall trade-off
   - Risk Scoring: Input a sample taxpayer
   
4. **Code walkthrough** (1 min)
   - Show modular structure in `src/`
   - Highlight feature engineering in `features.py`

### Key Messages
✅ **"I built an end-to-end ML pipeline for tax compliance risk assessment"**  
✅ **"The model achieves 99.7% AUC with explainable features"**  
✅ **"I included threshold tuning because operational deployment requires flexibility"**  
✅ **"The code is production-ready with tests, documentation, and error handling"**

---

## 🛠️ Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'streamlit'`
**Solution**: Activate virtual environment and reinstall
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: `FileNotFoundError: data/raw/synthetic_taxpayers.csv`
**Solution**: Run the pipeline first
```bash
python main.py
```

### Issue: Streamlit dashboard is blank
**Solution**: Check browser console for errors. Try:
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Issue: Charts not displaying in Streamlit
**Solution**: Ensure matplotlib/seaborn are installed
```bash
pip install matplotlib seaborn --upgrade
```

### Issue: Model performance is different
**Solution**: This is normal due to random seed variations. Rerun with same seed for reproducibility.

---

## 🎯 Next Steps

1. **Customize the project**:
   - Update author information in README
   - Add your own insights to the analysis
   - Experiment with different models (Gradient Boosting, XGBoost)

2. **Extend functionality**:
   - Add more features (temporal patterns, network analysis)
   - Implement SHAP values for explanations
   - Create API endpoint with FastAPI

3. **Prepare for interview**:
   - Practice explaining technical choices
   - Review sklearn documentation for Random Forest
   - Think about production deployment strategy

---

## 📞 Need Help?

If you encounter issues:
1. Check the Troubleshooting section above
2. Review error messages carefully
3. Search Stack Overflow for specific errors
4. Check package documentation

**Common Resources**:
- Scikit-learn: https://scikit-learn.org/stable/
- Streamlit: https://docs.streamlit.io/
- Pandas: https://pandas.pydata.org/docs/

---

Copyright (c) 2026 [EMEM AKPAN]. This project is licensed under the MIT License.
**Practice makes perfect   🎉**
