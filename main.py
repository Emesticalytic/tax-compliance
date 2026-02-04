"""
Tax Compliance Risk Analysis - End-to-End Pipeline
This script runs the complete data science pipeline from data generation to model evaluation.
"""

from src.data_generation import generate_data
from src.preprocessing import clean_data
from src.features import build_features
from src.train import train_models
from src.evaluate import evaluate_models
from src.visualizations import create_eda_charts, create_model_evaluation_charts
import os


def main():
    """Run the complete end-to-end pipeline."""
    print("=" * 70)
    print("TAX COMPLIANCE RISK ANALYSIS - PIPELINE")
    print("=" * 70)
    
    # Step 1: Generate synthetic data
    print("\n[1/6] Generating synthetic taxpayer data...")
    df = generate_data(n=10000, seed=42)
    print(f"✓ Generated {len(df)} taxpayer records")
    
    # Step 2: Data cleaning and preprocessing
    print("\n[2/6] Cleaning and preprocessing data...")
    df = clean_data(df)
    print("✓ Data cleaned successfully")
    
    # Step 3: Exploratory Data Analysis
    print("\n[3/6] Creating EDA visualizations...")
    create_eda_charts(df)
    print("✓ EDA charts saved to output/eda/")
    
    # Step 4: Feature engineering
    print("\n[4/6] Building features...")
    X, y = build_features(df)
    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"✓ Target distribution: {y.value_counts().to_dict()}")
    
    # Step 5: Model training
    print("\n[5/6] Training models...")
    model = train_models(X, y)
    print("✓ Model trained and saved to output/model/")
    
    # Step 6: Model evaluation
    print("\n[6/6] Evaluating model performance...")
    results = evaluate_models(model, X, y)
    create_model_evaluation_charts(model, X, y)
    print("✓ Evaluation complete. Charts saved to output/model/")
    print(f"✓ Model AUC: {results['auc']:.4f}")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  • View EDA charts in output/eda/")
    print("  • View model evaluation in output/model/")
    print("  • View dashboard at output/dashboard/roc_curve.html")
    print("  • Run Streamlit app: streamlit run streamlit_app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
