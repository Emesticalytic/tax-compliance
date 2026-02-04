"""
Unit tests for Tax Compliance Analysis Pipeline
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generation import generate_data
from src.preprocessing import clean_data
from src.features import build_features
from src.train import train_models
from src.evaluate import evaluate_models


class TestDataGeneration:
    """Test data generation module."""
    
    def test_generate_data_shape(self):
        """Test that generated data has correct shape."""
        df = generate_data(n=1000, seed=42)
        assert len(df) == 1000
        assert 'taxpayer_id' in df.columns
        assert 'risk_flag' in df.columns
    
    def test_generate_data_reproducibility(self):
        """Test that data generation is reproducible with same seed."""
        df1 = generate_data(n=100, seed=42)
        df2 = generate_data(n=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_generate_data_different_seeds(self):
        """Test that different seeds produce different data."""
        df1 = generate_data(n=100, seed=42)
        df2 = generate_data(n=100, seed=123)
        assert not df1.equals(df2)
    
    def test_risk_flag_distribution(self):
        """Test that risk flag has both classes."""
        df = generate_data(n=1000, seed=42)
        assert df['risk_flag'].nunique() == 2
        assert df['risk_flag'].min() == 0
        assert df['risk_flag'].max() == 1


class TestPreprocessing:
    """Test preprocessing module."""
    
    def test_clean_data_preserves_shape(self):
        """Test that cleaning preserves number of rows."""
        df = generate_data(n=100, seed=42)
        df_clean = clean_data(df)
        assert len(df_clean) == len(df)
    
    def test_clean_data_adds_income_log(self):
        """Test that income_log column is added."""
        df = generate_data(n=100, seed=42)
        df_clean = clean_data(df)
        assert 'income_log' in df_clean.columns
    
    def test_clean_data_handles_copy(self):
        """Test that cleaning doesn't modify original dataframe."""
        df = generate_data(n=100, seed=42)
        df_original = df.copy()
        df_clean = clean_data(df)
        pd.testing.assert_frame_equal(df, df_original)


class TestFeatureEngineering:
    """Test feature engineering module."""
    
    def test_build_features_returns_correct_shapes(self):
        """Test that feature building returns correct shapes."""
        df = generate_data(n=100, seed=42)
        df = clean_data(df)
        X, y = build_features(df)
        
        assert len(X) == len(df)
        assert len(y) == len(df)
        assert X.shape[1] == 5  # 5 features
    
    def test_build_features_correct_columns(self):
        """Test that X has correct feature columns."""
        df = generate_data(n=100, seed=42)
        df = clean_data(df)
        X, y = build_features(df)
        
        expected_features = ['declared_income', 'property_count', 'director_flag', 
                           'late_filing_count', 'previous_penalty']
        assert list(X.columns) == expected_features
    
    def test_build_features_y_is_binary(self):
        """Test that target variable is binary."""
        df = generate_data(n=100, seed=42)
        df = clean_data(df)
        X, y = build_features(df)
        
        assert y.nunique() == 2
        assert set(y.unique()) == {0, 1}


class TestModelTraining:
    """Test model training module."""
    
    def test_train_models_returns_model(self):
        """Test that training returns a model object."""
        df = generate_data(n=100, seed=42)
        df = clean_data(df)
        X, y = build_features(df)
        
        model = train_models(X, y)
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_train_models_saves_file(self):
        """Test that model is saved to disk."""
        df = generate_data(n=100, seed=42)
        df = clean_data(df)
        X, y = build_features(df)
        
        train_models(X, y)
        assert os.path.exists('output/model/risk_model.pkl')
    
    def test_model_predictions_shape(self):
        """Test that model predictions have correct shape."""
        df = generate_data(n=100, seed=42)
        df = clean_data(df)
        X, y = build_features(df)
        
        model = train_models(X, y)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)


class TestModelEvaluation:
    """Test model evaluation module."""
    
    def test_evaluate_models_returns_metrics(self):
        """Test that evaluation returns required metrics."""
        df = generate_data(n=100, seed=42)
        df = clean_data(df)
        X, y = build_features(df)
        model = train_models(X, y)
        
        results = evaluate_models(model, X, y)
        
        assert 'auc' in results
        assert 'fpr' in results
        assert 'tpr' in results
        assert 'precision' in results
        assert 'recall' in results
    
    def test_evaluate_models_auc_range(self):
        """Test that AUC is in valid range [0, 1]."""
        df = generate_data(n=1000, seed=42)
        df = clean_data(df)
        X, y = build_features(df)
        model = train_models(X, y)
        
        results = evaluate_models(model, X, y)
        assert 0 <= results['auc'] <= 1
    
    def test_evaluate_models_probabilities_range(self):
        """Test that predicted probabilities are in [0, 1]."""
        df = generate_data(n=100, seed=42)
        df = clean_data(df)
        X, y = build_features(df)
        model = train_models(X, y)
        
        results = evaluate_models(model, X, y)
        probabilities = results['probabilities']
        
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)


# Clean up after tests
def teardown_module(module):
    """Clean up test artifacts."""
    # Remove test output files if needed
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
