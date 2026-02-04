import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import os


def create_eda_charts(df):
    """Create exploratory data analysis visualizations."""
    os.makedirs("output/eda", exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Risk Flag Distribution
    fig, ax = plt.subplots()
    risk_counts = df['risk_flag'].value_counts()
    ax.bar(['Low Risk', 'High Risk'], risk_counts.values, color=['green', 'red'], alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title('Risk Flag Distribution')
    ax.set_xlabel('Risk Category')
    for i, v in enumerate(risk_counts.values):
        ax.text(i, v + 50, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/eda/risk_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Income Distribution
    fig, ax = plt.subplots()
    ax.hist(df['declared_income'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Declared Income (£)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Declared Income')
    ax.axvline(df['declared_income'].median(), color='red', linestyle='--', 
               label=f'Median: £{df["declared_income"].median():,.0f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig('output/eda/income_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Correlation Heatmap
    features = ['declared_income', 'property_count', 'director_flag', 
                'late_filing_count', 'previous_penalty', 'risk_flag']
    corr_matrix = df[features].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('output/eda/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Risk by Property Count
    fig, ax = plt.subplots()
    risk_by_property = df.groupby('property_count')['risk_flag'].mean()
    ax.bar(risk_by_property.index, risk_by_property.values, color='coral', alpha=0.7)
    ax.set_xlabel('Property Count')
    ax.set_ylabel('Risk Rate')
    ax.set_title('Risk Rate by Property Count')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig('output/eda/risk_by_property.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Boxplot of Income by Risk Flag
    fig, ax = plt.subplots()
    df.boxplot(column='declared_income', by='risk_flag', ax=ax, patch_artist=True)
    ax.set_xlabel('Risk Flag (0=Low, 1=High)')
    ax.set_ylabel('Declared Income (£)')
    ax.set_title('Income Distribution by Risk Category')
    plt.suptitle('')  # Remove default title
    plt.tight_layout()
    plt.savefig('output/eda/income_by_risk.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_model_evaluation_charts(model, X, y):
    """Create model evaluation visualizations."""
    os.makedirs("output/model", exist_ok=True)
    
    # Get predictions
    y_pred = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. ROC Curve
    fpr, tpr, thresholds = roc_curve(y, probs)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, probs)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Risk Classification Model')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/model/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y, probs)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, color='purple')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/model/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Low Risk', 'High Risk'],
                yticklabels=['Low Risk', 'High Risk'])
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('output/model/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Feature Importance
    feature_names = X.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(importances)), importances[indices], color='teal', alpha=0.7)
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance from Random Forest Model')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('output/model/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Prediction Probability Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(probs[y == 0], bins=50, alpha=0.5, label='Low Risk (Actual)', color='green')
    ax.hist(probs[y == 1], bins=50, alpha=0.5, label='High Risk (Actual)', color='red')
    ax.set_xlabel('Predicted Probability of High Risk')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Predicted Probabilities by Actual Risk Category')
    ax.legend()
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1, label='Default Threshold')
    plt.tight_layout()
    plt.savefig('output/model/probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
