from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report
import pandas as pd


def evaluate_models(model, X, y):
    """Evaluate the trained model and return performance metrics."""
    # Get predictions
    y_pred = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y, probs)
    
    # Get ROC curve data
    fpr, tpr, _ = roc_curve(y, probs)
    
    # Get Precision-Recall curve data
    precision, recall, _ = precision_recall_curve(y, probs)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Classification report
    report = classification_report(y, y_pred, output_dict=True)
    
    results = {
        'auc': auc,
        'fpr': fpr,
        'tpr': tpr,
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1': report['1']['f1-score'],
        'accuracy': report['accuracy'],
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred,
        'probabilities': probs
    }
    
    print(f"\n{'='*60}")
    print("MODEL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"AUC-ROC Score: {auc:.4f}")
    print(f"\nClassification Report:")
    print(f"  Precision (Risk=1): {report['1']['precision']:.4f}")
    print(f"  Recall (Risk=1):    {report['1']['recall']:.4f}")
    print(f"  F1-Score (Risk=1):  {report['1']['f1-score']:.4f}")
    print(f"  Accuracy:           {report['accuracy']:.4f}")
    print(f"{'='*60}\n")
    
    return results
