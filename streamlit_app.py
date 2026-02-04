import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, classification_report, roc_auc_score

# Page config
st.set_page_config(
    page_title="Tax Compliance Risk Analytics Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔍 Tax Compliance Risk Analytics</h1>', unsafe_allow_html=True)
st.markdown("---")

# Load data and model
@st.cache_data
def load_data():
    """Load the synthetic taxpayer data."""
    if os.path.exists('data/raw/synthetic_taxpayers.csv'):
        return pd.read_csv('data/raw/synthetic_taxpayers.csv')
    else:
        st.warning("Data file not found. Please run `python main.py` first to generate data.")
        return None

@st.cache_resource
def load_model():
    """Load the trained model."""
    if os.path.exists('output/model/risk_model.pkl'):
        return joblib.load('output/model/risk_model.pkl')
    else:
        st.warning("Model file not found. Please run `python main.py` first to train the model.")
        return None

df = load_data()
model = load_model()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", [
    "📊 Overview",
    "🔎 Data Explorer",
    "📈 Model Performance",
    "⚠️ Risk Scoring",
    "🎯 Feature Importance",
    "🎚️ Threshold Analysis"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This dashboard demonstrates end-to-end tax compliance risk analysis using machine learning.

**Key Features:**
- Synthetic data generation
- Risk prediction modeling
- Interactive visualizations
- Operational threshold tuning
""")

# Check if data and model are loaded
if df is None or model is None:
    st.error("⚠️ Missing data or model files. Please run the pipeline first:")
    st.code("python main.py", language="bash")
    st.stop()

# Generate predictions
X = df[['declared_income', 'property_count', 'director_flag', 'late_filing_count', 'previous_penalty']]
y = df['risk_flag']
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

# Add predictions to dataframe
df['predicted_risk'] = y_pred
df['risk_score'] = y_proba

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page == "📊 Overview":
    st.header("📊 Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Taxpayers", f"{len(df):,}")
    with col2:
        st.metric("High Risk Cases", f"{df['risk_flag'].sum():,}")
    with col3:
        auc = roc_auc_score(y, y_proba)
        st.metric("Model AUC", f"{auc:.4f}")
    with col4:
        accuracy = (y == y_pred).mean()
        st.metric("Accuracy", f"{accuracy:.2%}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Project Scope")
        st.markdown("""
        This project demonstrates a complete data science workflow for tax compliance:
        
        **1. Data Generation**
        - Synthetic taxpayer data (10,000 records)
        - Realistic features and risk patterns
        
        **2. Exploratory Analysis**
        - Distribution analysis
        - Feature correlations
        - Risk pattern identification
        
        **3. ML Model Development**
        - Random Forest Classifier
        - Class-imbalance handling
        - Cross-validation
        
        **4. Model Evaluation**
        - ROC-AUC analysis
        - Precision-Recall curves
        - Feature importance
        
        **5. Operational Deployment**
        - Interactive dashboard
        - Threshold tuning
        - Risk scoring system
        """)
    
    with col2:
        st.subheader("🎯 Business Value")
        st.markdown("""
        **Key Benefits:**
        
        ✅ **Efficient Resource Allocation**
        - Target high-risk cases for investigation
        - Reduce manual review workload
        
        ✅ **Data-Driven Decisions**
        - Evidence-based risk assessment
        - Transparent scoring methodology
        
        ✅ **Scalable Solution**
        - Handles large taxpayer populations
        - Automated risk scoring
        
        ✅ **Continuous Improvement**
        - Model monitoring and retraining
        - Performance tracking
        
        **Key Highlights:**
        - Supervised learning for risk targeting ✓
        - Class-imbalance handling ✓
        - Reproducible pipeline ✓
        - Production-ready code ✓
        """)

# ============================================================================
# PAGE: DATA EXPLORER
# ============================================================================
elif page == "🔎 Data Explorer":
    st.header("🔎 Data Explorer")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Filters")
        
        income_range = st.slider(
            "Declared Income (£)",
            float(df['declared_income'].min()),
            float(df['declared_income'].max()),
            (float(df['declared_income'].min()), float(df['declared_income'].max()))
        )
        
        risk_filter = st.multiselect(
            "Risk Flag",
            options=[0, 1],
            default=[0, 1],
            format_func=lambda x: "High Risk" if x == 1 else "Low Risk"
        )
        
        property_range = st.slider(
            "Property Count",
            int(df['property_count'].min()),
            int(df['property_count'].max()),
            (int(df['property_count'].min()), int(df['property_count'].max()))
        )
    
    # Apply filters
    filtered_df = df[
        (df['declared_income'] >= income_range[0]) &
        (df['declared_income'] <= income_range[1]) &
        (df['risk_flag'].isin(risk_filter)) &
        (df['property_count'] >= property_range[0]) &
        (df['property_count'] <= property_range[1])
    ]
    
    with col2:
        st.subheader(f"Taxpayer Records ({len(filtered_df):,} records)")
        
        # Display dataframe
        display_cols = ['taxpayer_id', 'declared_income', 'property_count', 
                       'director_flag', 'late_filing_count', 'previous_penalty', 
                       'risk_flag', 'risk_score']
        st.dataframe(
            filtered_df[display_cols].style.format({
                'declared_income': '£{:,.2f}',
                'risk_score': '{:.4f}'
            }).background_gradient(subset=['risk_score'], cmap='RdYlGn_r'),
            height=400
        )
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(filtered_df[['declared_income', 'property_count', 'late_filing_count', 
                                  'risk_score']].describe().T.style.format("{:.2f}"))

# ============================================================================
# PAGE: MODEL PERFORMANCE
# ============================================================================
elif page == "📈 Model Performance":
    st.header("📈 Model Performance")
    
    tab1, tab2, tab3, tab4 = st.tabs(["ROC Curve", "Precision-Recall", "Confusion Matrix", "Metrics"])
    
    with tab1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={auc:.4f})',
                                line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                                line=dict(color='gray', width=2, dash='dash')))
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            title='Receiver Operating Characteristic (ROC) Curve',
            hovermode='x',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"""
        **Interpretation:** The model achieves an AUC of {auc:.4f}, indicating excellent discrimination 
        between high-risk and low-risk taxpayers. An AUC of 1.0 is perfect, while 0.5 is random.
        """)
    
    with tab2:
        st.subheader("Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(y, y_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', 
                                line=dict(color='purple', width=3)))
        fig.update_layout(
            xaxis_title='Recall (Sensitivity)',
            yaxis_title='Precision',
            title='Precision-Recall Curve',
            hovermode='x',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Interpretation:** This curve shows the trade-off between precision (% of flagged cases 
        that are truly high-risk) and recall (% of high-risk cases that are caught). Use the 
        Threshold Analysis page to optimize this trade-off.
        """)
    
    with tab3:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Low Risk', 'Predicted High Risk'],
            y=['Actual Low Risk', 'Actual High Risk'],
            text=cm,
            texttemplate='%{text}',
            colorscale='Blues',
            showscale=True
        ))
        fig.update_layout(
            title='Confusion Matrix',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("True Negatives", f"{cm[0, 0]:,}")
        with col2:
            st.metric("False Positives", f"{cm[0, 1]:,}")
        with col3:
            st.metric("False Negatives", f"{cm[1, 0]:,}")
        with col4:
            st.metric("True Positives", f"{cm[1, 1]:,}")
    
    with tab4:
        st.subheader("Classification Report")
        report = classification_report(y, y_pred, output_dict=True, 
                                      target_names=['Low Risk', 'High Risk'])
        
        report_df = pd.DataFrame(report).T
        report_df = report_df[report_df.index.isin(['Low Risk', 'High Risk', 'accuracy', 'macro avg', 'weighted avg'])]
        
        st.dataframe(report_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', axis=0),
                    use_container_width=True)
        
        st.markdown("---")
        st.subheader("Key Metrics Explained")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Precision:** Of all cases we flag as high-risk, what % are actually high-risk?
            - High precision = fewer false alarms
            - Important when investigation capacity is limited
            """)
            
            st.markdown("""
            **Recall (Sensitivity):** Of all actual high-risk cases, what % do we catch?
            - High recall = catching more true high-risk cases
            - Important when missing high-risk cases is costly
            """)
        
        with col2:
            st.markdown("""
            **F1-Score:** Harmonic mean of precision and recall
            - Balances both metrics
            - Useful when you need a single metric
            """)
            
            st.markdown("""
            **Accuracy:** Overall % of correct predictions
            - Can be misleading with imbalanced classes
            - Use with caution in risk modeling
            """)

# ============================================================================
# PAGE: RISK SCORING
# ============================================================================
elif page == "⚠️ Risk Scoring":
    st.header("⚠️ Risk Scoring")
    
    st.markdown("""
    Enter taxpayer details to get a real-time risk assessment from the trained model.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Features")
        
        declared_income = st.number_input("Declared Income (£)", min_value=0.0, value=50000.0, step=1000.0)
        property_count = st.number_input("Property Count", min_value=0, value=1, step=1)
        director_flag = st.selectbox("Director Flag", options=[0, 1], 
                                     format_func=lambda x: "Yes" if x == 1 else "No")
        late_filing_count = st.number_input("Late Filing Count", min_value=0, value=0, step=1)
        previous_penalty = st.selectbox("Previous Penalty", options=[0, 1],
                                       format_func=lambda x: "Yes" if x == 1 else "No")
        
        if st.button("Calculate Risk Score", type="primary"):
            # Make prediction
            input_data = pd.DataFrame({
                'declared_income': [declared_income],
                'property_count': [property_count],
                'director_flag': [director_flag],
                'late_filing_count': [late_filing_count],
                'previous_penalty': [previous_penalty]
            })
            
            risk_score = model.predict_proba(input_data)[0, 1]
            risk_class = "HIGH RISK" if risk_score > 0.5 else "LOW RISK"
            risk_color = "red" if risk_score > 0.5 else "green"
            
            with col2:
                st.subheader("Risk Assessment")
                
                st.markdown(f"""
                <div style='background-color: {risk_color}; padding: 2rem; border-radius: 10px; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>Risk Score: {risk_score:.2%}</h2>
                    <h3 style='color: white; margin: 0.5rem 0 0 0;'>{risk_class}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Feature contribution
                st.subheader("Feature Contributions")
                feature_importance = model.feature_importances_
                feature_names = ['Declared Income', 'Property Count', 'Director Flag', 
                               'Late Filing', 'Previous Penalty']
                
                fig = go.Figure(go.Bar(
                    x=feature_importance,
                    y=feature_names,
                    orientation='h',
                    marker_color='teal'
                ))
                fig.update_layout(
                    title='Feature Importance in Model',
                    xaxis_title='Importance',
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **Next Steps:**
                - High risk scores (>0.5) warrant further investigation
                - Review taxpayer history and supporting documentation
                - Consider threshold adjustments based on operational capacity
                """)

# ============================================================================
# PAGE: FEATURE IMPORTANCE
# ============================================================================
elif page == "🎯 Feature Importance":
    st.header("🎯 Feature Importance Analysis")
    
    feature_names = ['Declared Income', 'Property Count', 'Director Flag', 
                    'Late Filing Count', 'Previous Penalty']
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Feature Importance Rankings")
        
        fig = go.Figure(go.Bar(
            x=importances[indices],
            y=[feature_names[i] for i in indices],
            orientation='h',
            marker_color='teal',
            text=[f"{importances[i]:.4f}" for i in indices],
            textposition='auto'
        ))
        fig.update_layout(
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 3 Features")
        for i, idx in enumerate(indices[:3], 1):
            st.metric(
                f"#{i} {feature_names[idx]}",
                f"{importances[idx]:.4f}"
            )
        
        st.markdown("---")
        st.info("""
        **Feature Importance** measures how much each feature contributes 
        to the model's predictions. Higher values indicate more influential features.
        """)
    
    st.markdown("---")
    
    # Feature distributions by risk
    st.subheader("Feature Distributions by Risk Category")
    
    feature_to_plot = st.selectbox("Select Feature", feature_names)
    feature_col = ['declared_income', 'property_count', 'director_flag', 
                   'late_filing_count', 'previous_penalty'][feature_names.index(feature_to_plot)]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Low Risk', 'High Risk'))
    
    # Low risk
    low_risk_data = df[df['risk_flag'] == 0][feature_col]
    fig.add_trace(
        go.Histogram(x=low_risk_data, name='Low Risk', marker_color='green', opacity=0.7),
        row=1, col=1
    )
    
    # High risk
    high_risk_data = df[df['risk_flag'] == 1][feature_col]
    fig.add_trace(
        go.Histogram(x=high_risk_data, name='High Risk', marker_color='red', opacity=0.7),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text=feature_to_plot, row=1, col=1)
    fig.update_xaxes(title_text=feature_to_plot, row=1, col=2)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: THRESHOLD ANALYSIS
# ============================================================================
elif page == "🎚️ Threshold Analysis":
    st.header("🎚️ Threshold Analysis")
    
    st.markdown("""
    Adjust the classification threshold to optimize for different operational objectives.
    The default threshold is 0.5, but you can tune it based on investigation capacity and priorities.
    """)
    
    threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.01)
    
    # Calculate metrics at threshold
    y_pred_threshold = (y_proba >= threshold).astype(int)
    
    # Metrics
    precision, recall, thresholds = precision_recall_curve(y, y_proba)
    
    # Find closest threshold
    idx = np.argmin(np.abs(thresholds - threshold))
    precision_at_threshold = precision[idx]
    recall_at_threshold = recall[idx]
    
    # Calculate other metrics
    from sklearn.metrics import f1_score, accuracy_score
    f1 = f1_score(y, y_pred_threshold)
    accuracy = accuracy_score(y, y_pred_threshold)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred_threshold)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Precision", f"{precision_at_threshold:.2%}")
    with col2:
        st.metric("Recall", f"{recall_at_threshold:.2%}")
    with col3:
        st.metric("F1-Score", f"{f1:.2%}")
    with col4:
        st.metric("Accuracy", f"{accuracy:.2%}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Precision-Recall Trade-off")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall, y=precision, mode='lines',
            name='PR Curve', line=dict(color='purple', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[recall_at_threshold], y=[precision_at_threshold],
            mode='markers', name=f'Current Threshold ({threshold:.2f})',
            marker=dict(color='red', size=15, symbol='star')
        ))
        fig.update_layout(
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=400,
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Confusion Matrix at Current Threshold")
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Low', 'Predicted High'],
            y=['Actual Low', 'Actual High'],
            text=cm,
            texttemplate='%{text}',
            colorscale='Blues',
            showscale=True
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Operational impact
    st.subheader("Operational Impact Analysis")
    
    total_flagged = y_pred_threshold.sum()
    total_high_risk = y.sum()
    caught_high_risk = ((y == 1) & (y_pred_threshold == 1)).sum()
    missed_high_risk = ((y == 1) & (y_pred_threshold == 0)).sum()
    false_alarms = ((y == 0) & (y_pred_threshold == 1)).sum()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cases Flagged for Investigation", f"{total_flagged:,}")
        st.metric("True High-Risk Cases Caught", f"{caught_high_risk:,}")
    
    with col2:
        st.metric("Missed High-Risk Cases", f"{missed_high_risk:,}")
        st.metric("False Alarms", f"{false_alarms:,}")
    
    with col3:
        investigation_rate = total_flagged / len(df)
        st.metric("Investigation Rate", f"{investigation_rate:.1%}")
        detection_rate = caught_high_risk / total_high_risk if total_high_risk > 0 else 0
        st.metric("Detection Rate", f"{detection_rate:.1%}")
    
    st.info(f"""
    **Recommendation:** 
    - **Lower threshold** (e.g., 0.3): Catch more high-risk cases but investigate more total cases
    - **Higher threshold** (e.g., 0.7): Focus only on highest-risk cases, fewer investigations
    - **Current setting** ({threshold:.2f}): Flagging {total_flagged:,} cases ({investigation_rate:.1%} of population)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>Tax Compliance Risk Analytics Dashboard | Built with Streamlit & Scikit-learn</p>
    <p>For demonstration purposes only - using synthetic data</p>
</div>
""", unsafe_allow_html=True)
