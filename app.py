"""
Streamlit Dashboard for Precision Indoor Cultivation
====================================================
Multi-page dashboard for pH prediction, NHI estimation, and plant growth stage classification.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import dataset insights module
try:
    from data_insights import analyze_dataset_insights, generate_dataset_recommendations
except ImportError:
    # Fallback if module not available
    def analyze_dataset_insights(df):
        return {'pH': {}, 'NHI': {}, 'growth_stage': {}}
    def generate_dataset_recommendations(pred_value, metric_type, df, insights):
        return []

# Page configuration
st.set_page_config(
    page_title="Precision Indoor Cultivation Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E7D32;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "NPK_New Dataset.xlsx")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")

# Model loading functions
@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    try:
        # Load scalers
        if os.path.exists(os.path.join(MODELS_DIR, "ph_scaler.pkl")):
            models['ph_scaler'] = joblib.load(os.path.join(MODELS_DIR, "ph_scaler.pkl"))
        if os.path.exists(os.path.join(MODELS_DIR, "nhi_scaler.pkl")):
            models['nhi_scaler'] = joblib.load(os.path.join(MODELS_DIR, "nhi_scaler.pkl"))
        if os.path.exists(os.path.join(MODELS_DIR, "growth_stage_scaler.pkl")):
            models['growth_stage_scaler'] = joblib.load(os.path.join(MODELS_DIR, "growth_stage_scaler.pkl"))
        
        # Load encoders
        if os.path.exists(os.path.join(MODELS_DIR, "growth_stage_encoder.pkl")):
            models['growth_stage_encoder'] = joblib.load(os.path.join(MODELS_DIR, "growth_stage_encoder.pkl"))
        
        # Load TabNet models
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
            
            # Check for .zip files or directories (TabNet saves as .zip)
            ph_model_path = os.path.join(MODELS_DIR, "ph_tabnet.zip")
            if not os.path.exists(ph_model_path):
                ph_model_path = os.path.join(MODELS_DIR, "ph_tabnet")
            if os.path.exists(ph_model_path):
                models['ph_tabnet'] = TabNetRegressor()
                models['ph_tabnet'].load_model(ph_model_path)
            
            nhi_model_path = os.path.join(MODELS_DIR, "nhi_tabnet.zip")
            if not os.path.exists(nhi_model_path):
                nhi_model_path = os.path.join(MODELS_DIR, "nhi_tabnet")
            if os.path.exists(nhi_model_path):
                models['nhi_tabnet'] = TabNetRegressor()
                models['nhi_tabnet'].load_model(nhi_model_path)
            
            growth_model_path = os.path.join(MODELS_DIR, "growth_stage_tabnet.zip")
            if not os.path.exists(growth_model_path):
                growth_model_path = os.path.join(MODELS_DIR, "growth_stage_tabnet")
            if os.path.exists(growth_model_path):
                models['growth_stage_tabnet'] = TabNetClassifier()
                models['growth_stage_tabnet'].load_model(growth_model_path)
        except Exception as e:
            st.warning(f"TabNet models not loaded: {e}")
            
    except Exception as e:
        st.error(f"Error loading models: {e}")
    
    return models

@st.cache_data
def load_data():
    """Load dataset"""
    try:
        if os.path.exists(DATA_PATH):
            df = pd.read_excel(DATA_PATH)
            return df
        else:
            st.error(f"Dataset not found at {DATA_PATH}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def get_dataset_insights(df):
    """Get dataset-specific insights"""
    if df is not None:
        try:
            return analyze_dataset_insights(df.copy())
        except Exception as e:
            st.warning(f"Could not generate dataset insights: {e}")
            return {'pH': {}, 'NHI': {}, 'growth_stage': {}}
    return {'pH': {}, 'NHI': {}, 'growth_stage': {}}

# Sidebar navigation
st.sidebar.title("🌱 Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    [" Overview", " pH Prediction", " NHI Estimation", " Growth Stage Classification", " Data Insights"]
)

# Load data and models
df = load_data()
models = load_models()
dataset_insights = get_dataset_insights(df) if df is not None else {}

# ===============================
# Overview Page
# ===============================
if page == " Overview":
    st.markdown('<div class="main-header"> Precision Indoor Cultivation Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Precision Indoor Cultivation Analytics Platform
    
    This dashboard provides comprehensive tools for analyzing and predicting key metrics in precision agriculture:
    
    - **pH Prediction**: Predict soil pH levels based on sensor data
    - **NHI Estimation**: Estimate Nutrient Health Index for optimal plant nutrition
    - **Growth Stage Classification**: Classify plant growth stages using machine learning
    - **Data Insights**: Explore sensor data interconnections and patterns
    """)
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(df))
        
        with col2:
            st.metric("Features", len(df.columns))
        
        with col3:
            if 'pH' in df.columns:
                st.metric("Avg pH", f"{df['pH'].mean():.2f}")
        
        with col4:
            if 'NHI' in df.columns:
                st.metric("Avg NHI", f"{df['NHI'].mean():.2f}")
        
        st.markdown("---")
        
        # Quick stats
        st.subheader(" Dataset Overview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Model status
        st.subheader(" Model Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ph_status = " Loaded" if 'ph_tabnet' in models else " Not Available"
            st.markdown(f"**pH Model**: {ph_status}")
        
        with col2:
            nhi_status = "Loaded" if 'nhi_tabnet' in models else " Not Available"
            st.markdown(f"**NHI Model**: {nhi_status}")
        
        with col3:
            stage_status = " Loaded" if 'growth_stage_tabnet' in models else " Not Available"
            st.markdown(f"**Growth Stage Model**: {stage_status}")
        
        # Dataset-Specific Findings Section
        st.markdown("---")
        st.subheader("🔬 Dataset-Specific Findings & Insights")
        st.markdown("""
        **This section presents insights derived directly from your dataset analysis**, 
        distinguishing data-driven findings from general research recommendations.
        """)
        
        if dataset_insights:
            # pH Findings
            if 'pH' in dataset_insights and dataset_insights['pH']:
                with st.expander(" pH Dataset Analysis", expanded=True):
                    ph_ins = dataset_insights['pH']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean pH", f"{ph_ins.get('mean', 0):.2f}")
                        st.metric("Optimal Range Coverage", f"{ph_ins.get('optimal_range_pct', 0):.1f}%")
                    with col2:
                        st.metric("Acidic Samples", f"{ph_ins.get('acidic_pct', 0):.1f}%")
                        st.metric("Alkaline Samples", f"{ph_ins.get('alkaline_pct', 0):.1f}%")
                    
                    if ph_ins.get('recommendations'):
                        st.markdown("**Key Dataset Patterns:**")
                        for rec in ph_ins['recommendations']:
                            st.info(rec)
            
            # NHI Findings
            if 'NHI' in dataset_insights and dataset_insights['NHI']:
                with st.expander(" NHI Dataset Analysis", expanded=True):
                    nhi_ins = dataset_insights['NHI']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean NHI", f"{nhi_ins.get('mean', 0):.2f}")
                    with col2:
                        st.metric("Optimal (≥80)", f"{nhi_ins.get('optimal_pct', 0):.1f}%")
                    with col3:
                        st.metric("Good (60-80)", f"{nhi_ins.get('good_pct', 0):.1f}%")
                    with col4:
                        st.metric("Critical (<30)", f"{nhi_ins.get('critical_pct', 0):.1f}%")
                    
                    if nhi_ins.get('recommendations'):
                        st.markdown("**Key Dataset Patterns:**")
                        for rec in nhi_ins['recommendations']:
                            st.info(rec)
            
            # Growth Stage Findings
            if 'growth_stage' in dataset_insights and dataset_insights['growth_stage']:
                with st.expander("🌿 Growth Stage Dataset Analysis", expanded=True):
                    stage_ins = dataset_insights['growth_stage']
                    if stage_ins.get('distribution'):
                        st.markdown("**Stage Distribution:**")
                        for stage, count in stage_ins['distribution'].items():
                            pct = stage_ins['distribution_pct'].get(stage, 0)
                            st.metric(f"{stage}", f"{count} samples ({pct:.1f}%)")
                    
                    if stage_ins.get('recommendations'):
                        st.markdown("**Key Dataset Patterns:**")
                        for rec in stage_ins['recommendations']:
                            st.info(rec)
        else:
            st.info(" **Tip**: Run the training scripts (pH.py, NHI.py, plantgrowthstageclassification.py) to generate dataset-specific insights.")
        
        st.markdown("---")
        st.markdown("""
        ###  About Data-Driven Insights
        
        ** All insights and recommendations are derived from YOUR dataset**:
        - Patterns, correlations, and distributions discovered in YOUR data
        - Context relative to YOUR dataset's characteristics
        - Percentile rankings showing where predictions fall in YOUR dataset
        - Recommendations based on patterns found in YOUR specific cultivation environment
        - Stage-specific nutrient levels observed in YOUR dataset
        """)

# ===============================
# pH Prediction Page
# ===============================
elif page == " pH Prediction":
    st.markdown('<div class="main-header"> pH Prediction</div>', unsafe_allow_html=True)
    
    if 'ph_tabnet' not in models or 'ph_scaler' not in models:
        st.warning(" pH models not loaded. Please run pH.py first to train the models.")
    else:
        st.markdown("""
        ### Predict soil pH levels from sensor readings
        
        Enter sensor values below to get pH predictions using our TabNet model.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Sensor Input")
            col1a, col2a = st.columns(2)
            
            with col1a:
                nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=100.0, value=42.0, step=0.1)
                phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
                potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=100.0, value=38.0, step=0.1)
            
            with col2a:
                conductivity = st.number_input("Conductivity", min_value=0.0, max_value=10.0, value=1.2, step=0.1)
                moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=28.0, step=0.1)
                temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=26.0, step=0.1)
            
            if st.button("🔮 Predict pH", type="primary"):
                sensor_input = {
                    'nitrogen': nitrogen,
                    'phosphorus': phosphorus,
                    'potassium': potassium,
                    'conductivity': conductivity,
                    'moisture': moisture,
                    'temperature': temperature
                }
                
                # Prepare input
                features = ['nitrogen', 'phosphorus', 'potassium', 'conductivity', 'moisture', 'temperature']
                df_input = pd.DataFrame([sensor_input])[features]
                X_scaled = models['ph_scaler'].transform(df_input)
                
                # Predict
                pred_ph = models['ph_tabnet'].predict(X_scaled)[0][0]
                
                st.success(f"### Predicted pH: **{pred_ph:.2f}**")
                
                # Data-driven insights only
                st.markdown("###  Data-Driven Insights & Recommendations")
                st.markdown("*Based on analysis of your dataset*")
                
                # Dataset-specific recommendations
                if df is not None and 'pH' in df.columns:
                    data_recs = generate_dataset_recommendations(pred_ph, 'pH', df, dataset_insights)
                    for rec in data_recs:
                        st.info(rec)
                    
                    # Show dataset insights if available
                    if 'pH' in dataset_insights and dataset_insights['pH']:
                        ph_insights = dataset_insights['pH']
                        if ph_insights.get('recommendations'):
                            st.markdown("**Dataset Patterns Found:**")
                            for rec in ph_insights['recommendations']:
                                st.markdown(f"- {rec}")
                    
                    # Data-driven recommendations based on prediction
                    ph_mean = df['pH'].mean()
                    ph_std = df['pH'].std()
                    
                    st.markdown("###  Data-Driven Recommendations")
                    
                    if pred_ph < 6.0:
                        st.error(" **Low pH Detected** (Acidic Soil)")
                        acidic_pct = (df['pH'] < 6.0).sum() / len(df) * 100
                        st.markdown(f"""
                        **Based on Your Dataset Analysis**:
                        - **Dataset Context**: {acidic_pct:.1f}% of samples in your dataset have pH < 6.0
                        - **Your Prediction**: pH {pred_ph:.2f} is {'below' if pred_ph < ph_mean else 'above'} the dataset mean ({ph_mean:.2f})
                        - **Recommendation**: Based on {acidic_pct:.0f}% of your samples being acidic, consider pH adjustment strategies
                        - **Pattern**: Your dataset shows pH ranges from {df['pH'].min():.2f} to {df['pH'].max():.2f}
                        """)
                    elif pred_ph > 7.5:
                        st.warning(" **High pH Detected** (Alkaline Soil)")
                        alkaline_pct = (df['pH'] > 7.5).sum() / len(df) * 100
                        st.markdown(f"""
                        **Based on Your Dataset Analysis**:
                        - **Dataset Context**: {alkaline_pct:.1f}% of samples in your dataset have pH > 7.5
                        - **Your Prediction**: pH {pred_ph:.2f} is {'above' if pred_ph > ph_mean else 'below'} the dataset mean ({ph_mean:.2f})
                        - **Recommendation**: Based on {alkaline_pct:.0f}% of your samples being alkaline, consider pH adjustment strategies
                        - **Pattern**: Your dataset shows pH ranges from {df['pH'].min():.2f} to {df['pH'].max():.2f}
                        """)
                    else:
                        st.success(" **pH in Optimal Range** (6.0-7.5)")
                        optimal_pct = (df['pH'].between(6.0, 7.5).sum() / len(df) * 100)
                        st.markdown(f"""
                        **Based on Your Dataset Analysis**:
                        - **Dataset Context**: {optimal_pct:.1f}% of samples in your dataset are in optimal range (6.0-7.5)
                        - **Your Prediction**: pH {pred_ph:.2f} aligns with {optimal_pct:.0f}% of your dataset samples
                        - **Recommendation**: Maintain current conditions as they match optimal patterns in your dataset
                        - **Pattern**: Your dataset shows pH ranges from {df['pH'].min():.2f} to {df['pH'].max():.2f}
                        """)
                else:
                    st.info("Run pH.py to generate dataset insights")
        
        with col2:
            st.subheader(" Model Performance")
            metrics_path = os.path.join(METRICS_DIR, "ph_model_comparison.csv")
            if os.path.exists(metrics_path):
                metrics_df = pd.read_csv(metrics_path)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Feature importance
                imp_path = os.path.join(METRICS_DIR, "ph_feature_importance.csv")
                if os.path.exists(imp_path):
                    imp_df = pd.read_csv(imp_path)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.barplot(data=imp_df, x='Importance', y='Feature', ax=ax, palette='viridis')
                    ax.set_title('Feature Importance')
                    st.pyplot(fig)
            else:
                st.info("Run pH.py to generate metrics")
        
        # Visualizations
        if df is not None and 'pH' in df.columns:
            st.markdown("---")
            st.subheader(" Data Visualizations & Insights")
            
            # Create tabs for different visualization categories
            tab1, tab2, tab3, tab4, tab5 = st.tabs([" Distribution & Correlations", "Feature Relationships", "📊 Model Performance", "🔍 Residual Analysis", "🌐 Clusters & Global Insights"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    # pH Distribution
                    plot_path = os.path.join(PLOTS_DIR, "ph_distribution.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="pH Distribution", use_container_width=True)
                    else:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.histplot(df['pH'], kde=True, bins=30, ax=ax)
                        ax.set_title('pH Distribution')
                        ax.set_xlabel('pH')
                        st.pyplot(fig)
                
                with col2:
                    # Correlation Matrix
                    plot_path = os.path.join(PLOTS_DIR, "ph_correlation_matrix.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="Sensor Correlation Matrix", use_container_width=True)
                    else:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        corr_cols = ['pH', 'nitrogen', 'phosphorus', 'potassium', 'conductivity', 'moisture', 'temperature']
                        corr_data = df[corr_cols].corr()
                        sns.heatmap(corr_data, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                        ax.set_title('Correlation Matrix')
                        st.pyplot(fig)
            
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    # pH vs Nutrients
                    plot_path = os.path.join(PLOTS_DIR, "ph_vs_nutrients.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="pH vs NPK Nutrients", use_container_width=True)
                    else:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        nutrients = ["nitrogen", "phosphorus", "potassium"]
                        for i, col in enumerate(nutrients):
                            plt.subplot(1, 3, i+1)
                            sns.scatterplot(x=df[col], y=df["pH"], alpha=0.4)
                            plt.xlabel(col.capitalize())
                            plt.ylabel("pH")
                        plt.tight_layout()
                        st.pyplot(fig)
                
                with col2:
                    # Environmental Factors
                    plot_path = os.path.join(PLOTS_DIR, "ph_environmental_factors.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="pH vs Environmental Factors", use_container_width=True)
                    else:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        plt.subplot(1, 2, 1)
                        sns.scatterplot(x=df["conductivity"], y=df["pH"], alpha=0.4)
                        plt.title("pH vs Conductivity")
                        plt.subplot(1, 2, 2)
                        sns.scatterplot(x=df["moisture"], y=df["pH"], alpha=0.4)
                        plt.title("pH vs Moisture")
                        plt.tight_layout()
                        st.pyplot(fig)
            
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    # Model Comparison
                    plot_path = os.path.join(PLOTS_DIR, "ph_model_comparison.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="Model Performance Comparison", use_container_width=True)
                    else:
                        st.info("Model comparison plot not available")
                
                with col2:
                    # Predictions Scatter
                    plot_path = os.path.join(PLOTS_DIR, "ph_predictions_scatter.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="Prediction Accuracy (Actual vs Predicted)", use_container_width=True)
                    else:
                        st.info("Prediction scatter plot not available")
            
            with tab4:
                col1, col2 = st.columns(2)
                with col1:
                    # Residual Analysis
                    plot_path = os.path.join(PLOTS_DIR, "ph_residual_analysis.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="Residual Analysis - TabNet", use_container_width=True)
                    else:
                        st.info("Residual analysis plot not available")
                
                with col2:
                    # Feature Importance
                    plot_path = os.path.join(PLOTS_DIR, "ph_feature_importance.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="Feature Importance for pH Prediction", use_container_width=True)
                    else:
                        st.info("Feature importance plot not available")
            
            with tab5:
                st.subheader("Clustering Analysis & Global Insights")
                col1, col2 = st.columns(2)
                
                with col1:
                    # K-means clustering visualization
                    if df is not None:
                        try:
                            # Use available features
                            available_features = ['nitrogen', 'phosphorus', 'potassium', 'conductivity', 'moisture', 'temperature']
                            features_to_use = [f for f in available_features if f in df.columns]
                            
                            if len(features_to_use) >= 3:  # Need at least 3 features
                                df_cluster = df[features_to_use].dropna()
                                
                                if len(df_cluster) > 100:  # Need sufficient data points
                                    scaler_cluster = StandardScaler()
                                    X_cluster = scaler_cluster.fit_transform(df_cluster[features_to_use])
                                    
                                    # Apply PCA for 2D visualization
                                    pca = PCA(n_components=2)
                                    X_pca = pca.fit_transform(X_cluster)
                                    
                                    # K-means clustering
                                    n_clusters = min(5, max(2, len(df_cluster) // 200))
                                    
                                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                                    clusters = kmeans.fit_predict(X_cluster)
                                    
                                    # Color by pH if available, otherwise by cluster
                                    if 'pH' in df.columns:
                                        ph_values = df.loc[df_cluster.index, 'pH'].values
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=ph_values, cmap='coolwarm', alpha=0.6, s=20)
                                        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                                        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                                        ax.set_title('pH Data Clusters (Colored by pH)')
                                        plt.colorbar(scatter, ax=ax, label='pH')
                                    else:
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=20)
                                        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                                        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                                        ax.set_title('pH Data Clusters (K-means)')
                                        plt.colorbar(scatter, ax=ax, label='Cluster')
                                    
                                    st.pyplot(fig)
                                    
                                    st.info(f"**Insight**: Data grouped into {n_clusters} distinct clusters, revealing soil condition patterns.")
                                else:
                                    st.warning("Insufficient data points for clustering. Need at least 100 samples.")
                            else:
                                st.warning(f"Insufficient features available. Found: {features_to_use}")
                        except Exception as e:
                            st.error(f"Clustering visualization error: {str(e)}")
                    else:
                        st.warning("Dataset not loaded.")
                
                with col2:
                    # Global statistics and insights
                    if df is not None and 'pH' in df.columns:
                        st.markdown("###  Global pH Statistics")
                        ph_stats = df['pH'].describe()
                        st.dataframe(ph_stats, use_container_width=True)
                        
                        st.markdown("###  Key Insights")
                        ph_mean = df['pH'].mean()
                        ph_std = df['pH'].std()
                        ph_min = df['pH'].min()
                        ph_max = df['pH'].max()
                        optimal_pct = (df['pH'].between(6.0, 7.5).sum() / len(df) * 100)
                        status = 'Optimal' if 6.0 <= ph_mean <= 7.5 else 'Needs Attention'
                        variability = 'Low' if ph_std < 0.5 else 'Moderate' if ph_std < 1.0 else 'High'
                        
                        st.markdown(f"""
                        - **Mean pH**: {ph_mean:.2f} - {status}
                        - **pH Range**: {ph_min:.2f} - {ph_max:.2f}
                        - **Variability**: {variability} (σ = {ph_std:.2f})
                        - **Optimal Range Coverage**: {optimal_pct:.1f}% of samples
                        """)
                    else:
                        st.warning("pH data not available for statistics.")

# ===============================
# NHI Estimation Page
# ===============================
elif page == " NHI Estimation":
    st.markdown('<div class="main-header"> Nutrient Health Index (NHI) Estimation</div>', unsafe_allow_html=True)
    
    if 'nhi_tabnet' not in models or 'nhi_scaler' not in models:
        st.warning(" NHI models not loaded. Please run NHI.py first to train the models.")
    else:
        st.markdown("""
        ### Estimate Nutrient Health Index from sensor data
        
        NHI is a composite metric indicating overall nutrient health status (0-100 scale).
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Sensor Input")
            col1a, col2a = st.columns(2)
            
            with col1a:
                nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=100.0, value=42.0, step=0.1, key='nhi_n')
                phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, max_value=100.0, value=30.0, step=0.1, key='nhi_p')
                potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=100.0, value=38.0, step=0.1, key='nhi_k')
            
            with col2a:
                conductivity = st.number_input("Conductivity", min_value=0.0, max_value=10.0, value=1.2, step=0.1, key='nhi_cond')
                moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=28.0, step=0.1, key='nhi_moist')
                temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=26.0, step=0.1, key='nhi_temp')
                ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1, key='nhi_ph')
            
            if st.button(" Predict NHI", type="primary"):
                sensor_input = {
                    'nitrogen': nitrogen,
                    'phosphorus': phosphorus,
                    'potassium': potassium,
                    'conductivity': conductivity,
                    'moisture': moisture,
                    'temperature': temperature,
                    'pH': ph
                }
                
                features = ['nitrogen', 'phosphorus', 'potassium', 'conductivity', 'moisture', 'temperature', 'pH']
                df_input = pd.DataFrame([sensor_input])[features]
                X_scaled = models['nhi_scaler'].transform(df_input)
                
                pred_nhi = models['nhi_tabnet'].predict(X_scaled)[0][0]
                # Convert to Python float for Streamlit progress bar
                pred_nhi_float = float(pred_nhi)
                
                st.success(f"### Predicted NHI: **{pred_nhi_float:.2f}**")
                
                # Progress bar (needs Python float, not numpy float32)
                st.progress(pred_nhi_float / 100.0)
                
                # Data-driven insights only
                st.markdown("###  Data-Driven Insights & Recommendations")
                st.markdown("*Based on analysis of your dataset*")
                
                # Dataset-specific recommendations
                if df is not None:
                    # Calculate NHI if not present
                    if 'NHI' not in df.columns:
                        if all(col in df.columns for col in ['nitrogen', 'phosphorus', 'potassium']):
                            df['NHI'] = (
                                0.4 * (df['nitrogen'] / df['nitrogen'].max()) +
                                0.35 * (df['phosphorus'] / df['phosphorus'].max()) +
                                0.25 * (df['potassium'] / df['potassium'].max())
                            ) * 100
                    
                    if 'NHI' in df.columns:
                        data_recs = generate_dataset_recommendations(pred_nhi_float, 'NHI', df, dataset_insights)
                        for rec in data_recs:
                            st.info(rec)
                        
                        # Show dataset insights if available
                        if 'NHI' in dataset_insights and dataset_insights['NHI']:
                            nhi_insights = dataset_insights['NHI']
                            if nhi_insights.get('recommendations'):
                                st.markdown("**Dataset Patterns Found:**")
                                for rec in nhi_insights['recommendations']:
                                    st.markdown(f"- {rec}")
                        
                        # Data-driven recommendations based on prediction
                        nhi_mean = df['NHI'].mean()
                        nhi_std = df['NHI'].std()
                        
                        st.markdown("###  Data-Driven Recommendations")
                        
                        if pred_nhi < 30:
                            st.error(" **Critical: Very Low Nutrient Health**")
                            critical_pct = (df['NHI'] < 30).sum() / len(df) * 100
                            st.markdown(f"""
                            **Based on Your Dataset Analysis**:
                            - **Dataset Context**: {critical_pct:.1f}% of samples in your dataset have NHI < 30
                            - **Your Prediction**: NHI {pred_nhi_float:.2f} is {'below' if pred_nhi_float < nhi_mean else 'above'} the dataset mean ({nhi_mean:.2f})
                            - **Recommendation**: Based on {critical_pct:.0f}% of your samples being critical, immediate nutrient intervention is needed
                            - **Pattern**: Your dataset shows NHI ranges from {df['NHI'].min():.2f} to {df['NHI'].max():.2f}
                            """)
                        elif pred_nhi < 60:
                            st.warning(" **Warning: Below Optimal Nutrient Levels**")
                            warning_pct = ((df['NHI'] >= 30) & (df['NHI'] < 60)).sum() / len(df) * 100
                            st.markdown(f"""
                            **Based on Your Dataset Analysis**:
                            - **Dataset Context**: {warning_pct:.1f}% of samples in your dataset have NHI 30-60
                            - **Your Prediction**: NHI {pred_nhi_float:.2f} is {'below' if pred_nhi_float < nhi_mean else 'above'} the dataset mean ({nhi_mean:.2f})
                            - **Recommendation**: Based on {warning_pct:.0f}% of your samples in this range, consider nutrient supplementation
                            - **Pattern**: Your dataset shows NHI ranges from {df['NHI'].min():.2f} to {df['NHI'].max():.2f}
                            """)
                        elif pred_nhi < 80:
                            st.info(" **Good: Adequate Nutrient Levels**")
                            good_pct = ((df['NHI'] >= 60) & (df['NHI'] < 80)).sum() / len(df) * 100
                            st.markdown(f"""
                            **Based on Your Dataset Analysis**:
                            - **Dataset Context**: {good_pct:.1f}% of samples in your dataset have NHI 60-80
                            - **Your Prediction**: NHI {pred_nhi_float:.2f} aligns with {good_pct:.0f}% of your dataset samples
                            - **Recommendation**: Maintain current nutrient management as it matches good patterns in your dataset
                            - **Pattern**: Your dataset shows NHI ranges from {df['NHI'].min():.2f} to {df['NHI'].max():.2f}
                            """)
                        else:
                            st.success(" **Excellent: Optimal Nutrient Health Status**")
                            optimal_pct = (df['NHI'] >= 80).sum() / len(df) * 100
                            st.markdown(f"""
                            **Based on Your Dataset Analysis**:
                            - **Dataset Context**: {optimal_pct:.1f}% of samples in your dataset have NHI ≥ 80
                            - **Your Prediction**: NHI {pred_nhi_float:.2f} aligns with {optimal_pct:.0f}% of your dataset samples
                            - **Recommendation**: Continue current practices as they match optimal patterns in your dataset
                            - **Pattern**: Your dataset shows NHI ranges from {df['NHI'].min():.2f} to {df['NHI'].max():.2f}
                            """)
                else:
                    st.info("Run NHI.py to generate dataset insights")
        
        with col2:
            st.subheader(" Model Performance")
            metrics_path = os.path.join(METRICS_DIR, "nhi_model_comparison.csv")
            if os.path.exists(metrics_path):
                metrics_df = pd.read_csv(metrics_path)
                st.dataframe(metrics_df, use_container_width=True)
                
                imp_path = os.path.join(METRICS_DIR, "nhi_feature_importance.csv")
                if os.path.exists(imp_path):
                    imp_df = pd.read_csv(imp_path)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.barplot(data=imp_df, x='Importance', y='Feature', ax=ax, palette='viridis')
                    ax.set_title('Feature Importance')
                    st.pyplot(fig)
            else:
                st.info("Run NHI.py to generate metrics")
        
        # Visualizations
        if df is not None:
            st.markdown("---")
            st.subheader("NHI Visualizations & Insights")
            
            # Create tabs for different visualization categories
            tab1, tab2, tab3, tab4, tab5 = st.tabs([" Distribution & Correlations", " Nutrient Relationships", "📊 Model Performance", "🔍 Residual Analysis", "🌐 Clusters & Global Insights"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    # NHI Distribution and Nutrients
                    plot_path = os.path.join(PLOTS_DIR, "nhi_distribution_nutrients.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="NHI Distribution & NPK Relationships", use_container_width=True)
                    elif 'NHI' in df.columns:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.histplot(df['NHI'], kde=True, bins=30, ax=ax, color='green')
                        ax.set_title('NHI Distribution')
                        ax.set_xlabel('NHI')
                        st.pyplot(fig)
                
                with col2:
                    # Correlation Matrix
                    plot_path = os.path.join(PLOTS_DIR, "nhi_correlation_matrix.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="NHI Correlation Matrix - Sensor Interconnections", use_container_width=True)
                    elif 'NHI' in df.columns:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        corr_cols = ["NHI", "nitrogen", "phosphorus", "potassium",
                                     "conductivity", "moisture", "temperature", "pH"]
                        corr_matrix = df[corr_cols].corr()
                        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
                                    square=True, linewidths=0.5, ax=ax)
                        ax.set_title("NHI Correlation Matrix")
                        st.pyplot(fig)
            
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    # Environmental Factors
                    plot_path = os.path.join(PLOTS_DIR, "nhi_environmental_factors.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="NHI vs Environmental Factors", use_container_width=True)
                    elif 'NHI' in df.columns:
                        fig, ax = plt.subplots(figsize=(12, 4))
                        env_factors = ['conductivity', 'moisture', 'temperature', 'pH']
                        for i, factor in enumerate(env_factors):
                            plt.subplot(1, 4, i+1)
                            sns.scatterplot(x=df[factor], y=df["NHI"], alpha=0.4)
                            plt.xlabel(factor.capitalize())
                            plt.ylabel("NHI")
                            plt.title(f"NHI vs {factor.capitalize()}")
                        plt.tight_layout()
                        st.pyplot(fig)
                
                with col2:
                    # Feature Importance
                    plot_path = os.path.join(PLOTS_DIR, "nhi_feature_importance.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="Feature Importance for NHI Prediction", use_container_width=True)
                    else:
                        st.info("Feature importance plot not available")
            
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    # Model Comparison
                    plot_path = os.path.join(PLOTS_DIR, "nhi_model_comparison.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="NHI Model Performance Comparison", use_container_width=True)
                    else:
                        st.info("Model comparison plot not available")
                
                with col2:
                    # Predictions Scatter
                    plot_path = os.path.join(PLOTS_DIR, "nhi_predictions_scatter.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="Prediction Accuracy (Actual vs Predicted)", use_container_width=True)
                    else:
                        st.info("Prediction scatter plot not available")
            
            with tab4:
                col1, col2 = st.columns(2)
                with col1:
                    # Residual Analysis
                    plot_path = os.path.join(PLOTS_DIR, "nhi_residual_analysis.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="Residual Analysis - TabNet", use_container_width=True)
                    else:
                        st.info("Residual analysis plot not available")
                
                with col2:
                    st.info(" **Insights**: Residual analysis helps identify model bias and prediction patterns.")
            
            with tab5:
                st.subheader(" Clustering Analysis & Global Insights")
                
                # Check if NHI exists, if not calculate it
                if df is not None:
                    if 'NHI' not in df.columns:
                        # Calculate NHI if not present
                        if all(col in df.columns for col in ['nitrogen', 'phosphorus', 'potassium']):
                            df['NHI'] = (
                                0.4 * (df['nitrogen'] / df['nitrogen'].max()) +
                                0.35 * (df['phosphorus'] / df['phosphorus'].max()) +
                                0.25 * (df['potassium'] / df['potassium'].max())
                            ) * 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # K-means clustering visualization for NHI
                    if df is not None:
                        try:
                            # Use available features
                            available_features = ['nitrogen', 'phosphorus', 'potassium', 'conductivity', 'moisture', 'temperature', 'pH']
                            features_to_use = [f for f in available_features if f in df.columns]
                            
                            if len(features_to_use) >= 3:  # Need at least 3 features
                                df_cluster = df[features_to_use].dropna()
                                
                                if len(df_cluster) > 100:  # Need sufficient data points
                                    scaler_cluster = StandardScaler()
                                    X_cluster = scaler_cluster.fit_transform(df_cluster[features_to_use])
                                    
                                    # Apply PCA for 2D visualization
                                    pca = PCA(n_components=2)
                                    X_pca = pca.fit_transform(X_cluster)
                                    
                                    # K-means clustering
                                    n_clusters = min(5, max(2, len(df_cluster) // 200))
                                    
                                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                                    clusters = kmeans.fit_predict(X_cluster)
                                    
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='RdYlGn', alpha=0.6, s=20)
                                    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                                    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                                    ax.set_title('NHI Data Clusters (K-means)')
                                    plt.colorbar(scatter, ax=ax, label='Cluster')
                                    st.pyplot(fig)
                                    
                                    st.info(f"**Insight**: {n_clusters} distinct nutrient health clusters identified, revealing soil condition patterns.")
                                else:
                                    st.warning("Insufficient data points for clustering. Need at least 100 samples.")
                            else:
                                st.warning(f"Insufficient features available. Found: {features_to_use}")
                        except Exception as e:
                            st.error(f"Clustering visualization error: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                    else:
                        st.warning("Dataset not loaded. Please ensure data file is available.")
                
                with col2:
                    # Global NHI statistics and insights
                    if df is not None:
                        # Calculate NHI if not present
                        if 'NHI' not in df.columns:
                            if all(col in df.columns for col in ['nitrogen', 'phosphorus', 'potassium']):
                                df['NHI'] = (
                                    0.4 * (df['nitrogen'] / df['nitrogen'].max()) +
                                    0.35 * (df['phosphorus'] / df['phosphorus'].max()) +
                                    0.25 * (df['potassium'] / df['potassium'].max())
                                ) * 100
                        
                        if 'NHI' in df.columns:
                            st.markdown("###  Global NHI Statistics")
                            nhi_stats = df['NHI'].describe()
                            st.dataframe(nhi_stats, use_container_width=True)
                            
                            st.markdown("###  Key Insights")
                            nhi_mean = df['NHI'].mean()
                            nhi_std = df['NHI'].std()
                            nhi_min = df['NHI'].min()
                            nhi_max = df['NHI'].max()
                            optimal_pct = (df['NHI'] >= 80).sum() / len(df) * 100
                            critical_pct = (df['NHI'] < 30).sum() / len(df) * 100
                            status = 'Optimal' if nhi_mean >= 80 else 'Needs Attention'
                            variability = 'Low' if nhi_std < 10 else 'Moderate' if nhi_std < 20 else 'High'
                            
                            st.markdown(f"""
                            - **Mean NHI**: {nhi_mean:.2f} - {status}
                            - **NHI Range**: {nhi_min:.2f} - {nhi_max:.2f}
                            - **Variability**: {variability} (σ = {nhi_std:.2f})
                            - **Optimal Range Coverage**: {optimal_pct:.1f}% of samples
                            - **Critical Range**: {critical_pct:.1f}% require immediate attention
                            """)
                        else:
                            st.warning("NHI column not available. Cannot calculate statistics.")
                    else:
                        st.warning("Dataset not loaded.")

# ===============================
# Growth Stage Classification Page
# ===============================
elif page == "🌿 Growth Stage Classification":
    st.markdown('<div class="main-header"> Plant Growth Stage Classification</div>', unsafe_allow_html=True)
    
    if 'growth_stage_tabnet' not in models or 'growth_stage_scaler' not in models:
        st.warning(" Growth stage models not loaded. Please run plantgrowthstageclassification.py first.")
    else:
        st.markdown("""
        ### Classify plant growth stage from sensor readings
        
        Predicts whether plants are in Seedling, Vegetative, or Flowering stage.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Sensor Input")
            col1a, col2a = st.columns(2)
            
            with col1a:
                nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=100.0, value=42.0, step=0.1, key='stage_n')
                phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, max_value=100.0, value=30.0, step=0.1, key='stage_p')
                potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=100.0, value=38.0, step=0.1, key='stage_k')
            
            with col2a:
                conductivity = st.number_input("Conductivity", min_value=0.0, max_value=10.0, value=1.2, step=0.1, key='stage_cond')
                moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=28.0, step=0.1, key='stage_moist')
                temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=26.0, step=0.1, key='stage_temp')
                ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1, key='stage_ph')
            
            if st.button(" Classify Growth Stage", type="primary"):
                sensor_input = {
                    'nitrogen': nitrogen,
                    'phosphorus': phosphorus,
                    'potassium': potassium,
                    'conductivity': conductivity,
                    'moisture': moisture,
                    'temperature': temperature,
                    'pH': ph
                }
                
                features = ['nitrogen', 'phosphorus', 'potassium', 'conductivity', 'moisture', 'temperature', 'pH']
                df_input = pd.DataFrame([sensor_input])[features]
                X_scaled = models['growth_stage_scaler'].transform(df_input)
                
                # TabNetClassifier.predict returns 1D array, not 2D
                pred_encoded = models['growth_stage_tabnet'].predict(X_scaled)[0]
                pred_stage = models['growth_stage_encoder'].inverse_transform([pred_encoded])[0]
                
                st.success(f"### Predicted Growth Stage: **{pred_stage}**")
                
                # Data-driven insights only
                st.markdown("###  Data-Driven Insights & Recommendations")
                st.markdown("*Based on analysis of your dataset*")
                
                # Dataset-specific recommendations
                if df is not None:
                    # Find growth stage column
                    growth_col = None
                    for col in df.columns:
                        if 'growth' in col.lower() or 'stage' in col.lower():
                            growth_col = col
                            break
                    
                    if growth_col:
                        data_recs = generate_dataset_recommendations(pred_stage, 'growth_stage', df, dataset_insights)
                        for rec in data_recs:
                            st.info(rec)
                        
                        # Show dataset insights if available
                        if 'growth_stage' in dataset_insights and dataset_insights['growth_stage']:
                            stage_insights = dataset_insights['growth_stage']
                            if stage_insights.get('recommendations'):
                                st.markdown("**Dataset Patterns Found:**")
                                for rec in stage_insights['recommendations']:
                                    st.markdown(f"- {rec}")
                        
                        # Data-driven recommendations based on prediction
                        stage_counts = df[growth_col].value_counts()
                        total = len(df)
                        pred_count = stage_counts.get(pred_stage, 0)
                        pred_pct = (pred_count / total * 100) if total > 0 else 0
                        
                        st.markdown("###  Data-Driven Recommendations")
                        
                        if pred_stage in stage_counts.index:
                            # Get average nutrient levels for this stage from dataset
                            if all(col in df.columns for col in ['nitrogen', 'phosphorus', 'potassium']):
                                stage_nutrients = df[df[growth_col] == pred_stage][['nitrogen', 'phosphorus', 'potassium']].mean()
                                
                                st.info(f" **{pred_stage} Stage Detected**")
                                st.markdown(f"""
                                **Based on Your Dataset Analysis**:
                                - **Dataset Context**: {pred_stage} represents {pred_pct:.1f}% of your dataset ({pred_count} samples)
                                - **Stage-Specific Patterns**: In your dataset, {pred_stage} stage shows average nutrient levels:
                                  - Nitrogen: {stage_nutrients['nitrogen']:.1f}
                                  - Phosphorus: {stage_nutrients['phosphorus']:.1f}
                                  - Potassium: {stage_nutrients['potassium']:.1f}
                                - **Recommendation**: Based on {pred_pct:.0f}% of your samples being in {pred_stage} stage, 
                                  nutrient management should align with patterns observed in your dataset for this stage
                                - **Dataset Pattern**: Your dataset contains {len(stage_counts)} distinct growth stages
                                """)
                            
                            # Compare to other stages
                            if len(stage_counts) > 1:
                                dominant_stage = stage_counts.idxmax()
                                if pred_stage == dominant_stage:
                                    st.success(f" This is the dominant stage in your dataset ({pred_pct:.1f}% of samples)")
                                else:
                                    dominant_pct = (stage_counts[dominant_stage] / total * 100)
                                    st.info(f" Note: {dominant_stage} is the most common stage ({dominant_pct:.1f}% of samples) in your dataset")
                        else:
                            st.warning(f"**{pred_stage} Stage Detected**")
                            st.markdown(f"""
                            **Based on Your Dataset Analysis**:
                            - This stage is not commonly found in your dataset
                            - Consider verifying sensor readings or model prediction
                            """)
                    else:
                        st.info("Growth stage column not found in dataset")
                else:
                    st.info("Run plantgrowthstageclassification.py to generate dataset insights")
        
        with col2:
            st.subheader(" Model Performance")
            metrics_path = os.path.join(METRICS_DIR, "growth_stage_model_comparison.csv")
            if os.path.exists(metrics_path):
                metrics_df = pd.read_csv(metrics_path)
                st.dataframe(metrics_df, use_container_width=True)
                
                imp_path = os.path.join(METRICS_DIR, "growth_stage_feature_importance.csv")
                if os.path.exists(imp_path):
                    imp_df = pd.read_csv(imp_path)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.barplot(data=imp_df, x='Importance', y='Feature', ax=ax, palette='viridis')
                    ax.set_title('Feature Importance')
                    st.pyplot(fig)
            else:
                st.info("Run plantgrowthstageclassification.py to generate metrics")
        
        # Visualizations
        if df is not None and 'growth_stage' in df.columns:
            st.markdown("---")
            st.subheader(" Growth Stage Analysis & Insights")
            
            # Create tabs for different visualization categories
            tab1, tab2, tab3, tab4, tab5 = st.tabs([" Distribution & Patterns", " Stage-Specific Analysis", "📊 Model Performance", "🎯 Classification Metrics", "🌐 Clusters & Global Insights"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    # Growth Stage Distribution
                    plot_path = os.path.join(PLOTS_DIR, "growth_stage_distribution.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="Growth Stage Distribution & Nutrient Analysis", use_container_width=True)
                    else:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        stage_counts = df['growth_stage'].value_counts()
                        ax.bar(stage_counts.index.astype(str), stage_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                        ax.set_title('Growth Stage Distribution')
                        ax.set_xlabel('Growth Stage')
                        ax.set_ylabel('Count')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                
                with col2:
                    # Feature Means by Stage
                    plot_path = os.path.join(PLOTS_DIR, "growth_stage_feature_means.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="Feature Means by Growth Stage", use_container_width=True)
                    else:
                        st.info("Feature means plot not available")
            
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    # Correlation Matrices by Stage
                    plot_path = os.path.join(PLOTS_DIR, "growth_stage_correlation_matrices.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="Stage-Specific Correlation Matrices", use_container_width=True)
                    else:
                        st.info("Correlation matrices plot not available")
                
                with col2:
                    # Feature Importance
                    plot_path = os.path.join(PLOTS_DIR, "growth_stage_feature_importance.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="Feature Importance for Growth Stage Classification", use_container_width=True)
                    else:
                        st.info("Feature importance plot not available")
            
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    # Model Comparison
                    plot_path = os.path.join(PLOTS_DIR, "growth_stage_model_comparison.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="Classification Model Performance Comparison", use_container_width=True)
                    else:
                        st.info("Model comparison plot not available")
                
                with col2:
                    st.info(" **Model Insights**: TabNet shows superior performance for snapshot-based classification, while LSTM/TCN capture temporal patterns.")
            
            with tab4:
                col1, col2 = st.columns(2)
                with col1:
                    # Confusion Matrices
                    plot_path = os.path.join(PLOTS_DIR, "growth_stage_confusion_matrices.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption="Confusion Matrices - All Models", use_container_width=True)
                    else:
                        st.info("Confusion matrices plot not available")
                
                with col2:
                    st.info(" **Classification Insights**: Confusion matrices reveal which growth stages are most accurately predicted and where models struggle.")
            
            with tab5:
                st.subheader(" Clustering Analysis & Global Insights")
                col1, col2 = st.columns(2)
                
                with col1:
                    # K-means clustering visualization for growth stages
                    if df is not None and 'growth_stage' in df.columns:
                        try:
                            features_cluster = ['nitrogen', 'phosphorus', 'potassium', 'conductivity', 'moisture', 'temperature', 'pH']
                            df_cluster = df[features_cluster + ['growth_stage']].dropna()
                            
                            if len(df_cluster) > 0:
                                scaler_cluster = StandardScaler()
                                X_cluster = scaler_cluster.fit_transform(df_cluster[features_cluster])
                                
                                # Apply PCA for 2D visualization
                                pca = PCA(n_components=2)
                                X_pca = pca.fit_transform(X_cluster)
                                
                                # Use actual growth stages for coloring
                                stage_map = {stage: i for i, stage in enumerate(df_cluster['growth_stage'].unique())}
                                stage_colors = df_cluster['growth_stage'].map(stage_map)
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=stage_colors, cmap='Set2', alpha=0.6, s=20)
                                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                                ax.set_title('Growth Stage Clusters (PCA Visualization)')
                                
                                # Add legend
                                for stage, idx in stage_map.items():
                                    ax.scatter([], [], c=plt.cm.Set2(idx), label=stage, s=50)
                                ax.legend()
                                
                                st.pyplot(fig)
                                
                                st.info(f"**Insight**: {len(stage_map)} distinct growth stage clusters identified, showing clear separation in sensor data patterns.")
                        except Exception as e:
                            st.warning(f"Clustering visualization unavailable: {str(e)}")
                
                with col2:
                    # Global growth stage statistics and insights
                    if df is not None and 'growth_stage' in df.columns:
                        st.markdown("###  Growth Stage Distribution")
                        stage_counts = df['growth_stage'].value_counts()
                        st.dataframe(stage_counts.to_frame('Count'), use_container_width=True)
                        
                        st.markdown("### 💡 Key Insights")
                        total = len(df)
                        for stage, count in stage_counts.items():
                            pct = (count / total * 100)
                            st.markdown(f"- **{stage}**: {count} samples ({pct:.1f}%)")
                        
                        st.markdown("### 🔬 Research Insights")
                        st.markdown("""
                        **Stage-Specific Nutrient Patterns** (Marschner, 2012):
                        - **Seedling**: Requires balanced NPK with emphasis on P for root development
                        - **Vegetative**: High N demand for leaf and stem growth
                        - **Flowering**: Increased P and K requirements for reproductive development
                        
                        **Management Implications**:
                        - Adjust fertilization schedules based on predicted growth stages
                        - Monitor transitions between stages for optimal nutrient timing
                        - Use stage predictions to optimize resource allocation
                        """)

# ===============================
# Data Insights Page
# ===============================
elif page == " Data Insights":
    st.markdown('<div class="main-header"> Data Insights & Interconnections</div>', unsafe_allow_html=True)
    
    if df is None:
        st.error("Dataset not available. Please ensure NPK_New Dataset.xlsx is in the data folder.")
    else:
        st.markdown("""
        ### Explore sensor data patterns, correlations, and interconnections
        
        This section provides comprehensive data engineering insights.
        """)
        
        # Data overview
        st.subheader(" Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Records", len(df))
            st.metric("Features", len(df.columns))
        
        with col2:
            st.metric("Missing Values", df.isna().sum().sum())
            st.metric("Duplicate Rows", df.duplicated().sum())
        
        # Statistical summary
        st.subheader(" Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Correlation analysis
        st.subheader("🔗 Correlation Matrix")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                       square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title('Sensor Data Correlation Matrix', fontsize=14, fontweight='bold')
            st.pyplot(fig)
        
        # Feature distributions
        st.subheader(" Feature Distributions")
        selected_features = st.multiselect(
            "Select features to visualize",
            numeric_cols,
            default=numeric_cols[:6] if len(numeric_cols) >= 6 else numeric_cols
        )
        
        if selected_features:
            n_cols = 3
            n_rows = (len(selected_features) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
            
            for i, feature in enumerate(selected_features):
                if i < len(axes):
                    sns.histplot(df[feature], kde=True, ax=axes[i], bins=30)
                    axes[i].set_title(f'{feature} Distribution')
                    axes[i].set_xlabel(feature)
            
            # Hide unused subplots
            for i in range(len(selected_features), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Pairwise relationships
        st.subheader(" Pairwise Relationships")
        if len(numeric_cols) >= 2:
            x_feature = st.selectbox("X-axis feature", numeric_cols, index=0)
            y_feature = st.selectbox("Y-axis feature", numeric_cols, index=min(1, len(numeric_cols)-1))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x=x_feature, y=y_feature, alpha=0.5, ax=ax)
            ax.set_title(f'{y_feature} vs {x_feature}')
            st.pyplot(fig)
        
        # Time series (if applicable)
        if 'date' in df.columns or 'timestamp' in df.columns:
            st.subheader("⏱ Temporal Trends")
            time_col = 'date' if 'date' in df.columns else 'timestamp'
            df[time_col] = pd.to_datetime(df[time_col])
            df_sorted = df.sort_values(time_col)
            
            selected_metrics = st.multiselect(
                "Select metrics to plot over time",
                numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
            
            if selected_metrics:
                fig, ax = plt.subplots(figsize=(14, 6))
                for metric in selected_metrics:
                    ax.plot(df_sorted[time_col], df_sorted[metric], label=metric, alpha=0.7)
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.set_title('Temporal Trends')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p> Precision Indoor Cultivation Dashboard | Data Engineering & Machine Learning</p>
    <p>Powered by TabNet, LSTM, and TCN models</p>
</div>
""", unsafe_allow_html=True)

