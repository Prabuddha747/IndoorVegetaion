# Precision Indoor Cultivation: Predictive Analytics for Soil Health and Plant Growth

## Overview

This research project implements a comprehensive machine learning pipeline for precision indoor cultivation, focusing on three critical predictive tasks:

1. **pH Prediction**: Regression-based prediction of soil pH levels
2. **Nutrient Health Index (NHI) Estimation**: Composite metric estimation for overall nutrient health
3. **Plant Growth Stage Classification**: Multi-class classification of plant growth stages

The project employs deep learning models (TabNet, LSTM, TCN) to capture both snapshot-based and temporal patterns in sensor data, enabling data-driven decision support for precision agriculture.

---

## Project Structure

```
research_paper/
├── data/
│   └── NPK_New Dataset.xlsx          # Main dataset
├── models/                           # Trained model files
│   ├── ph_tabnet/                    # pH TabNet model
│   ├── nhi_tabnet/                   # NHI TabNet model
│   ├── growth_stage_tabnet/          # Growth stage TabNet model
│   ├── *_scaler.pkl                  # Feature scalers
│   └── *_encoder.pkl                 # Label encoders
├── results/
│   ├── plots/                        # Generated visualizations
│   └── metrics/                      # Model performance metrics
├── pH.py                             # pH prediction module
├── NHI.py                            # NHI estimation module
├── plantgrowthstageclassification.py # Growth stage classification module
├── app.py                            # Streamlit dashboard
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Methodology

### 1. pH Prediction (`pH.py`)

**Objective**: Predict soil pH levels from sensor readings (nitrogen, phosphorus, potassium, conductivity, moisture, temperature).

**Models**:
- **Primary**: TabNet Regressor (snapshot-based, interpretable)
- **Temporal Baselines**: LSTM (long-term dependencies) and TCN (short-term patterns)

**Key Features**:
- Comprehensive data exploration and visualization
- Feature importance analysis
- Residual analysis for model diagnostics
- Model comparison across architectures

**Outputs**:
- Trained models saved in `models/`
- Performance metrics in `results/metrics/ph_model_comparison.csv`
- Feature importance in `results/metrics/ph_feature_importance.csv`
- Visualizations in `results/plots/ph_*.png`

### 2. NHI Estimation (`NHI.py`)

**Objective**: Estimate Nutrient Health Index (0-100 scale) as a composite metric of overall nutrient status.

**NHI Calculation** (if not present in dataset):
```
NHI = (0.4 × N_norm + 0.35 × P_norm + 0.25 × K_norm) × 100
```

**Models**:
- **Primary**: TabNet Regressor
- **Temporal Baselines**: LSTM and TCN

**Key Insights**:
- Nutrient interconnections and correlations
- Environmental factor impacts on NHI
- Temporal nutrient dynamics (if time-series data available)

**Outputs**:
- Trained models and scalers
- Performance metrics and feature importance
- Comprehensive visualizations of nutrient relationships

### 3. Plant Growth Stage Classification (`plantgrowthstageclassification.py`)

**Objective**: Classify plants into growth stages (Seedling, Vegetative, Flowering) based on sensor data.

**Models**:
- **Primary**: TabNet Classifier (instantaneous state-based)
- **Temporal Baselines**: LSTM and TCN (for comparative analysis)

**Key Features**:
- Stage-specific nutrient requirements analysis
- Feature importance by growth stage
- Confusion matrices for model evaluation
- Classification metrics (accuracy, precision, recall, F1-score)

**Outputs**:
- Trained classifiers and encoders
- Classification reports and confusion matrices
- Stage-specific feature analysis visualizations

---

## Data Engineering Insights

### Sensor Interconnections

The correlation analysis reveals critical interconnections:

1. **NPK Relationships**: Strong correlations between nitrogen, phosphorus, and potassium levels
2. **Environmental Factors**: Temperature and moisture show significant impact on nutrient availability
3. **pH Dependencies**: pH levels correlate with conductivity and nutrient absorption rates
4. **Temporal Patterns**: Sequential models (LSTM, TCN) capture nutrient dynamics over time

### Feature Importance Findings

**pH Prediction**:
- Conductivity and moisture are primary drivers
- NPK levels show moderate importance
- Temperature has contextual significance

**NHI Estimation**:
- Nitrogen typically shows highest importance
- Phosphorus and potassium follow closely
- Environmental factors (pH, temperature) provide contextual signals

**Growth Stage Classification**:
- Stage-specific nutrient requirements vary significantly
- Seedling: Balanced NPK with focus on root development
- Vegetative: Higher nitrogen emphasis
- Flowering: Increased phosphorus and potassium needs

---

## Model Architecture Comparison

### TabNet (Primary Model)
- **Advantages**: 
  - Interpretable feature importance
  - Excellent performance on structured data
  - No temporal dependencies required
- **Use Case**: Snapshot-based predictions

### LSTM (Temporal Baseline)
- **Advantages**:
  - Captures long-term dependencies
  - Effective for sequential patterns
- **Limitations**: 
  - Requires sequence data
  - More complex training
- **Use Case**: Long-term nutrient trend analysis

### TCN (Temporal Baseline)
- **Advantages**:
  - Captures short-term patterns efficiently
  - Parallel processing capability
- **Limitations**:
  - Limited long-term memory
- **Use Case**: Short-term nutrient dynamics

### Performance Trade-offs

| Model | Accuracy/Stability | Interpretability | Temporal Awareness |
|-------|-------------------|------------------|-------------------|
| TabNet | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| LSTM | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| TCN | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

---

## Installation & Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch pytorch-tabnet joblib streamlit openpyxl
```

### 3. Prepare Dataset

Place `NPK_New Dataset.xlsx` in the `data/` directory.

---

## Usage

### Training Models

Run each module to train models and generate insights:

```bash
# Train pH prediction model
python pH.py

# Train NHI estimation model
python NHI.py

# Train growth stage classification model
python plantgrowthstageclassification.py
```

Each script will:
- Load and preprocess data
- Train TabNet, LSTM, and TCN models
- Generate performance metrics
- Create visualizations
- Save trained models

### Running the Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard provides:
- **Overview**: Dataset statistics and model status
- **pH Prediction**: Interactive pH prediction interface
- **NHI Estimation**: NHI prediction with health status interpretation
- **Growth Stage Classification**: Stage prediction with recommendations
- **Data Insights**: Comprehensive data exploration and visualization

---

## Results & Metrics

### Model Performance

Model performance metrics are saved in `results/metrics/`:

- `ph_model_comparison.csv`: MAE, RMSE, R² for pH models
- `nhi_model_comparison.csv`: MAE, RMSE, R² for NHI models
- `growth_stage_model_comparison.csv`: Accuracy, Precision, Recall, F1 for classification

### Visualizations

All plots are saved in `results/plots/`:

**pH Prediction**:
- `ph_distribution.png`: pH value distribution
- `ph_vs_nutrients.png`: Scatter plots of pH vs NPK
- `ph_correlation_matrix.png`: Feature correlations
- `ph_model_comparison.png`: Model performance comparison
- `ph_predictions_scatter.png`: Prediction vs actual scatter plots
- `ph_feature_importance.png`: TabNet feature importance
- `ph_residual_analysis.png`: Residual plots for diagnostics

**NHI Estimation**:
- `nhi_distribution_nutrients.png`: NHI distribution and NPK relationships
- `nhi_environmental_factors.png`: NHI vs environmental factors
- `nhi_correlation_matrix.png`: Comprehensive correlation matrix
- `nhi_model_comparison.png`: Model performance comparison
- `nhi_predictions_scatter.png`: Prediction accuracy visualization
- `nhi_feature_importance.png`: Feature importance analysis
- `nhi_residual_analysis.png`: Residual diagnostics

**Growth Stage Classification**:
- `growth_stage_distribution.png`: Stage distribution and nutrient analysis
- `growth_stage_correlation_matrices.png`: Stage-specific correlations
- `growth_stage_feature_means.png`: Feature means by stage
- `growth_stage_model_comparison.png`: Classification metrics comparison
- `growth_stage_confusion_matrices.png`: Confusion matrices for all models
- `growth_stage_feature_importance.png`: Feature importance for classification

---

## Key Findings

### 1. Model Performance

- **TabNet** consistently outperforms temporal models for snapshot-based predictions
- **LSTM** and **TCN** provide complementary insights for temporal patterns
- Feature importance reveals stage-specific nutrient dependencies

### 2. Data Insights

- Strong correlations between NPK levels and overall plant health
- Environmental factors (temperature, moisture) significantly impact nutrient availability
- Growth stages exhibit distinct nutrient requirement patterns

### 3. Practical Applications

- **Real-time Monitoring**: Dashboard enables real-time sensor data analysis
- **Decision Support**: Model predictions guide nutrient supplementation decisions
- **Precision Agriculture**: Data-driven approach to optimize indoor cultivation

---

## Future Enhancements

1. **Real-time Data Integration**: Connect to live sensor feeds
2. **Advanced Temporal Models**: Implement Transformer-based architectures
3. **Multi-task Learning**: Joint training of pH, NHI, and growth stage models
4. **Anomaly Detection**: Identify unusual sensor readings
5. **Recommendation System**: Automated nutrient adjustment recommendations
6. **Mobile App**: Extend dashboard to mobile platform

---

## Technical Specifications

### Hardware Requirements
- CPU: Multi-core processor recommended
- RAM: 8GB minimum (16GB recommended)
- GPU: Optional (CUDA-compatible GPU for faster training)

### Software Requirements
- Python 3.8+
- PyTorch 1.12+
- Streamlit 1.0+

### Model Hyperparameters

**TabNet**:
- `n_d=16, n_a=16`: Decision and attention dimensions
- `n_steps=5`: Number of decision steps
- `learning_rate=0.02`: Optimizer learning rate
- `max_epochs=200`: Maximum training epochs
- `patience=30`: Early stopping patience

**LSTM**:
- `hidden_size=64`: Hidden layer dimension
- `num_layers=2`: Number of LSTM layers
- `sequence_length=10`: Input sequence length
- `learning_rate=0.001`: Adam optimizer learning rate
- `epochs=50`: Training epochs

**TCN**:
- `channels=64`: Convolutional channels
- `kernel_size=3`: Convolutional kernel size
- `num_layers=2`: Number of convolutional layers
- `sequence_length=5`: Input sequence length
- `learning_rate=0.001`: Adam optimizer learning rate
- `epochs=50`: Training epochs

---

## Citation

If you use this work in your research, please cite:

```
Precision Indoor Cultivation: Predictive Analytics for Soil Health and Plant Growth
Data Engineering & Machine Learning Approach
```

---

## License

This project is for research purposes. Please ensure proper attribution when using or modifying the code.

---

## Contact & Support

For questions or issues, please refer to the project documentation or create an issue in the repository.

---

## Acknowledgments

- PyTorch TabNet library for TabNet implementation
- Streamlit for dashboard framework
- scikit-learn for preprocessing and evaluation metrics
- All open-source contributors to the libraries used in this project


