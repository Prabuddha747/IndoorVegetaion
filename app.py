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

# -------------------------------
# Torch model definitions (for loading inference weights)
# -------------------------------

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])


class TCNRegressor(nn.Module):
    def __init__(self, input_size, channels=64, kernel=3, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_channels = input_size if i == 0 else channels
            layers.append(weight_norm(nn.Conv1d(in_channels, channels, kernel, padding=kernel // 2)))
            layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.mean(dim=2)
        return self.fc(x)


class TCNRegressorSingle(nn.Module):
    def __init__(self, input_size, channels=64, kernel=3):
        super().__init__()
        self.conv = weight_norm(nn.Conv1d(input_size, channels, kernel, padding=kernel // 2))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv(x))
        x = x.mean(dim=2)
        return self.fc(x)


class AutoencoderRegressor(nn.Module):
    def __init__(self, input_dim, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        y_hat = self.head(z)
        return x_hat, y_hat


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden=64, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden=64, num_layers=2, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])


class TCNClassifier(nn.Module):
    def __init__(self, input_size, channels=64, kernel=3, num_layers=2, num_classes=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_channels = input_size if i == 0 else channels
            layers.append(weight_norm(nn.Conv1d(in_channels, channels, kernel, padding=kernel // 2)))
            layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.mean(dim=2)
        return self.fc(x)


class AutoencoderClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=3, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.head = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        logits = self.head(z)
        return x_hat, logits

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

        # Load PyTorch models (LSTM/GRU/TCN/Autoencoder)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models["torch_device"] = device

        def _load_state(model, path):
            state = torch.load(path, map_location=device)
            model.load_state_dict(state)
            model.eval()
            return model

        def _rnn_layers_from_state(state: dict) -> int:
            # Works for both LSTM and GRU state dicts.
            # If layer 1 weights exist, model was trained with >=2 layers.
            return 2 if any("weight_ih_l1" in k for k in state.keys()) else 1

        def _load_lstm_regressor(path: str, input_dim: int):
            state = torch.load(path, map_location=device)
            num_layers = _rnn_layers_from_state(state)
            model = LSTMRegressor(input_dim, num_layers=num_layers)
            model.load_state_dict(state)
            model.eval()
            return model

        def _load_gru_regressor(path: str, input_dim: int):
            state = torch.load(path, map_location=device)
            num_layers = _rnn_layers_from_state(state)
            model = GRURegressor(input_dim, num_layers=num_layers)
            model.load_state_dict(state)
            model.eval()
            return model

        def _load_tcn_regressor(path: str, input_dim: int):
            state = torch.load(path, map_location=device)
            if any(k.startswith("conv_layers.") for k in state.keys()):
                model = TCNRegressor(input_dim, num_layers=2)
            else:
                model = TCNRegressorSingle(input_dim)
            model.load_state_dict(state)
            model.eval()
            return model

        # pH
        ph_input_dim = 6
        if os.path.exists(os.path.join(MODELS_DIR, "ph_lstm.pt")):
            models["ph_lstm"] = _load_lstm_regressor(os.path.join(MODELS_DIR, "ph_lstm.pt"), ph_input_dim)
        if os.path.exists(os.path.join(MODELS_DIR, "ph_gru.pt")):
            models["ph_gru"] = _load_gru_regressor(os.path.join(MODELS_DIR, "ph_gru.pt"), ph_input_dim)
        if os.path.exists(os.path.join(MODELS_DIR, "ph_tcn.pt")):
            models["ph_tcn"] = _load_tcn_regressor(os.path.join(MODELS_DIR, "ph_tcn.pt"), ph_input_dim)
        if os.path.exists(os.path.join(MODELS_DIR, "ph_autoencoder.pt")):
            models["ph_autoencoder"] = _load_state(AutoencoderRegressor(ph_input_dim), os.path.join(MODELS_DIR, "ph_autoencoder.pt"))

        # NHI
        nhi_input_dim = 7
        if os.path.exists(os.path.join(MODELS_DIR, "nhi_lstm.pt")):
            models["nhi_lstm"] = _load_lstm_regressor(os.path.join(MODELS_DIR, "nhi_lstm.pt"), nhi_input_dim)
        if os.path.exists(os.path.join(MODELS_DIR, "nhi_gru.pt")):
            models["nhi_gru"] = _load_gru_regressor(os.path.join(MODELS_DIR, "nhi_gru.pt"), nhi_input_dim)
        if os.path.exists(os.path.join(MODELS_DIR, "nhi_tcn.pt")):
            models["nhi_tcn"] = _load_tcn_regressor(os.path.join(MODELS_DIR, "nhi_tcn.pt"), nhi_input_dim)
        if os.path.exists(os.path.join(MODELS_DIR, "nhi_autoencoder.pt")):
            models["nhi_autoencoder"] = _load_state(AutoencoderRegressor(nhi_input_dim), os.path.join(MODELS_DIR, "nhi_autoencoder.pt"))

        # Growth stage
        stage_input_dim = 7
        n_stage_classes = 3
        if "growth_stage_encoder" in models:
            try:
                n_stage_classes = len(models["growth_stage_encoder"].classes_)
            except Exception:
                n_stage_classes = 3

        if os.path.exists(os.path.join(MODELS_DIR, "growth_stage_lstm.pt")):
            models["growth_stage_lstm"] = _load_state(
                LSTMClassifier(stage_input_dim, num_classes=n_stage_classes),
                os.path.join(MODELS_DIR, "growth_stage_lstm.pt"),
            )
        if os.path.exists(os.path.join(MODELS_DIR, "growth_stage_gru.pt")):
            models["growth_stage_gru"] = _load_state(
                GRUClassifier(stage_input_dim, num_classes=n_stage_classes),
                os.path.join(MODELS_DIR, "growth_stage_gru.pt"),
            )
        if os.path.exists(os.path.join(MODELS_DIR, "growth_stage_tcn.pt")):
            models["growth_stage_tcn"] = _load_state(
                TCNClassifier(stage_input_dim, num_classes=n_stage_classes),
                os.path.join(MODELS_DIR, "growth_stage_tcn.pt"),
            )
        if os.path.exists(os.path.join(MODELS_DIR, "growth_stage_autoencoder.pt")):
            models["growth_stage_autoencoder"] = _load_state(
                AutoencoderClassifier(stage_input_dim, num_classes=n_stage_classes),
                os.path.join(MODELS_DIR, "growth_stage_autoencoder.pt"),
            )
            
    except Exception as e:
        st.error(f"Error loading models: {e}")
    
    return models


def _as_sequence(x_2d: np.ndarray, seq_len: int) -> torch.Tensor:
    # Repeat the single observation seq_len times so sequence models can run in the dashboard.
    x = np.repeat(x_2d[np.newaxis, :, :], repeats=seq_len, axis=1)  # (1, seq, features)
    return torch.tensor(x, dtype=torch.float32)


def _create_sequences_np(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.asarray(Xs), np.asarray(ys)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def _classification_fpr_fnr_macro(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < n_classes and 0 <= int(p) < n_classes:
            cm[int(t), int(p)] += 1
    fprs, fnrs = [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        tn = int(cm.sum() - (tp + fn + fp))
        fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = (fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        fprs.append(fpr)
        fnrs.append(fnr)
    return float(np.mean(fprs)), float(np.mean(fnrs))


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else float("nan")

    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    support_sum = 0
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        support = int(np.sum(y_true == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precision_sum += prec * support
        recall_sum += rec * support
        f1_sum += f1 * support
        support_sum += support
    precision = precision_sum / support_sum if support_sum > 0 else float("nan")
    recall = recall_sum / support_sum if support_sum > 0 else float("nan")
    f1 = f1_sum / support_sum if support_sum > 0 else float("nan")
    fpr, fnr = _classification_fpr_fnr_macro(y_true, y_pred, n_classes)
    return acc, precision, recall, f1, fpr, fnr


def _maybe_compute_growth_stage_metrics(df: pd.DataFrame, models: dict):
    if df is None or "growth_stage_scaler" not in models or "growth_stage_encoder" not in models:
        return None

    features = ["nitrogen", "phosphorus", "potassium", "conductivity", "moisture", "temperature", "pH"]
    if not all(f in df.columns for f in features):
        return None

    df_work = df.copy()
    if "growth_stage" not in df_work.columns:
        if "Plant Growth Stage" in df_work.columns:
            df_work["growth_stage"] = df_work["Plant Growth Stage"].astype(str).str.strip()
        else:
            return None

    df_clean = df_work[features + ["growth_stage"]].dropna().copy()
    if df_clean.empty:
        return None

    le = models["growth_stage_encoder"]
    try:
        y_all = le.transform(df_clean["growth_stage"].astype(str).str.strip())
    except Exception:
        return None
    n_classes = len(getattr(le, "classes_", [])) or int(np.max(y_all) + 1)

    rng = np.random.RandomState(42)
    idx = np.arange(len(df_clean))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr_idx, te_idx = idx[:split], idx[split:]
    X_train = df_clean.iloc[tr_idx][features].values
    X_test = df_clean.iloc[te_idx][features].values
    y_test = np.asarray(y_all)[te_idx]

    scaler = models["growth_stage_scaler"]
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    rows = []

    if "growth_stage_tabnet" in models:
        try:
            y_pred = models["growth_stage_tabnet"].predict(X_test_s).astype(int)
            acc, prec, rec, f1, fpr, fnr = _classification_metrics(y_test, y_pred, n_classes)
            rows.append({"Model": "TabNet", "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "ROC AUC": float("nan"), "FPR": fpr, "FNR": fnr})
        except Exception:
            pass

    # LSTM/GRU (seq_len=10)
    seq_len = 10
    X_seq, y_seq = _create_sequences_np(X_train_s, np.asarray(y_all)[tr_idx], seq_len=seq_len)
    if len(X_seq) > 0:
        split2 = int(0.8 * len(X_seq))
        X_te = X_seq[split2:]
        y_te = y_seq[split2:]
        x_t = torch.tensor(X_te, dtype=torch.float32)

        def _seq_pred(model_key: str):
            with torch.no_grad():
                logits = models[model_key](x_t)
                return torch.argmax(logits, dim=1).cpu().numpy().astype(int)

        if "growth_stage_lstm" in models:
            y_pred = _seq_pred("growth_stage_lstm")
            acc, prec, rec, f1, fpr, fnr = _classification_metrics(y_te, y_pred, n_classes)
            rows.append({"Model": "LSTM", "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "ROC AUC": float("nan"), "FPR": fpr, "FNR": fnr})
        if "growth_stage_gru" in models:
            y_pred = _seq_pred("growth_stage_gru")
            acc, prec, rec, f1, fpr, fnr = _classification_metrics(y_te, y_pred, n_classes)
            rows.append({"Model": "GRU", "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "ROC AUC": float("nan"), "FPR": fpr, "FNR": fnr})

    # TCN (seq_len=5)
    seq_len_tcn = 5
    X_seq_t, y_seq_t = _create_sequences_np(X_train_s, np.asarray(y_all)[tr_idx], seq_len=seq_len_tcn)
    if len(X_seq_t) > 0 and "growth_stage_tcn" in models:
        split2 = int(0.8 * len(X_seq_t))
        X_te = X_seq_t[split2:]
        y_te = y_seq_t[split2:]
        x_t = torch.tensor(X_te, dtype=torch.float32)
        with torch.no_grad():
            logits = models["growth_stage_tcn"](x_t)
            y_pred = torch.argmax(logits, dim=1).cpu().numpy().astype(int)
        acc, prec, rec, f1, fpr, fnr = _classification_metrics(y_te, y_pred, n_classes)
        rows.append({"Model": "TCN", "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "ROC AUC": float("nan"), "FPR": fpr, "FNR": fnr})

    if "growth_stage_autoencoder" in models:
        x_t = torch.tensor(X_test_s.astype(np.float32), dtype=torch.float32)
        with torch.no_grad():
            _, logits = models["growth_stage_autoencoder"](x_t)
            y_pred = torch.argmax(logits, dim=1).cpu().numpy().astype(int)
        acc, prec, rec, f1, fpr, fnr = _classification_metrics(y_test, y_pred, n_classes)
        rows.append({"Model": "Autoencoder", "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "ROC AUC": float("nan"), "FPR": fpr, "FNR": fnr})

    return pd.DataFrame(rows) if rows else None

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
                
                # Predict (select model)
                available_models = []
                if 'ph_tabnet' in models:
                    available_models.append("TabNet")
                if 'ph_lstm' in models:
                    available_models.append("LSTM")
                if 'ph_gru' in models:
                    available_models.append("GRU")
                if 'ph_tcn' in models:
                    available_models.append("TCN")
                if 'ph_autoencoder' in models:
                    available_models.append("Autoencoder")

                chosen = st.session_state.get("ph_model_choice", "TabNet")
                if chosen not in available_models and available_models:
                    chosen = available_models[0]

                # Store choice UI (kept near prediction for clarity)
                st.session_state["ph_model_choice"] = st.selectbox(
                    "Model",
                    available_models if available_models else ["TabNet"],
                    index=(available_models.index(chosen) if chosen in available_models else 0),
                    key="ph_model_choice_select",
                )
                chosen = st.session_state["ph_model_choice"]

                if chosen == "TabNet":
                    pred_ph = float(models['ph_tabnet'].predict(X_scaled)[0][0])
                elif chosen == "LSTM":
                    x_t = _as_sequence(X_scaled.astype(np.float32), seq_len=10)
                    with torch.no_grad():
                        pred_ph = float(models["ph_lstm"](x_t).cpu().numpy().flatten()[0])
                elif chosen == "GRU":
                    x_t = _as_sequence(X_scaled.astype(np.float32), seq_len=10)
                    with torch.no_grad():
                        pred_ph = float(models["ph_gru"](x_t).cpu().numpy().flatten()[0])
                elif chosen == "TCN":
                    x_t = _as_sequence(X_scaled.astype(np.float32), seq_len=5)
                    with torch.no_grad():
                        pred_ph = float(models["ph_tcn"](x_t).cpu().numpy().flatten()[0])
                else:
                    x_t = torch.tensor(X_scaled.astype(np.float32), dtype=torch.float32)
                    with torch.no_grad():
                        _, y_hat = models["ph_autoencoder"](x_t)
                        pred_ph = float(y_hat.cpu().numpy().flatten()[0])
                
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

                available_models = []
                if 'nhi_tabnet' in models:
                    available_models.append("TabNet")
                if 'nhi_lstm' in models:
                    available_models.append("LSTM")
                if 'nhi_gru' in models:
                    available_models.append("GRU")
                if 'nhi_tcn' in models:
                    available_models.append("TCN")
                if 'nhi_autoencoder' in models:
                    available_models.append("Autoencoder")

                chosen = st.session_state.get("nhi_model_choice", "TabNet")
                if chosen not in available_models and available_models:
                    chosen = available_models[0]

                st.session_state["nhi_model_choice"] = st.selectbox(
                    "Model",
                    available_models if available_models else ["TabNet"],
                    index=(available_models.index(chosen) if chosen in available_models else 0),
                    key="nhi_model_choice_select",
                )
                chosen = st.session_state["nhi_model_choice"]

                if chosen == "TabNet":
                    pred_nhi_float = float(models['nhi_tabnet'].predict(X_scaled)[0][0])
                elif chosen == "LSTM":
                    x_t = _as_sequence(X_scaled.astype(np.float32), seq_len=10)
                    with torch.no_grad():
                        pred_nhi_float = float(models["nhi_lstm"](x_t).cpu().numpy().flatten()[0])
                elif chosen == "GRU":
                    x_t = _as_sequence(X_scaled.astype(np.float32), seq_len=10)
                    with torch.no_grad():
                        pred_nhi_float = float(models["nhi_gru"](x_t).cpu().numpy().flatten()[0])
                elif chosen == "TCN":
                    x_t = _as_sequence(X_scaled.astype(np.float32), seq_len=5)
                    with torch.no_grad():
                        pred_nhi_float = float(models["nhi_tcn"](x_t).cpu().numpy().flatten()[0])
                else:
                    x_t = torch.tensor(X_scaled.astype(np.float32), dtype=torch.float32)
                    with torch.no_grad():
                        _, y_hat = models["nhi_autoencoder"](x_t)
                        pred_nhi_float = float(y_hat.cpu().numpy().flatten()[0])
                
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
                        
                        if pred_nhi_float < 30:
                            st.error(" **Critical: Very Low Nutrient Health**")
                            critical_pct = (df['NHI'] < 30).sum() / len(df) * 100
                            st.markdown(f"""
                            **Based on Your Dataset Analysis**:
                            - **Dataset Context**: {critical_pct:.1f}% of samples in your dataset have NHI < 30
                            - **Your Prediction**: NHI {pred_nhi_float:.2f} is {'below' if pred_nhi_float < nhi_mean else 'above'} the dataset mean ({nhi_mean:.2f})
                            - **Recommendation**: Based on {critical_pct:.0f}% of your samples being critical, immediate nutrient intervention is needed
                            - **Pattern**: Your dataset shows NHI ranges from {df['NHI'].min():.2f} to {df['NHI'].max():.2f}
                            """)
                        elif pred_nhi_float < 60:
                            st.warning(" **Warning: Below Optimal Nutrient Levels**")
                            warning_pct = ((df['NHI'] >= 30) & (df['NHI'] < 60)).sum() / len(df) * 100
                            st.markdown(f"""
                            **Based on Your Dataset Analysis**:
                            - **Dataset Context**: {warning_pct:.1f}% of samples in your dataset have NHI 30-60
                            - **Your Prediction**: NHI {pred_nhi_float:.2f} is {'below' if pred_nhi_float < nhi_mean else 'above'} the dataset mean ({nhi_mean:.2f})
                            - **Recommendation**: Based on {warning_pct:.0f}% of your samples in this range, consider nutrient supplementation
                            - **Pattern**: Your dataset shows NHI ranges from {df['NHI'].min():.2f} to {df['NHI'].max():.2f}
                            """)
                        elif pred_nhi_float < 80:
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
elif page == " Growth Stage Classification" or page == "🌿 Growth Stage Classification":
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

                available_models = []
                if 'growth_stage_tabnet' in models:
                    available_models.append("TabNet")
                if 'growth_stage_lstm' in models:
                    available_models.append("LSTM")
                if 'growth_stage_gru' in models:
                    available_models.append("GRU")
                if 'growth_stage_tcn' in models:
                    available_models.append("TCN")
                if 'growth_stage_autoencoder' in models:
                    available_models.append("Autoencoder")

                chosen = st.session_state.get("stage_model_choice", "TabNet")
                if chosen not in available_models and available_models:
                    chosen = available_models[0]

                st.session_state["stage_model_choice"] = st.selectbox(
                    "Model",
                    available_models if available_models else ["TabNet"],
                    index=(available_models.index(chosen) if chosen in available_models else 0),
                    key="stage_model_choice_select",
                )
                chosen = st.session_state["stage_model_choice"]

                if chosen == "TabNet":
                    pred_encoded = int(models['growth_stage_tabnet'].predict(X_scaled)[0])
                elif chosen == "LSTM":
                    x_t = _as_sequence(X_scaled.astype(np.float32), seq_len=10)
                    with torch.no_grad():
                        logits = models["growth_stage_lstm"](x_t)
                        pred_encoded = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
                elif chosen == "GRU":
                    x_t = _as_sequence(X_scaled.astype(np.float32), seq_len=10)
                    with torch.no_grad():
                        logits = models["growth_stage_gru"](x_t)
                        pred_encoded = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
                elif chosen == "TCN":
                    x_t = _as_sequence(X_scaled.astype(np.float32), seq_len=5)
                    with torch.no_grad():
                        logits = models["growth_stage_tcn"](x_t)
                        pred_encoded = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
                else:
                    x_t = torch.tensor(X_scaled.astype(np.float32), dtype=torch.float32)
                    with torch.no_grad():
                        _, logits = models["growth_stage_autoencoder"](x_t)
                        pred_encoded = int(torch.argmax(logits, dim=1).cpu().numpy()[0])

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

                # Highlight key classification metrics for currently selected model (if available)
                selected_name = st.session_state.get("stage_model_choice", "TabNet")
                if "Model" in metrics_df.columns and selected_name in metrics_df["Model"].astype(str).tolist():
                    row = metrics_df[metrics_df["Model"].astype(str) == str(selected_name)].iloc[0]
                    cols = st.columns(4)
                    if "ROC AUC" in metrics_df.columns:
                        cols[0].metric("ROC AUC", f"{float(row['ROC AUC']):.2f}" if pd.notna(row["ROC AUC"]) else "—")
                    if "Accuracy" in metrics_df.columns:
                        cols[1].metric("Accuracy", f"{float(row['Accuracy']):.2f}")
                    if "Precision" in metrics_df.columns:
                        cols[2].metric("Precision", f"{float(row['Precision']):.2f}")
                    if "Recall" in metrics_df.columns:
                        cols[3].metric("Recall", f"{float(row['Recall']):.2f}")

                    cols2 = st.columns(3)
                    if "F1-Score" in metrics_df.columns:
                        cols2[0].metric("F1-score", f"{float(row['F1-Score']):.2f}")
                    if "FPR" in metrics_df.columns:
                        cols2[1].metric("False Positive Rate (FPR)", f"{float(row['FPR']):.2f}" if pd.notna(row["FPR"]) else "—")
                    if "FNR" in metrics_df.columns:
                        cols2[2].metric("False Negative Rate (FNR)", f"{float(row['FNR']):.2f}" if pd.notna(row["FNR"]) else "—")
                
                imp_path = os.path.join(METRICS_DIR, "growth_stage_feature_importance.csv")
                if os.path.exists(imp_path):
                    imp_df = pd.read_csv(imp_path)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.barplot(data=imp_df, x='Importance', y='Feature', ax=ax, palette='viridis')
                    ax.set_title('Feature Importance')
                    st.pyplot(fig)
            else:
                metrics_df = _maybe_compute_growth_stage_metrics(df, models)
                if metrics_df is not None and not metrics_df.empty:
                    st.info("Metrics computed live from your dataset (no training script run).")
                    st.dataframe(metrics_df, use_container_width=True)
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

