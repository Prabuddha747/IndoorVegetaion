"""
Nutrient Health Index (NHI) Prediction
======================================
This module implements NHI estimation using TabNet, LSTM, and TCN models.
NHI is a composite metric indicating overall nutrient health status.
"""

# ===============================
# Configuration
# ===============================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "NPK_New Dataset.xlsx")

PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ===============================
# Imports
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score,
    confusion_matrix, roc_auc_score
)
import joblib

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from pytorch_tabnet.tab_model import TabNetClassifier

# ===============================
# Data Loading & Exploration
# ===============================

print("Loading dataset for NHI prediction...")
df = pd.read_excel(DATA_PATH)

print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isna().sum()}")

# Calculate NHI if not present (weighted combination of NPK)
nhi_was_synthetic = False
if 'NHI' not in df.columns:
    # Normalize NPK values and create composite NHI score
    df['NHI'] = (
        0.4 * (df['nitrogen'] / df['nitrogen'].max()) +
        0.35 * (df['phosphorus'] / df['phosphorus'].max()) +
        0.25 * (df['potassium'] / df['potassium'].max())
    ) * 100
    nhi_was_synthetic = True
    print("\nNHI calculated as weighted combination of NPK values")
else:
    # Some datasets store NHI normalized to ~[0, 3] or [0, 1]. Rescale to 0–100 for consistent bins/UI.
    try:
        nhi_max = float(pd.to_numeric(df["NHI"], errors="coerce").max())
        if np.isfinite(nhi_max) and nhi_max <= 3.5:
            df["NHI"] = pd.to_numeric(df["NHI"], errors="coerce") * 100.0
            print(f"\nDetected low-range NHI (max={nhi_max:.3f}); rescaled NHI by ×100 to match 0–100 scale.")
    except Exception:
        pass


def nhi_to_class(nhi_values: np.ndarray) -> np.ndarray:
    # 0: Critical (<30), 1: Warning (30–60), 2: Good (60–80), 3: Optimal (>=80)
    nhi_values = np.asarray(nhi_values, dtype=float)
    return np.digitize(nhi_values, bins=[30.0, 60.0, 80.0], right=False)


def nhi_to_class_with_edges(nhi_values: np.ndarray, edges: list[float]) -> np.ndarray:
    # edges are internal cut points (len = n_classes-1)
    nhi_values = np.asarray(nhi_values, dtype=float)
    return np.digitize(nhi_values, bins=np.asarray(edges, dtype=float), right=False)


# Class names for display
NHI_CLASS_NAMES = ["Critical", "Warning", "Good", "Optimal"]


def _fpr_fnr_macro(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[float, float]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fprs, fnrs = [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = (fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        fprs.append(fpr)
        fnrs.append(fnr)
    return float(np.mean(fprs)), float(np.mean(fnrs))


def _approx_multiclass_auc_from_regression(pred: np.ndarray, y_true_cls: np.ndarray, centers: list[float]) -> float:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    scores = np.stack([-np.abs(pred - c) for c in centers], axis=1)
    scores = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=1, keepdims=True)
    try:
        return float(roc_auc_score(y_true_cls, probs, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")

# ===============================
# Data Engineering & Visualizations
# ===============================

# NHI distribution
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
sns.histplot(df["NHI"], kde=True, bins=30, color='green')
plt.title("Distribution of Nutrient Health Index (NHI)")
plt.xlabel("NHI")
plt.ylabel("Frequency")

# NHI vs individual nutrients
plt.subplot(2, 2, 2)
sns.scatterplot(x=df["nitrogen"], y=df["NHI"], alpha=0.4, color='blue')
plt.xlabel("Nitrogen")
plt.ylabel("NHI")
plt.title("NHI vs Nitrogen")

plt.subplot(2, 2, 3)
sns.scatterplot(x=df["phosphorus"], y=df["NHI"], alpha=0.4, color='orange')
plt.xlabel("Phosphorus")
plt.ylabel("NHI")
plt.title("NHI vs Phosphorus")

plt.subplot(2, 2, 4)
sns.scatterplot(x=df["potassium"], y=df["NHI"], alpha=0.4, color='red')
plt.xlabel("Potassium")
plt.ylabel("NHI")
plt.title("NHI vs Potassium")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/nhi_distribution_nutrients.png", dpi=300)
plt.close()

# NHI vs environmental factors
plt.figure(figsize=(12, 4))
env_factors = ['conductivity', 'moisture', 'temperature', 'pH']
for i, factor in enumerate(env_factors):
    plt.subplot(1, 4, i+1)
    sns.scatterplot(x=df[factor], y=df["NHI"], alpha=0.4)
    plt.xlabel(factor.capitalize())
    plt.ylabel("NHI")
    plt.title(f"NHI vs {factor.capitalize()}")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/nhi_environmental_factors.png", dpi=300)
plt.close()

# Comprehensive correlation matrix
plt.figure(figsize=(10, 8))
corr_cols = ["NHI", "nitrogen", "phosphorus", "potassium",
             "conductivity", "moisture", "temperature", "pH"]
corr_matrix = df[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
            square=True, linewidths=0.5)
plt.title("NHI Correlation Matrix - Sensor Interconnections")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/nhi_correlation_matrix.png", dpi=300)
plt.close()

# Temporal patterns (if time-based data exists)
if 'date' in df.columns or 'timestamp' in df.columns:
    time_col = 'date' if 'date' in df.columns else 'timestamp'
    df[time_col] = pd.to_datetime(df[time_col])
    df_sorted = df.sort_values(time_col)
    
    plt.figure(figsize=(14, 6))
    plt.plot(df_sorted[time_col], df_sorted['NHI'], alpha=0.6, linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('NHI')
    plt.title('NHI Temporal Trends')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/nhi_temporal_trends.png", dpi=300)
    plt.close()

# ===============================
# Feature Engineering
# ===============================

ALL_FEATURES = [
    "nitrogen", "phosphorus", "potassium",
    "conductivity", "moisture", "temperature", "pH"
]

# If NHI is synthetic from NPK, using N/P/K as inputs makes the task trivial (leakage).
# In that case, train a "no-leak" model using only environmental/context features.
if "nhi_was_synthetic" in globals() and nhi_was_synthetic:
    FEATURES = ["conductivity", "moisture", "temperature", "pH"]
else:
    FEATURES = ALL_FEATURES
TARGET = "NHI"

# Remove rows with missing target
df_clean = df[FEATURES + [TARGET]].dropna()

X = df_clean[FEATURES]
y_cont = df_clean[TARGET]

X_train, X_test, y_train_cont, y_test_cont = train_test_split(
    X, y_cont, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

joblib.dump(scaler, f"{MODELS_DIR}/nhi_scaler.pkl")
joblib.dump(FEATURES, f"{MODELS_DIR}/nhi_features.pkl")

# ===============================
# TabNet Classifier
# ===============================

print("\n" + "="*60)
print("Training TabNet Classifier for NHI...")
print("="*60)

tabnet_nhi = TabNetClassifier(
    n_d=12, n_a=12, n_steps=4,
    optimizer_params=dict(lr=0.01, weight_decay=1e-4),
    mask_type="entmax",
    lambda_sparse=1e-3
)

# Bin edges for classification evaluation/training
fixed_edges = [30.0, 60.0, 80.0]
bin_scheme = "fixed(30/60/80)"
eval_edges = fixed_edges

vals_all = np.asarray(y_train_cont.values, dtype=float)
fixed_cls_all = nhi_to_class(vals_all)
vals, counts = np.unique(fixed_cls_all, return_counts=True)
max_frac = float(np.max(counts) / np.sum(counts)) if np.sum(counts) else 1.0

if len(np.unique(fixed_cls_all)) < 2 or max_frac > 0.90:
    # percentile bins (non-uniform)
    ps = np.quantile(vals_all, [0.10, 0.40, 0.70]).tolist()
    ps = sorted(list(dict.fromkeys([float(v) for v in ps])))
    if len(ps) == 3:
        eval_edges = ps
        bin_scheme = f"percentiles({eval_edges[0]:.1f}/{eval_edges[1]:.1f}/{eval_edges[2]:.1f})"
        print("\nNOTE: Fixed NHI bins are not informative. Using percentile bins for classifier:", bin_scheme)

y_train = nhi_to_class_with_edges(y_train_cont.values, eval_edges).astype(int)
y_test = nhi_to_class_with_edges(y_test_cont.values, eval_edges).astype(int)

tabnet_nhi.fit(
    X_train_s,
    y_train,
    eval_set=[(X_test_s, y_test)],
    max_epochs=200,
    patience=30
)

y_pred_tabnet = tabnet_nhi.predict(X_test_s).astype(int)
y_proba_tabnet = tabnet_nhi.predict_proba(X_test_s)

# ===============================
# LSTM Classifier
# ===============================

print("\n" + "="*60)
print("Training LSTM for NHI Prediction...")
print("="*60)

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

SEQ_LEN_LSTM = 10
X_seq, y_seq = create_sequences(X_train_s, y_train, SEQ_LEN_LSTM)

split = int(0.8 * len(X_seq))
X_tr_lstm, X_te_lstm = X_seq[:split], X_seq[split:]
y_tr_lstm, y_te_lstm = y_seq[:split], y_seq[split:]

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden=64, num_layers=2, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

lstm_model = LSTMClassifier(X_tr_lstm.shape[2], num_classes=len(NHI_CLASS_NAMES)).to(device)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

X_tr_t = torch.tensor(X_tr_lstm, dtype=torch.float32).to(device)
y_tr_t = torch.tensor(y_tr_lstm, dtype=torch.long).to(device)

# Training loop
lstm_model.train()
for epoch in range(25):
    optimizer.zero_grad()
    logits = lstm_model(X_tr_t)
    loss = criterion(logits, y_tr_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")

lstm_model.eval()
X_te_t = torch.tensor(X_te_lstm, dtype=torch.float32).to(device)
with torch.no_grad():
    logits = lstm_model(X_te_t)
    y_proba_lstm = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred_lstm = np.argmax(y_proba_lstm, axis=1).astype(int)

torch.save(lstm_model.state_dict(), f"{MODELS_DIR}/nhi_lstm.pt")

# ===============================
# GRU Classifier
# ===============================

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden=64, num_layers=2, num_classes=4):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

gru_model = GRUClassifier(X_tr_lstm.shape[2], num_classes=len(NHI_CLASS_NAMES)).to(device)
optimizer_gru = torch.optim.Adam(gru_model.parameters(), lr=0.001, weight_decay=1e-4)

X_tr_t = torch.tensor(X_tr_lstm, dtype=torch.float32).to(device)
y_tr_t = torch.tensor(y_tr_lstm, dtype=torch.long).to(device)

gru_model.train()
for epoch in range(25):
    optimizer_gru.zero_grad()
    logits = gru_model(X_tr_t)
    loss = criterion(logits, y_tr_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gru_model.parameters(), 1.0)
    optimizer_gru.step()
    if (epoch + 1) % 10 == 0:
        print(f"[GRU] Epoch {epoch+1}/50, Loss: {loss.item():.4f}")

gru_model.eval()
X_te_t = torch.tensor(X_te_lstm, dtype=torch.float32).to(device)
with torch.no_grad():
    logits = gru_model(X_te_t)
    y_proba_gru = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred_gru = np.argmax(y_proba_gru, axis=1).astype(int)

torch.save(gru_model.state_dict(), f"{MODELS_DIR}/nhi_gru.pt")

# ===============================
# Autoencoder + Classifier (supervised)
# ===============================

class AutoencoderClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=4, latent_dim=16, hidden_dim=64):
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

ae_model = AutoencoderClassifier(input_dim=X_train_s.shape[1], num_classes=len(NHI_CLASS_NAMES)).to(device)
opt_ae = torch.optim.Adam(ae_model.parameters(), lr=0.001, weight_decay=1e-4)
mse = nn.MSELoss()
ce = nn.CrossEntropyLoss()

X_tr_ae = torch.tensor(X_train_s, dtype=torch.float32).to(device)
y_tr_ae = torch.tensor(y_train, dtype=torch.long).to(device)
X_te_ae = torch.tensor(X_test_s, dtype=torch.float32).to(device)

ae_model.train()
alpha = 0.3
for epoch in range(40):
    opt_ae.zero_grad()
    x_hat, logits = ae_model(X_tr_ae)
    loss_recon = mse(x_hat, X_tr_ae)
    loss_cls = ce(logits, y_tr_ae)
    loss = loss_recon + alpha * loss_cls
    loss.backward()
    opt_ae.step()

ae_model.eval()
with torch.no_grad():
    _, logits = ae_model(X_te_ae)
    y_proba_ae = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred_ae = np.argmax(y_proba_ae, axis=1).astype(int)

torch.save(ae_model.state_dict(), f"{MODELS_DIR}/nhi_autoencoder.pt")

# ===============================
# TCN Classifier
# ===============================

print("\n" + "="*60)
print("Training TCN Classifier for NHI...")
print("="*60)

SEQ_LEN_TCN = 5
X_seq_tcn, y_seq_tcn = create_sequences(X_train_s, y_train, SEQ_LEN_TCN)

split = int(0.8 * len(X_seq_tcn))
X_tr_tcn, X_te_tcn = X_seq_tcn[:split], X_seq_tcn[split:]
y_tr_tcn, y_te_tcn = y_seq_tcn[:split], y_seq_tcn[split:]

class TCNClassifier(nn.Module):
    def __init__(self, input_size, channels=64, kernel=3, num_layers=2, num_classes=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_channels = input_size if i == 0 else channels
            layers.append(weight_norm(
                nn.Conv1d(in_channels, channels, kernel, padding=kernel//2)
            ))
            layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.mean(dim=2)
        return self.fc(x)

tcn_model = TCNClassifier(X_tr_tcn.shape[2], num_classes=len(NHI_CLASS_NAMES)).to(device)
optimizer = torch.optim.Adam(tcn_model.parameters(), lr=0.001, weight_decay=1e-4)

X_tr_t = torch.tensor(X_tr_tcn, dtype=torch.float32).to(device)
y_tr_t = torch.tensor(y_tr_tcn, dtype=torch.long).to(device)

# Training loop
tcn_model.train()
for epoch in range(25):
    optimizer.zero_grad()
    logits = tcn_model(X_tr_t)
    loss = criterion(logits, y_tr_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(tcn_model.parameters(), 1.0)
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")

tcn_model.eval()
X_te_t = torch.tensor(X_te_tcn, dtype=torch.float32).to(device)
with torch.no_grad():
    logits = tcn_model(X_te_t)
    y_proba_tcn = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred_tcn = np.argmax(y_proba_tcn, axis=1).astype(int)

torch.save(tcn_model.state_dict(), f"{MODELS_DIR}/nhi_tcn.pt")


n_nhi_classes = 4
nhi_centers = [15.0, 45.0, 70.0, 90.0]  # representative points for Critical/Warning/Good/Optimal

# If fixed bins collapse into 1 class OR are extremely imbalanced,
# switch to percentile-based bins for evaluation so Accuracy/F1 are informative
# without forcing perfectly uniform class counts.
fixed_edges = [30.0, 60.0, 80.0]
fixed_cls_all = nhi_to_class(y_train_cont.values)
unique_fixed = np.unique(fixed_cls_all)

bin_scheme = "fixed(30/60/80)"
print("\nBinned NHI class distribution (train):", dict(zip(*np.unique(y_train, return_counts=True))))
print("Binned NHI class distribution (test):", dict(zip(*np.unique(y_test, return_counts=True))))


def _fpr_fnr_macro_from_cm(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fprs, fnrs = [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = (fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        fprs.append(fpr)
        fnrs.append(fnr)
    return float(np.mean(fprs)), float(np.mean(fnrs))


def _cls_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None, n_classes: int):
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    fpr, fnr = _fpr_fnr_macro_from_cm(y_true, y_pred, n_classes=n_classes)
    try:
        auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")) if y_proba is not None else float("nan")
    except Exception:
        auc = float("nan")
    return acc, prec, rec, f1, auc, fpr, fnr


def _tabnet_stratified_cv_summary(X_raw: pd.DataFrame, y_cls: np.ndarray, n_splits: int = 5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1_macros, bal_accs = [], [], []
    for tr_idx, va_idx in skf.split(X_raw, y_cls):
        x_tr = X_raw.iloc[tr_idx].values
        x_va = X_raw.iloc[va_idx].values
        y_tr = y_cls[tr_idx]
        y_va = y_cls[va_idx]

        sc = StandardScaler()
        x_tr_s = sc.fit_transform(x_tr)
        x_va_s = sc.transform(x_va)

        m = TabNetClassifier(
            n_d=8, n_a=8, n_steps=3,
            optimizer_params=dict(lr=0.01, weight_decay=1e-4),
            mask_type="entmax",
            lambda_sparse=1e-3
        )
        m.fit(
            x_tr_s, y_tr,
            eval_set=[(x_va_s, y_va)],
            max_epochs=60,
            patience=10
        )
        pred = m.predict(x_va_s).astype(int)
        accs.append(accuracy_score(y_va, pred))
        f1_macros.append(f1_score(y_va, pred, average="macro", zero_division=0))
        bal_accs.append(balanced_accuracy_score(y_va, pred))

    return (
        float(np.mean(accs)), float(np.std(accs)),
        float(np.mean(f1_macros)), float(np.std(f1_macros)),
        float(np.mean(bal_accs)), float(np.std(bal_accs)),
    )


n_classes = len(NHI_CLASS_NAMES)
cls_tabnet = _cls_metrics(y_test, y_pred_tabnet, y_proba_tabnet, n_classes)
cls_lstm = _cls_metrics(y_te_lstm.astype(int), y_pred_lstm.astype(int), y_proba_lstm, n_classes)
cls_gru = _cls_metrics(y_te_lstm.astype(int), y_pred_gru.astype(int), y_proba_gru, n_classes)
cls_tcn = _cls_metrics(y_te_tcn.astype(int), y_pred_tcn.astype(int), y_proba_tcn, n_classes)
cls_ae = _cls_metrics(y_test, y_pred_ae.astype(int), y_proba_ae, n_classes)

cv_acc_mean, cv_acc_std, cv_f1m_mean, cv_f1m_std, cv_bacc_mean, cv_bacc_std = _tabnet_stratified_cv_summary(
    X, nhi_to_class_with_edges(y_cont.values, eval_edges).astype(int), n_splits=5
)

comparison_df = pd.DataFrame({
    "Model": ["TabNet", "LSTM", "GRU", "TCN", "Autoencoder"],
    "Accuracy": [cls_tabnet[0], cls_lstm[0], cls_gru[0], cls_tcn[0], cls_ae[0]],
    "Precision": [cls_tabnet[1], cls_lstm[1], cls_gru[1], cls_tcn[1], cls_ae[1]],
    "Recall": [cls_tabnet[2], cls_lstm[2], cls_gru[2], cls_tcn[2], cls_ae[2]],
    "F1-Score": [cls_tabnet[3], cls_lstm[3], cls_gru[3], cls_tcn[3], cls_ae[3]],
    "Macro F1": [
        f1_score(y_test, y_pred_tabnet, average="macro", zero_division=0),
        f1_score(y_te_lstm.astype(int), y_pred_lstm.astype(int), average="macro", zero_division=0),
        f1_score(y_te_lstm.astype(int), y_pred_gru.astype(int), average="macro", zero_division=0),
        f1_score(y_te_tcn.astype(int), y_pred_tcn.astype(int), average="macro", zero_division=0),
        f1_score(y_test, y_pred_ae.astype(int), average="macro", zero_division=0),
    ],
    "Balanced Acc": [
        balanced_accuracy_score(y_test, y_pred_tabnet),
        balanced_accuracy_score(y_te_lstm.astype(int), y_pred_lstm.astype(int)),
        balanced_accuracy_score(y_te_lstm.astype(int), y_pred_gru.astype(int)),
        balanced_accuracy_score(y_te_tcn.astype(int), y_pred_tcn.astype(int)),
        balanced_accuracy_score(y_test, y_pred_ae.astype(int)),
    ],
    "ROC AUC": [cls_tabnet[4], cls_lstm[4], cls_gru[4], cls_tcn[4], cls_ae[4]],
    "FPR": [cls_tabnet[5], cls_lstm[5], cls_gru[5], cls_tcn[5], cls_ae[5]],
    "FNR": [cls_tabnet[6], cls_lstm[6], cls_gru[6], cls_tcn[6], cls_ae[6]],
    "BinScheme": [bin_scheme] * 5,
    "CV Accuracy (mean±std)": [
        f"{cv_acc_mean:.4f}±{cv_acc_std:.4f}",
        np.nan, np.nan, np.nan, np.nan
    ],
    "CV Macro F1 (mean±std)": [
        f"{cv_f1m_mean:.4f}±{cv_f1m_std:.4f}",
        np.nan, np.nan, np.nan, np.nan
    ],
    "CV Balanced Acc (mean±std)": [
        f"{cv_bacc_mean:.4f}±{cv_bacc_std:.4f}",
        np.nan, np.nan, np.nan, np.nan
    ],
})

comparison_df.to_csv(f"{METRICS_DIR}/nhi_model_comparison.csv", index=False)
print("\nModel Comparison:")
print(comparison_df.to_string(index=False))

# Feature importance (TabNet)
imp_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": tabnet_nhi.feature_importances_
}).sort_values("Importance", ascending=False)
imp_df.to_csv(f"{METRICS_DIR}/nhi_feature_importance.csv", index=False)

# Save model + bin metadata
tabnet_nhi.save_model(f"{MODELS_DIR}/nhi_tabnet")
joblib.dump(NHI_CLASS_NAMES, f"{MODELS_DIR}/nhi_class_names.pkl")
joblib.dump(eval_edges, f"{MODELS_DIR}/nhi_bin_edges.pkl")
joblib.dump(bin_scheme, f"{MODELS_DIR}/nhi_bin_scheme.pkl")

print("\n" + "="*60)
print("NHI Classification Analysis Complete!")
print("="*60)
print(f"\nResults saved to:")
print(f"  - Metrics: {METRICS_DIR}")
print(f"  - Models: {MODELS_DIR}")
print("\nModel Performance Summary:")
print(comparison_df.to_string(index=False))
print("\n" + "="*60)

