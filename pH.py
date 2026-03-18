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
df = pd.read_excel(DATA_PATH)


PH_CLASS_NAMES = ["Acidic", "Optimal", "Alkaline"]


def ph_to_class(ph_values: np.ndarray) -> np.ndarray:
    # 0: Acidic (<6.0), 1: Optimal (6.0–7.5), 2: Alkaline (>7.5)
    ph_values = np.asarray(ph_values, dtype=float)
    return np.digitize(ph_values, bins=[6.0, 7.5], right=False)


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


def _roc_auc_ovr_macro(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")

print(df.shape)
print(df.columns)
print(df.isna().sum())
plt.figure(figsize=(6,4))
sns.histplot(df["pH"], kde=True, bins=30)
plt.title("Distribution of Soil pH")
plt.xlabel("pH")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/ph_distribution.png", dpi=300)
plt.close()
nutrients = ["nitrogen", "phosphorus", "potassium"]

plt.figure(figsize=(12,4))
for i, col in enumerate(nutrients):
    plt.subplot(1,3,i+1)
    sns.scatterplot(x=df[col], y=df["pH"], alpha=0.4)
    plt.xlabel(col.capitalize())
    plt.ylabel("pH")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/ph_vs_nutrients.png", dpi=300)
plt.close()
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.scatterplot(x=df["conductivity"], y=df["pH"], alpha=0.4)
plt.title("pH vs Conductivity")

plt.subplot(1,2,2)
sns.scatterplot(x=df["moisture"], y=df["pH"], alpha=0.4)
plt.title("pH vs Moisture")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/ph_environmental_factors.png", dpi=300)
plt.close()

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.scatterplot(x=df["conductivity"], y=df["pH"], alpha=0.4)
plt.title("pH vs Conductivity")

plt.subplot(1,2,2)
sns.scatterplot(x=df["moisture"], y=df["pH"], alpha=0.4)
plt.title("pH vs Moisture")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/ph_environmental_factors.png", dpi=300)
plt.close()


plt.figure(figsize=(8,6))
sns.heatmap(
    df[["pH","nitrogen","phosphorus","potassium",
        "conductivity","moisture","temperature"]].corr(),
    annot=True, cmap="coolwarm", fmt=".2f"
)
plt.title("Sensor Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/ph_correlation_matrix.png", dpi=300)
plt.close()


FEATURES = [
    "nitrogen", "phosphorus", "potassium",
    "conductivity", "moisture", "temperature"
]
TARGET = "pH"

X = df[FEATURES]
y_cont = df[TARGET]
y = ph_to_class(y_cont.values).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

joblib.dump(scaler, f"{MODELS_DIR}/ph_scaler.pkl")


tabnet_ph = TabNetClassifier(
    n_d=12, n_a=12, n_steps=4,
    optimizer_params=dict(lr=0.01, weight_decay=1e-4),
    mask_type="entmax",
    lambda_sparse=1e-3
)

tabnet_ph.fit(
    X_train_s,
    y_train,
    eval_set=[(X_test_s, y_test)],
    max_epochs=200,
    patience=30
)

y_pred_tabnet = tabnet_ph.predict(X_test_s).astype(int)
y_pred_tabnet_train = tabnet_ph.predict(X_train_s).astype(int)
y_proba_tabnet = tabnet_ph.predict_proba(X_test_s)


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
    def __init__(self, input_size, hidden=64, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

device = "cuda" if torch.cuda.is_available() else "cpu"

lstm_model = LSTMClassifier(X_tr_lstm.shape[2], num_classes=len(PH_CLASS_NAMES)).to(device)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

X_tr_t = torch.tensor(X_tr_lstm, dtype=torch.float32).to(device)
y_tr_t = torch.tensor(y_tr_lstm, dtype=torch.long).to(device)

for epoch in range(25):
    optimizer.zero_grad()
    logits = lstm_model(X_tr_t)
    loss = criterion(logits, y_tr_t)
    loss.backward()
    optimizer.step()

X_te_t = torch.tensor(X_te_lstm, dtype=torch.float32).to(device)
with torch.no_grad():
    logits = lstm_model(X_te_t)
    y_proba_lstm = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred_lstm = np.argmax(y_proba_lstm, axis=1).astype(int)

torch.save(lstm_model.state_dict(), f"{MODELS_DIR}/ph_lstm.pt")

# ===============================
# GRU Model
# ===============================

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden=64, num_layers=2, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

gru_model = GRUClassifier(X_tr_lstm.shape[2], num_classes=len(PH_CLASS_NAMES)).to(device)
optimizer_gru = torch.optim.Adam(gru_model.parameters(), lr=0.001, weight_decay=1e-4)

X_tr_t = torch.tensor(X_tr_lstm, dtype=torch.float32).to(device)
y_tr_t = torch.tensor(y_tr_lstm, dtype=torch.long).to(device)

gru_model.train()
for epoch in range(25):
    optimizer_gru.zero_grad()
    logits = gru_model(X_tr_t)
    loss = criterion(logits, y_tr_t)
    loss.backward()
    optimizer_gru.step()

gru_model.eval()
X_te_t = torch.tensor(X_te_lstm, dtype=torch.float32).to(device)
with torch.no_grad():
    logits = gru_model(X_te_t)
    y_proba_gru = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred_gru = np.argmax(y_proba_gru, axis=1).astype(int)

torch.save(gru_model.state_dict(), f"{MODELS_DIR}/ph_gru.pt")

# ===============================
# Autoencoder + Regressor (supervised)
# ===============================

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

ae_model = AutoencoderClassifier(input_dim=X_train_s.shape[1], num_classes=len(PH_CLASS_NAMES)).to(device)
opt_ae = torch.optim.Adam(ae_model.parameters(), lr=0.001, weight_decay=1e-4)
mse = nn.MSELoss()
ce = nn.CrossEntropyLoss()

X_tr_ae = torch.tensor(X_train_s, dtype=torch.float32).to(device)
y_tr_ae = torch.tensor(y_train, dtype=torch.long).to(device)
X_te_ae = torch.tensor(X_test_s, dtype=torch.float32).to(device)

ae_model.train()
alpha = 0.3  # weight for supervised loss
for epoch in range(35):
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

torch.save(ae_model.state_dict(), f"{MODELS_DIR}/ph_autoencoder.pt")


SEQ_LEN_TCN = 5
X_seq_tcn, y_seq_tcn = create_sequences(X_train_s, y_train, SEQ_LEN_TCN)

split = int(0.8 * len(X_seq_tcn))
X_tr_tcn, X_te_tcn = X_seq_tcn[:split], X_seq_tcn[split:]
y_tr_tcn, y_te_tcn = y_seq_tcn[:split], y_seq_tcn[split:]

class TCNClassifier(nn.Module):
    def __init__(self, input_size, channels=64, kernel=3, num_classes=3):
        super().__init__()
        self.conv = weight_norm(
            nn.Conv1d(input_size, channels, kernel, padding=kernel//2)
        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.relu(self.conv(x))
        x = x.mean(dim=2)
        return self.fc(x)

tcn_model = TCNClassifier(X_tr_tcn.shape[2], num_classes=len(PH_CLASS_NAMES)).to(device)
optimizer = torch.optim.Adam(tcn_model.parameters(), lr=0.001, weight_decay=1e-4)

X_tr_t = torch.tensor(X_tr_tcn, dtype=torch.float32).to(device)
y_tr_t = torch.tensor(y_tr_tcn, dtype=torch.long).to(device)

for epoch in range(25):
    optimizer.zero_grad()
    logits = tcn_model(X_tr_t)
    loss = criterion(logits, y_tr_t)
    loss.backward()
    optimizer.step()

X_te_t = torch.tensor(X_te_tcn, dtype=torch.float32).to(device)
with torch.no_grad():
    logits = tcn_model(X_te_t)
    y_proba_tcn = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred_tcn = np.argmax(y_proba_tcn, axis=1).astype(int)

torch.save(tcn_model.state_dict(), f"{MODELS_DIR}/ph_tcn.pt")

def _cls_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None, n_classes: int):
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    fpr, fnr = _fpr_fnr_macro(y_true, y_pred, n_classes=n_classes)
    auc = _roc_auc_ovr_macro(y_true, y_proba) if y_proba is not None else float("nan")
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
        m.fit(x_tr_s, y_tr, eval_set=[(x_va_s, y_va)], max_epochs=60, patience=10)
        pred = m.predict(x_va_s).astype(int)
        accs.append(accuracy_score(y_va, pred))
        f1_macros.append(f1_score(y_va, pred, average="macro", zero_division=0))
        bal_accs.append(balanced_accuracy_score(y_va, pred))
    return (
        float(np.mean(accs)), float(np.std(accs)),
        float(np.mean(f1_macros)), float(np.std(f1_macros)),
        float(np.mean(bal_accs)), float(np.std(bal_accs)),
    )


n_ph_classes = len(PH_CLASS_NAMES)
cls_tabnet = _cls_metrics(y_test, y_pred_tabnet, y_proba_tabnet, n_ph_classes)
cls_lstm = _cls_metrics(y_te_lstm, y_pred_lstm, y_proba_lstm, n_ph_classes)
cls_gru = _cls_metrics(y_te_lstm, y_pred_gru, y_proba_gru, n_ph_classes)
cls_tcn = _cls_metrics(y_te_tcn, y_pred_tcn, y_proba_tcn, n_ph_classes)
cls_ae = _cls_metrics(y_test, y_pred_ae, y_proba_ae, n_ph_classes)
cv_acc_mean, cv_acc_std, cv_f1m_mean, cv_f1m_std, cv_bacc_mean, cv_bacc_std = _tabnet_stratified_cv_summary(
    X, y, n_splits=5
)


comparison_df = pd.DataFrame({
    "Model": ["TabNet", "LSTM", "GRU", "TCN", "Autoencoder"],
    "Accuracy": [cls_tabnet[0], cls_lstm[0], cls_gru[0], cls_tcn[0], cls_ae[0]],
    "Precision": [cls_tabnet[1], cls_lstm[1], cls_gru[1], cls_tcn[1], cls_ae[1]],
    "Recall": [cls_tabnet[2], cls_lstm[2], cls_gru[2], cls_tcn[2], cls_ae[2]],
    "F1-Score": [cls_tabnet[3], cls_lstm[3], cls_gru[3], cls_tcn[3], cls_ae[3]],
    "Macro F1": [
        f1_score(y_test, y_pred_tabnet, average="macro", zero_division=0),
        f1_score(y_te_lstm, y_pred_lstm, average="macro", zero_division=0),
        f1_score(y_te_lstm, y_pred_gru, average="macro", zero_division=0),
        f1_score(y_te_tcn, y_pred_tcn, average="macro", zero_division=0),
        f1_score(y_test, y_pred_ae, average="macro", zero_division=0),
    ],
    "Balanced Acc": [
        balanced_accuracy_score(y_test, y_pred_tabnet),
        balanced_accuracy_score(y_te_lstm, y_pred_lstm),
        balanced_accuracy_score(y_te_lstm, y_pred_gru),
        balanced_accuracy_score(y_te_tcn, y_pred_tcn),
        balanced_accuracy_score(y_test, y_pred_ae),
    ],
    "ROC AUC": [cls_tabnet[4], cls_lstm[4], cls_gru[4], cls_tcn[4], cls_ae[4]],
    "FPR": [cls_tabnet[5], cls_lstm[5], cls_gru[5], cls_tcn[5], cls_ae[5]],
    "FNR": [cls_tabnet[6], cls_lstm[6], cls_gru[6], cls_tcn[6], cls_ae[6]],
    "CV Accuracy (mean±std)": [f"{cv_acc_mean:.4f}±{cv_acc_std:.4f}", np.nan, np.nan, np.nan, np.nan],
    "CV Macro F1 (mean±std)": [f"{cv_f1m_mean:.4f}±{cv_f1m_std:.4f}", np.nan, np.nan, np.nan, np.nan],
    "CV Balanced Acc (mean±std)": [f"{cv_bacc_mean:.4f}±{cv_bacc_std:.4f}", np.nan, np.nan, np.nan, np.nan],
})

comparison_df.to_csv(f"{METRICS_DIR}/ph_model_comparison.csv", index=False)
print(comparison_df)


imp_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": tabnet_ph.feature_importances_
}).sort_values("Importance", ascending=False)

imp_df.to_csv(f"{METRICS_DIR}/ph_feature_importance.csv", index=False)


def predict_ph(sensor_input: dict):
    df_in = pd.DataFrame([sensor_input])[FEATURES]
    x = scaler.transform(df_in)
    pred_class = int(tabnet_ph.predict(x)[0])
    edges = [6.0, 7.5]
    centers = [edges[0] - 0.5, (edges[0] + edges[1]) / 2.0, edges[1] + 0.5]
    return pred_class, PH_CLASS_NAMES[pred_class], float(centers[pred_class])

example = {
    "nitrogen": 42,
    "phosphorus": 30,
    "potassium": 38,
    "conductivity": 1.2,
    "moisture": 28,
    "temperature": 26
}

pred_class, pred_label, pred_center = predict_ph(example)
print(f"Predicted pH class: {pred_label} (class={pred_class}), representative pH≈{pred_center:.2f}")

tabnet_ph.save_model(f"{MODELS_DIR}/ph_tabnet")
joblib.dump(PH_CLASS_NAMES, f"{MODELS_DIR}/ph_class_names.pkl")
joblib.dump([6.0, 7.5], f"{MODELS_DIR}/ph_bin_edges.pkl")

# ===============================
# Enhanced Visualizations & Insights
# ===============================

# Model comparison visualization
plt.figure(figsize=(12, 5))
metrics = ['Accuracy', 'F1-Score', 'ROC AUC']
x = np.arange(len(metrics))
width = 0.17

for i, model in enumerate(['TabNet', 'LSTM', 'GRU', 'TCN', 'Autoencoder']):
    values = [
        comparison_df[comparison_df['Model'] == model]['Accuracy'].values[0],
        comparison_df[comparison_df['Model'] == model]['F1-Score'].values[0],
        comparison_df[comparison_df['Model'] == model]['ROC AUC'].values[0],
    ]
    plt.bar(x + i*width, values, width, label=model, alpha=0.8)

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x + width, metrics)
plt.legend()
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/ph_model_comparison.png", dpi=300)
plt.close()

try:
    # Confusion matrix (TabNet)
    cm = confusion_matrix(y_test, y_pred_tabnet, labels=list(range(len(PH_CLASS_NAMES))))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=PH_CLASS_NAMES, yticklabels=PH_CLASS_NAMES)
    plt.title('Confusion Matrix - TabNet')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/ph_confusion_matrix_tabnet.png", dpi=300)
    plt.close()
except Exception:
    pass

# Feature importance visualization
plt.figure(figsize=(10, 6))
sns.barplot(data=imp_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance for pH Prediction (TabNet)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/ph_feature_importance.png", dpi=300)
plt.close()

print("\n" + "="*60)
print("NHI Prediction Analysis Complete!")
print("="*60)
print(f"\nResults saved to:")
print(f"  - Plots: {PLOTS_DIR}")
print(f"  - Metrics: {METRICS_DIR}")
print(f"  - Models: {MODELS_DIR}")
print("\nModel Performance Summary:")
print(comparison_df.to_string(index=False))
print("\n" + "="*60)
