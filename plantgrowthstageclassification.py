"""
Plant Growth Stage Classification
=================================
This module implements plant growth stage classification using TabNet, LSTM, and TCN models.
The classification task relies on instantaneous soil and environmental states.
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from pytorch_tabnet.tab_model import TabNetClassifier

# ===============================
# Data Loading & Exploration
# ===============================

print("Loading dataset for Plant Growth Stage Classification...")
df = pd.read_excel(DATA_PATH)

print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isna().sum()}")

# Check if growth_stage column exists, if not create based on NHI or other features
if 'growth_stage' not in df.columns:
    # Check if 'Plant Growth Stage' exists (with different naming)
    if 'Plant Growth Stage' in df.columns:
        df['growth_stage'] = df['Plant Growth Stage'].astype(str).str.strip()
        print("\nUsing existing 'Plant Growth Stage' column")
    elif 'NHI' in df.columns:
        df['growth_stage'] = pd.cut(
            df['NHI'],
            bins=[0, 30, 60, 100],
            labels=['Seedling', 'Vegetative', 'Flowering'],
            duplicates='drop'
        )
        print("\nGrowth stage created based on NHI levels")
    else:
        # Use NPK combination to infer stages
        npk_sum = df['nitrogen'] + df['phosphorus'] + df['potassium']
        # Ensure unique bin edges
        q33 = npk_sum.quantile(0.33)
        q67 = npk_sum.quantile(0.67)
        max_val = npk_sum.max()
        
        # Create bins with unique edges
        bins = [npk_sum.min()]
        if q33 > bins[-1]:
            bins.append(q33)
        if q67 > bins[-1]:
            bins.append(q67)
        if max_val > bins[-1]:
            bins.append(max_val)
        
        # Ensure at least 3 bins for 3 labels
        if len(bins) < 4:
            bins = [npk_sum.min(), npk_sum.quantile(0.33), npk_sum.quantile(0.67), npk_sum.max()]
            bins = sorted(list(set(bins)))  # Remove duplicates and sort
        
        if len(bins) >= 4:
            df['growth_stage'] = pd.cut(
                npk_sum,
                bins=bins[:4],  # Use first 4 unique bins
                labels=['Seedling', 'Vegetative', 'Flowering'],
                duplicates='drop'
            )
        else:
            # Fallback: use equal-width bins
            df['growth_stage'] = pd.cut(
                npk_sum,
                bins=3,
                labels=['Seedling', 'Vegetative', 'Flowering'],
                duplicates='drop'
            )
        print("\nGrowth stage created based on NPK levels")

# Clean up growth_stage values
df['growth_stage'] = df['growth_stage'].astype(str).str.strip()

# Remove any rows with NaN or 'nan' growth stages
df = df[df['growth_stage'].notna() & (df['growth_stage'] != 'nan')]

print(f"\nGrowth Stage Distribution:\n{df['growth_stage'].value_counts()}")

# ===============================
# Data Engineering & Visualizations
# ===============================

# Growth stage distribution
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
stage_counts = df['growth_stage'].value_counts()
plt.bar(stage_counts.index.astype(str), stage_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.title("Distribution of Growth Stages")
plt.xlabel("Growth Stage")
plt.ylabel("Count")
plt.xticks(rotation=45)

# Nutrient levels by growth stage
plt.subplot(2, 2, 2)
df_melted = df.melt(
    id_vars=['growth_stage'],
    value_vars=['nitrogen', 'phosphorus', 'potassium'],
    var_name='Nutrient',
    value_name='Level'
)
sns.boxplot(data=df_melted, x='growth_stage', y='Level', hue='Nutrient')
plt.title("Nutrient Levels by Growth Stage")
plt.xlabel("Growth Stage")
plt.ylabel("Nutrient Level")
plt.xticks(rotation=45)
plt.legend(title='Nutrient')

# Environmental factors by growth stage
plt.subplot(2, 2, 3)
env_melted = df.melt(
    id_vars=['growth_stage'],
    value_vars=['temperature', 'moisture', 'conductivity'],
    var_name='Factor',
    value_name='Value'
)
sns.boxplot(data=env_melted, x='growth_stage', y='Value', hue='Factor')
plt.title("Environmental Factors by Growth Stage")
plt.xlabel("Growth Stage")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.legend(title='Factor')

# pH distribution by growth stage
plt.subplot(2, 2, 4)
sns.boxplot(data=df, x='growth_stage', y='pH')
plt.title("pH Distribution by Growth Stage")
plt.xlabel("Growth Stage")
plt.ylabel("pH")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/growth_stage_distribution.png", dpi=300)
plt.close()

# Correlation heatmap by growth stage
stages = df['growth_stage'].unique()
fig, axes = plt.subplots(1, len(stages), figsize=(5*len(stages), 4))
if len(stages) == 1:
    axes = [axes]

for i, stage in enumerate(stages):
    stage_df = df[df['growth_stage'] == stage]
    corr_cols = ["nitrogen", "phosphorus", "potassium",
                 "conductivity", "moisture", "temperature", "pH"]
    corr_matrix = stage_df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f",
                square=True, ax=axes[i], cbar_kws={'shrink': 0.8})
    axes[i].set_title(f"Correlation Matrix - {stage}")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/growth_stage_correlation_matrices.png", dpi=300)
plt.close()

# Feature importance visualization by stage
plt.figure(figsize=(12, 6))
features = ["nitrogen", "phosphorus", "potassium", "conductivity", "moisture", "temperature", "pH"]
stage_means = df.groupby('growth_stage')[features].mean()
sns.heatmap(stage_means.T, annot=True, cmap="YlOrRd", fmt=".2f", cbar_kws={'label': 'Mean Value'})
plt.title("Feature Means by Growth Stage")
plt.xlabel("Growth Stage")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/growth_stage_feature_means.png", dpi=300)
plt.close()

# ===============================
# Feature Engineering
# ===============================

FEATURES = [
    "nitrogen", "phosphorus", "potassium",
    "conductivity", "moisture", "temperature", "pH"
]
TARGET = "growth_stage"

# Remove rows with missing target
df_clean = df[FEATURES + [TARGET]].dropna()

X = df_clean[FEATURES]
y = df_clean[TARGET]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

joblib.dump(le, f"{MODELS_DIR}/growth_stage_encoder.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Ensure y_train and y_test are numpy arrays (not pandas Series)
y_train = np.asarray(y_train, dtype=np.int64)
y_test = np.asarray(y_test, dtype=np.int64)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

joblib.dump(scaler, f"{MODELS_DIR}/growth_stage_scaler.pkl")

# ===============================
# TabNet Classifier
# ===============================

print("\n" + "="*60)
print("Training TabNet Classifier for Growth Stage...")
print("="*60)

n_classes = len(class_names)
tabnet_classifier = TabNetClassifier(
    n_d=16, n_a=16, n_steps=5,
    optimizer_params=dict(lr=0.02),
    mask_type="entmax"
)

# TabNetClassifier.fit expects 1D array for y (unlike TabNetRegressor which expects 2D)
# Ensure y_train and y_test are 1D numpy arrays
y_train_1d = y_train.ravel() if y_train.ndim > 1 else y_train
y_test_1d = y_test.ravel() if y_test.ndim > 1 else y_test

tabnet_classifier.fit(
    X_train_s,
    y_train_1d,
    eval_set=[(X_test_s, y_test_1d)],
    eval_name=["test"],
    eval_metric=["accuracy"],
    max_epochs=200,
    patience=30
)

y_pred_tabnet = tabnet_classifier.predict(X_test_s).flatten()

# ===============================
# LSTM Classifier
# ===============================

print("\n" + "="*60)
print("Training LSTM Classifier for Growth Stage...")
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
    def __init__(self, input_size, hidden=64, num_layers=2, num_classes=n_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

lstm_classifier = LSTMClassifier(X_tr_lstm.shape[2]).to(device)
optimizer = torch.optim.Adam(lstm_classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

X_tr_t = torch.tensor(X_tr_lstm, dtype=torch.float32).to(device)
y_tr_t = torch.tensor(y_tr_lstm, dtype=torch.long).to(device)

# Training loop
lstm_classifier.train()
for epoch in range(50):
    optimizer.zero_grad()
    preds = lstm_classifier(X_tr_t)
    loss = criterion(preds, y_tr_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lstm_classifier.parameters(), 1.0)
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")

lstm_classifier.eval()
X_te_t = torch.tensor(X_te_lstm, dtype=torch.float32).to(device)
with torch.no_grad():
    logits_lstm = lstm_classifier(X_te_t)
    y_proba_lstm = torch.softmax(logits_lstm, dim=1).cpu().numpy()
    y_pred_lstm = np.argmax(y_proba_lstm, axis=1)

torch.save(lstm_classifier.state_dict(), f"{MODELS_DIR}/growth_stage_lstm.pt")

# ===============================
# GRU Classifier
# ===============================

print("\n" + "="*60)
print("Training GRU Classifier for Growth Stage...")
print("="*60)

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden=64, num_layers=2, num_classes=n_classes):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

gru_classifier = GRUClassifier(X_tr_lstm.shape[2]).to(device)
optimizer_gru = torch.optim.Adam(gru_classifier.parameters(), lr=0.001)

X_tr_t = torch.tensor(X_tr_lstm, dtype=torch.float32).to(device)
y_tr_t = torch.tensor(y_tr_lstm, dtype=torch.long).to(device)

gru_classifier.train()
for epoch in range(50):
    optimizer_gru.zero_grad()
    preds = gru_classifier(X_tr_t)
    loss = criterion(preds, y_tr_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gru_classifier.parameters(), 1.0)
    optimizer_gru.step()
    if (epoch + 1) % 10 == 0:
        print(f"[GRU] Epoch {epoch+1}/50, Loss: {loss.item():.4f}")

gru_classifier.eval()
X_te_t = torch.tensor(X_te_lstm, dtype=torch.float32).to(device)
with torch.no_grad():
    logits_gru = gru_classifier(X_te_t)
    y_proba_gru = torch.softmax(logits_gru, dim=1).cpu().numpy()
    y_pred_gru = np.argmax(y_proba_gru, axis=1)

torch.save(gru_classifier.state_dict(), f"{MODELS_DIR}/growth_stage_gru.pt")

# ===============================
# Autoencoder + Classifier (supervised)
# ===============================

print("\n" + "="*60)
print("Training Autoencoder + Classifier for Growth Stage...")
print("="*60)

class AutoencoderClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, latent_dim=16, hidden_dim=64):
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

ae_classifier = AutoencoderClassifier(input_dim=X_train_s.shape[1], num_classes=n_classes).to(device)
opt_ae = torch.optim.Adam(ae_classifier.parameters(), lr=0.001)
mse = nn.MSELoss()
ce = nn.CrossEntropyLoss()

X_tr_ae = torch.tensor(X_train_s, dtype=torch.float32).to(device)
y_tr_ae = torch.tensor(y_train_1d, dtype=torch.long).to(device)
X_te_ae = torch.tensor(X_test_s, dtype=torch.float32).to(device)

ae_classifier.train()
alpha = 0.3
for epoch in range(120):
    opt_ae.zero_grad()
    x_hat, logits = ae_classifier(X_tr_ae)
    loss_recon = mse(x_hat, X_tr_ae)
    loss_cls = ce(logits, y_tr_ae)
    loss = loss_recon + alpha * loss_cls
    loss.backward()
    opt_ae.step()

ae_classifier.eval()
with torch.no_grad():
    _, logits_ae = ae_classifier(X_te_ae)
    y_proba_ae = torch.softmax(logits_ae, dim=1).cpu().numpy()
    y_pred_ae = np.argmax(y_proba_ae, axis=1)

torch.save(ae_classifier.state_dict(), f"{MODELS_DIR}/growth_stage_autoencoder.pt")

# ===============================
# TCN Classifier
# ===============================

print("\n" + "="*60)
print("Training TCN Classifier for Growth Stage...")
print("="*60)

SEQ_LEN_TCN = 5
X_seq_tcn, y_seq_tcn = create_sequences(X_train_s, y_train, SEQ_LEN_TCN)

split = int(0.8 * len(X_seq_tcn))
X_tr_tcn, X_te_tcn = X_seq_tcn[:split], X_seq_tcn[split:]
y_tr_tcn, y_te_tcn = y_seq_tcn[:split], y_seq_tcn[split:]

class TCNClassifier(nn.Module):
    def __init__(self, input_size, channels=64, kernel=3, num_layers=2, num_classes=n_classes):
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

tcn_classifier = TCNClassifier(X_tr_tcn.shape[2]).to(device)
optimizer = torch.optim.Adam(tcn_classifier.parameters(), lr=0.001)

X_tr_t = torch.tensor(X_tr_tcn, dtype=torch.float32).to(device)
y_tr_t = torch.tensor(y_tr_tcn, dtype=torch.long).to(device)

# Training loop
tcn_classifier.train()
for epoch in range(50):
    optimizer.zero_grad()
    preds = tcn_classifier(X_tr_t)
    loss = criterion(preds, y_tr_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(tcn_classifier.parameters(), 1.0)
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")

tcn_classifier.eval()
X_te_t = torch.tensor(X_te_tcn, dtype=torch.float32).to(device)
with torch.no_grad():
    logits_tcn = tcn_classifier(X_te_t)
    y_proba_tcn = torch.softmax(logits_tcn, dim=1).cpu().numpy()
    y_pred_tcn = np.argmax(y_proba_tcn, axis=1)

torch.save(tcn_classifier.state_dict(), f"{MODELS_DIR}/growth_stage_tcn.pt")

# ===============================
# Model Evaluation & Comparison
# ===============================

# Prepare test sets for comparison (need to align with sequence-based models)
# For TabNet, use full test set
# For LSTM/TCN, use sequence test sets

# Create sequence test sets aligned with original test indices
test_indices = X_test.index
test_seq_indices = [i for i in range(len(X_train_s)) if i + SEQ_LEN_LSTM < len(X_train_s)]
test_seq_indices = [i for i in test_seq_indices if i + SEQ_LEN_LSTM in test_indices.tolist()]

if len(test_seq_indices) > 0:
    X_test_seq_lstm, y_test_seq_lstm = create_sequences(
        X_train_s[test_seq_indices], 
        y_train[test_seq_indices], 
        SEQ_LEN_LSTM
    )
else:
    # Fallback: use validation split
    y_test_seq_lstm = y_te_lstm


def _fpr_fnr_macro(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fprs = []
    fnrs = []
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


def _roc_auc_ovr_macro(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int):
    try:
        if y_proba is None or y_proba.ndim != 2 or y_proba.shape[1] != n_classes:
            return float("nan")
        # For binary, roc_auc_score expects scores for the positive class or shape (n,2)
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


# TabNet probabilities for AUC/FPR/FNR
y_proba_tabnet = tabnet_classifier.predict_proba(X_test_s)

fpr_tabnet, fnr_tabnet = _fpr_fnr_macro(y_test, y_pred_tabnet, n_classes)
fpr_lstm, fnr_lstm = _fpr_fnr_macro(y_te_lstm, y_pred_lstm, n_classes)
fpr_gru, fnr_gru = _fpr_fnr_macro(y_te_lstm, y_pred_gru, n_classes)
fpr_tcn, fnr_tcn = _fpr_fnr_macro(y_te_tcn, y_pred_tcn, n_classes)
fpr_ae, fnr_ae = _fpr_fnr_macro(y_test, y_pred_ae, n_classes)

auc_tabnet = _roc_auc_ovr_macro(y_test, y_proba_tabnet, n_classes)
auc_lstm = _roc_auc_ovr_macro(y_te_lstm, y_proba_lstm, n_classes)
auc_gru = _roc_auc_ovr_macro(y_te_lstm, y_proba_gru, n_classes)
auc_tcn = _roc_auc_ovr_macro(y_te_tcn, y_proba_tcn, n_classes)
auc_ae = _roc_auc_ovr_macro(y_test, y_proba_ae, n_classes)

comparison_df = pd.DataFrame({
    "Model": ["TabNet", "LSTM", "GRU", "TCN", "Autoencoder"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_tabnet),
        accuracy_score(y_te_lstm, y_pred_lstm),
        accuracy_score(y_te_lstm, y_pred_gru),
        accuracy_score(y_te_tcn, y_pred_tcn)
        , accuracy_score(y_test, y_pred_ae)
    ],
    "Precision": [
        precision_score(y_test, y_pred_tabnet, average='weighted', zero_division=0),
        precision_score(y_te_lstm, y_pred_lstm, average='weighted', zero_division=0),
        precision_score(y_te_lstm, y_pred_gru, average='weighted', zero_division=0),
        precision_score(y_te_tcn, y_pred_tcn, average='weighted', zero_division=0)
        , precision_score(y_test, y_pred_ae, average='weighted', zero_division=0)
    ],
    "Recall": [
        recall_score(y_test, y_pred_tabnet, average='weighted', zero_division=0),
        recall_score(y_te_lstm, y_pred_lstm, average='weighted', zero_division=0),
        recall_score(y_te_lstm, y_pred_gru, average='weighted', zero_division=0),
        recall_score(y_te_tcn, y_pred_tcn, average='weighted', zero_division=0)
        , recall_score(y_test, y_pred_ae, average='weighted', zero_division=0)
    ],
    "F1-Score": [
        f1_score(y_test, y_pred_tabnet, average='weighted', zero_division=0),
        f1_score(y_te_lstm, y_pred_lstm, average='weighted', zero_division=0),
        f1_score(y_te_lstm, y_pred_gru, average='weighted', zero_division=0),
        f1_score(y_te_tcn, y_pred_tcn, average='weighted', zero_division=0)
        , f1_score(y_test, y_pred_ae, average='weighted', zero_division=0)
    ],
    "ROC AUC": [
        auc_tabnet,
        auc_lstm,
        auc_gru,
        auc_tcn,
        auc_ae
    ],
    "FPR": [
        fpr_tabnet,
        fpr_lstm,
        fpr_gru,
        fpr_tcn,
        fpr_ae
    ],
    "FNR": [
        fnr_tabnet,
        fnr_lstm,
        fnr_gru,
        fnr_tcn,
        fnr_ae
    ]
})

comparison_df.to_csv(f"{METRICS_DIR}/growth_stage_model_comparison.csv", index=False)
print("\nModel Comparison:")
print(comparison_df.to_string(index=False))

# Classification reports
print("\n" + "="*60)
print("TabNet Classification Report:")
print("="*60)
print(classification_report(y_test, y_pred_tabnet, target_names=class_names))

# Feature importance
imp_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": tabnet_classifier.feature_importances_
}).sort_values("Importance", ascending=False)

imp_df.to_csv(f"{METRICS_DIR}/growth_stage_feature_importance.csv", index=False)

# ===============================
# Enhanced Visualizations
# ===============================

# Model comparison bar chart
plt.figure(figsize=(14, 5))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.17

for i, model in enumerate(['TabNet', 'LSTM', 'GRU', 'TCN', 'Autoencoder']):
    values = [
        comparison_df[comparison_df['Model'] == model]['Accuracy'].values[0],
        comparison_df[comparison_df['Model'] == model]['Precision'].values[0],
        comparison_df[comparison_df['Model'] == model]['Recall'].values[0],
        comparison_df[comparison_df['Model'] == model]['F1-Score'].values[0]
    ]
    plt.bar(x + i*width, values, width, label=model, alpha=0.8)

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Growth Stage Classification Model Performance Comparison')
plt.xticks(x + width, metrics)
plt.legend()
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/growth_stage_model_comparison.png", dpi=300)
plt.close()

# Confusion matrices
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
predictions = [y_pred_tabnet, y_pred_lstm, y_pred_gru, y_pred_tcn, y_pred_ae]
targets = [y_test, y_te_lstm, y_te_lstm, y_te_tcn, y_test]
model_names = ['TabNet', 'LSTM', 'GRU', 'TCN', 'Autoencoder']

for i, (pred, target, name) in enumerate(zip(predictions, targets, model_names)):
    cm = confusion_matrix(target, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=class_names, yticklabels=class_names)
    axes[i].set_title(f'{name} Confusion Matrix')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/growth_stage_confusion_matrices.png", dpi=300)
plt.close()

# Feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=imp_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance for Growth Stage Classification (TabNet)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/growth_stage_feature_importance.png", dpi=300)
plt.close()

# Save model
tabnet_classifier.save_model(f"{MODELS_DIR}/growth_stage_tabnet")

# Prediction function
def predict_growth_stage(sensor_input: dict):
    """Predict growth stage from sensor inputs"""
    df_in = pd.DataFrame([sensor_input])[FEATURES]
    x = scaler.transform(df_in)
    pred_encoded = tabnet_classifier.predict(x)[0][0]
    return le.inverse_transform([pred_encoded])[0]

print("\n" + "="*60)
print("Plant Growth Stage Classification Analysis Complete!")
print("="*60)
print(f"\nResults saved to:")
print(f"  - Plots: {PLOTS_DIR}")
print(f"  - Metrics: {METRICS_DIR}")
print(f"  - Models: {MODELS_DIR}")
print("\nModel Performance Summary:")
print(comparison_df.to_string(index=False))
print("\n" + "="*60)

