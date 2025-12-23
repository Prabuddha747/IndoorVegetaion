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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from pytorch_tabnet.tab_model import TabNetRegressor
df = pd.read_excel(DATA_PATH)

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
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

joblib.dump(scaler, f"{MODELS_DIR}/ph_scaler.pkl")


tabnet_ph = TabNetRegressor(
    n_d=16, n_a=16, n_steps=5,
    optimizer_params=dict(lr=0.02),
    mask_type="entmax"
)

tabnet_ph.fit(
    X_train_s,
    y_train.values.reshape(-1,1),
    eval_set=[(X_test_s, y_test.values.reshape(-1,1))],
    max_epochs=200,
    patience=30
)

y_pred_tabnet = tabnet_ph.predict(X_test_s).flatten()


def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y.iloc[i+seq_len])
    return np.array(Xs), np.array(ys)


SEQ_LEN_LSTM = 10
X_seq, y_seq = create_sequences(X_train_s, y_train, SEQ_LEN_LSTM)

split = int(0.8 * len(X_seq))
X_tr_lstm, X_te_lstm = X_seq[:split], X_seq[split:]
y_tr_lstm, y_te_lstm = y_seq[:split], y_seq[split:]

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

device = "cuda" if torch.cuda.is_available() else "cpu"

lstm_model = LSTMRegressor(X_tr_lstm.shape[2]).to(device)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

X_tr_t = torch.tensor(X_tr_lstm, dtype=torch.float32).to(device)
y_tr_t = torch.tensor(y_tr_lstm, dtype=torch.float32).to(device)

for epoch in range(30):
    optimizer.zero_grad()
    preds = lstm_model(X_tr_t).squeeze()
    loss = criterion(preds, y_tr_t)
    loss.backward()
    optimizer.step()

X_te_t = torch.tensor(X_te_lstm, dtype=torch.float32).to(device)
with torch.no_grad():
    y_pred_lstm = lstm_model(X_te_t).cpu().numpy().flatten()

torch.save(lstm_model.state_dict(), f"{MODELS_DIR}/ph_lstm.pt")


SEQ_LEN_TCN = 5
X_seq_tcn, y_seq_tcn = create_sequences(X_train_s, y_train, SEQ_LEN_TCN)

split = int(0.8 * len(X_seq_tcn))
X_tr_tcn, X_te_tcn = X_seq_tcn[:split], X_seq_tcn[split:]
y_tr_tcn, y_te_tcn = y_seq_tcn[:split], y_seq_tcn[split:]

class TCNRegressor(nn.Module):
    def __init__(self, input_size, channels=64, kernel=3):
        super().__init__()
        self.conv = weight_norm(
            nn.Conv1d(input_size, channels, kernel, padding=kernel//2)
        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.relu(self.conv(x))
        x = x.mean(dim=2)
        return self.fc(x)

tcn_model = TCNRegressor(X_tr_tcn.shape[2]).to(device)
optimizer = torch.optim.Adam(tcn_model.parameters(), lr=0.001)

X_tr_t = torch.tensor(X_tr_tcn, dtype=torch.float32).to(device)
y_tr_t = torch.tensor(y_tr_tcn, dtype=torch.float32).to(device)

for epoch in range(30):
    optimizer.zero_grad()
    preds = tcn_model(X_tr_t).squeeze()
    loss = criterion(preds, y_tr_t)
    loss.backward()
    optimizer.step()

X_te_t = torch.tensor(X_te_tcn, dtype=torch.float32).to(device)
with torch.no_grad():
    y_pred_tcn = tcn_model(X_te_t).cpu().numpy().flatten()

torch.save(tcn_model.state_dict(), f"{MODELS_DIR}/ph_tcn.pt")


comparison_df = pd.DataFrame({
    "Model": ["TabNet", "LSTM", "TCN"],
    "MAE": [
        mean_absolute_error(y_test, y_pred_tabnet),
        mean_absolute_error(y_te_lstm, y_pred_lstm),
        mean_absolute_error(y_te_tcn, y_pred_tcn)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, y_pred_tabnet)),
        np.sqrt(mean_squared_error(y_te_lstm, y_pred_lstm)),
        np.sqrt(mean_squared_error(y_te_tcn, y_pred_tcn))
    ],
    "R2": [
        r2_score(y_test, y_pred_tabnet),
        r2_score(y_te_lstm, y_pred_lstm),
        r2_score(y_te_tcn, y_pred_tcn)
    ]
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
    return float(tabnet_ph.predict(x)[0][0])

example = {
    "nitrogen": 42,
    "phosphorus": 30,
    "potassium": 38,
    "conductivity": 1.2,
    "moisture": 28,
    "temperature": 26
}

print("Predicted pH:", round(predict_ph(example), 2))

tabnet_ph.save_model(f"{MODELS_DIR}/ph_tabnet")

# ===============================
# Enhanced Visualizations & Insights
# ===============================

# Model comparison visualization
plt.figure(figsize=(12, 5))
metrics = ['MAE', 'RMSE', 'R2']
x = np.arange(len(metrics))
width = 0.25

for i, model in enumerate(['TabNet', 'LSTM', 'TCN']):
    values = [
        comparison_df[comparison_df['Model'] == model]['MAE'].values[0],
        comparison_df[comparison_df['Model'] == model]['RMSE'].values[0],
        comparison_df[comparison_df['Model'] == model]['R2'].values[0]
    ]
    plt.bar(x + i*width, values, width, label=model, alpha=0.8)

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x + width, metrics)
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/ph_model_comparison.png", dpi=300)
plt.close()

# Prediction scatter plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
predictions = [y_pred_tabnet, y_pred_lstm, y_pred_tcn]
targets = [y_test, y_te_lstm, y_te_tcn]
model_names = ['TabNet', 'LSTM', 'TCN']

for i, (pred, target, name) in enumerate(zip(predictions, targets, model_names)):
    axes[i].scatter(target, pred, alpha=0.5)
    axes[i].plot([target.min(), target.max()], [target.min(), target.max()], 'r--', lw=2)
    axes[i].set_xlabel('Actual pH')
    axes[i].set_ylabel('Predicted pH')
    axes[i].set_title(f'{name} Predictions')
    r2 = r2_score(target, pred)
    axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/ph_predictions_scatter.png", dpi=300)
plt.close()

# Feature importance visualization
plt.figure(figsize=(10, 6))
sns.barplot(data=imp_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance for pH Prediction (TabNet)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/ph_feature_importance.png", dpi=300)
plt.close()

# Residual analysis
residuals_tabnet = y_test - y_pred_tabnet
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_pred_tabnet, residuals_tabnet, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted pH')
plt.ylabel('Residuals')
plt.title('Residual Plot - TabNet')

plt.subplot(1, 2, 2)
plt.hist(residuals_tabnet, bins=30, edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution - TabNet')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/ph_residual_analysis.png", dpi=300)
plt.close()

# Generate dataset-specific insights
try:
    from data_insights import analyze_dataset_insights
    print("\n" + "="*60)
    print("Generating Dataset-Specific Insights...")
    print("="*60)
    insights = analyze_dataset_insights(df)
    print("Dataset insights saved to:", os.path.join(METRICS_DIR, "dataset_insights.json"))
except ImportError:
    print("Note: data_insights module not available. Install required dependencies.")

print("\n" + "="*60)
print("pH Prediction Analysis Complete!")
print("="*60)
print(f"\nResults saved to:")
print(f"  - Plots: {PLOTS_DIR}")
print(f"  - Metrics: {METRICS_DIR}")
print(f"  - Models: {MODELS_DIR}")
print("\nModel Performance Summary:")
print(comparison_df.to_string(index=False))
print("\n" + "="*60)
