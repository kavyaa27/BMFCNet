import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from BMFCNet2 import process_dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)

# Load and preprocess data
data_dir = r"Depression dataset"
label_file = r"labels_processed.csv"
df_labels = pd.read_csv(label_file)

X, y = process_dataset(data_dir, df_labels)
X = np.transpose(X, (0, 2, 1))  # Ensure shape is (samples, time, channels)

# Split data (adjust test_size and stratify as needed)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Load your test data here ===
# Replace with your actual test data
# Example: X_test, y_test = ...
# Ensure X_test is shaped (samples, time, channels)
# X_test, y_test = ...

# Dictionary of model files
models = {
    "All": "bmfcnet_all.h5",
    "Eyes Open": "bmfcnet_dataOpen.h5",
    "Eyes Closed": "bmfcnet_dataClose.h5"
}

# Prepare ROC plotting

metrics_df = pd.DataFrame()

for state, model_file in models.items():
    print(f"\n📦 Loading model: {model_file}")
    model = load_model(model_file)

    # === Predict ===
    y_prob = model.predict(X_test)
    y_pred = (y_prob > 0.5).astype(int)

    # === Confusion Matrix ===
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Non-MDD", "MDD"],
                yticklabels=["Non-MDD", "MDD"])
    plt.title(f"Confusion Matrix - {state}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # === Classification Report ===
    report = classification_report(y_test, y_pred, output_dict=True)
    for label in ["0", "1"]:  # 0 = Non-MDD, 1 = MDD
        # List to hold all metrics
        metrics_list = []

        # Inside your loop, collect metrics like this:
        metrics_list.append({
            'State': state,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_prob)
        })

        # After the loop ends
        metrics_df = pd.DataFrame(metrics_list)
        print("\nAll State Metrics Summary:")
        print(metrics_df)

    # === ROC + AUC ===
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{state} (AUC = {roc_auc:.2f})")

# === Finalize ROC Curve ===
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Classification Report Bar Plot ===
melted = pd.melt(metrics_df, id_vars=["Model", "Class"], value_vars=["Precision", "Recall", "F1-score"])
plt.figure(figsize=(10, 6))
sns.barplot(data=melted, x="Class", y="value", hue="Model")
plt.title("Classification Metrics per Class")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.legend(title="Model")
plt.tight_layout()
plt.show()
