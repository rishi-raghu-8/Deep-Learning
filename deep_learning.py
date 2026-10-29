"""
Week 5: Deep Learning Application in Data Science
A CNN for handwritten digit classification using scikit-learn's public Digits dataset.
Dataset images are 8x8 grayscale; pixel intensities are normalized to [0, 1].
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(1)
OUT = Path("outputs")
PLOTS = OUT / "visualizations"
PLOTS.mkdir(parents=True, exist_ok=True)

digits = load_digits()
X = digits.images.astype(np.float32) / 16.0
y = digits.target.astype(np.int64)

# Stratified train/validation/test split: 70% / 15% / 15%.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

def image_tensor(arr):
    return torch.tensor(arr[:, None, :, :], dtype=torch.float32)

train_ds = TensorDataset(image_tensor(X_train), torch.tensor(y_train, dtype=torch.long))
val_ds = TensorDataset(image_tensor(X_val), torch.tensor(y_val, dtype=torch.long))
test_ds = TensorDataset(image_tensor(X_test), torch.tensor(y_test, dtype=torch.long))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64), nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = DigitCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
epochs, patience = 25, 6
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_loss, best_state, wait = float("inf"), None, 0

def evaluate(loader):
    model.eval()
    loss_sum, actual, predicted = 0.0, [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss_sum += criterion(logits, yb).item() * len(yb)
            actual.extend(yb.numpy())
            predicted.extend(logits.argmax(1).numpy())
    return loss_sum / len(loader.dataset), accuracy_score(actual, predicted), np.array(actual), np.array(predicted)

for epoch in range(epochs):
    model.train()
    loss_sum, actual, predicted = 0.0, [], []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(yb)
        actual.extend(yb.numpy())
        predicted.extend(logits.argmax(1).detach().numpy())
    train_loss = loss_sum / len(train_loader.dataset)
    train_acc = accuracy_score(actual, predicted)
    val_loss, val_acc, _, _ = evaluate(val_loader)
    for key, value in [("train_loss", train_loss), ("val_loss", val_loss),
                       ("train_acc", train_acc), ("val_acc", val_acc)]:
        history[key].append(value)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            break

model.load_state_dict(best_state)
test_loss, test_acc, y_true, y_pred = evaluate(test_loader)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)
cm = confusion_matrix(y_true, y_pred, labels=np.arange(10))

print("Dataset samples:", len(X))
print("Train/validation/test:", len(X_train), len(X_val), len(X_test))
print("Epochs completed:", len(history["train_loss"]))
print(f"Test accuracy: {test_acc:.4f}")
print(f"Macro precision: {precision:.4f}")
print(f"Macro recall: {recall:.4f}")
print(f"Macro F1-score: {f1:.4f}")

# Save plots and results
fig, ax = plt.subplots()
ax.plot(history["train_loss"], label="Training loss")
ax.plot(history["val_loss"], label="Validation loss")
ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training and Validation Loss")
ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "01_training_loss.png", dpi=180); plt.close(fig)

fig, ax = plt.subplots()
ax.plot(history["train_acc"], label="Training accuracy")
ax.plot(history["val_acc"], label="Validation accuracy")
ax.set(xlabel="Epoch", ylabel="Accuracy", title="Training and Validation Accuracy")
ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "02_training_accuracy.png", dpi=180); plt.close(fig)

fig, ax = plt.subplots()
im = ax.imshow(cm); fig.colorbar(im, ax=ax)
ax.set(title="Confusion Matrix — Test Set", xlabel="Predicted digit", ylabel="True digit")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
for i in range(10):
    for j in range(10):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
fig.tight_layout(); fig.savefig(PLOTS / "03_confusion_matrix.png", dpi=180); plt.close(fig)

pd.DataFrame([{
    "test_accuracy": test_acc, "macro_precision": precision, "macro_recall": recall,
    "macro_f1": f1, "test_loss": test_loss, "epochs_completed": len(history["train_loss"])
}]).to_csv(OUT / "evaluation_metrics.csv", index=False)
torch.save(model.state_dict(), OUT / "digit_cnn_model.pth")
print("Outputs saved to:", OUT.resolve())
