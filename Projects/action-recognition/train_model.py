import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

#CONFIG 
DATA_DIR      = "data/raw"
MODEL_DIR     = "models"
SEQUENCE_LEN  = 30
INPUT_SIZE    = 132        
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
DROPOUT       = 0.3
EPOCHS        = 60
BATCH_SIZE    = 32
LR            = 1e-3
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"



# MODEL
class ActionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)                       # (B, T, 2H)
        attn_w = torch.softmax(self.attention(out), dim=1)  # (B, T, 1)
        context = (attn_w * out).sum(dim=1)         # (B, 2H)
        return self.classifier(context)



def load_dataset(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"\n[ERROR] Data directory '{data_dir}' not found.\n"
            "  Please collect data first by running:\n"
            "    python data_collector.py --action walking  --samples 200\n"
            "    python data_collector.py --action jumping  --samples 200\n"
            "    python data_collector.py --action sitting  --samples 200\n"
            "    python data_collector.py --action standing --samples 200\n"
            "    python data_collector.py --action waving   --samples 200\n"
        )
    X, y = [], []
    actions = sorted(os.listdir(data_dir))
    label_map = {i: a for i, a in enumerate(actions)}

    for idx, action in enumerate(actions):
        action_dir = os.path.join(data_dir, action)
        files = [f for f in os.listdir(action_dir) if f.endswith(".npy")]
        print(f"  {action:15s}  {len(files)} samples")
        for f in files:
            seq = np.load(os.path.join(action_dir, f))
            X.append(seq)
            y.append(idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), label_map


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"\n[INFO] Loading dataset from '{DATA_DIR}' ...")
    X, y, label_map = load_dataset(DATA_DIR)
    num_classes = len(label_map)
    print(f"\n[INFO] Dataset: {X.shape[0]} samples, {num_classes} classes  |  Device: {DEVICE}")

    # Save label map
    np.save(os.path.join(MODEL_DIR, "label_map.npy"), label_map)
    with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    # Split
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    to_tensor = lambda arr: torch.tensor(arr)
    train_loader = DataLoader(TensorDataset(to_tensor(X_tr), to_tensor(y_tr)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(to_tensor(X_val), to_tensor(y_val)),
                              batch_size=BATCH_SIZE)

    # Model
    model = ActionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes, DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n[INFO] Training for {EPOCHS} epochs ...\n")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss, tr_correct = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(xb)
            tr_correct += (logits.argmax(1) == yb).sum().item()
        scheduler.step()

        model.eval()
        val_loss, val_correct = 0.0, 0
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                preds = logits.argmax(1)
                val_correct += (preds == yb).sum().item()
                all_preds.extend(preds.cpu().tolist())
                all_true.extend(yb.cpu().tolist())

        t_acc = tr_correct / len(X_tr)
        v_acc = val_correct / len(X_val)
        history["train_loss"].append(tr_loss / len(X_tr))
        history["val_loss"].append(val_loss / len(X_val))
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "action_lstm_best.pth"))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train Loss: {history['train_loss'][-1]:.4f}  Acc: {t_acc:.3f} | "
                  f"Val Loss: {history['val_loss'][-1]:.4f}  Acc: {v_acc:.3f}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "action_lstm.pth"))
    print(f"\n[DONE] Best val accuracy: {best_val_acc:.3f}")
    print(f"[INFO] Model saved to '{MODEL_DIR}/'")

    # Classification report
    print("\n── Classification Report")
    labels = [label_map[i] for i in range(num_classes)]
    print(classification_report(all_true, all_preds, target_names=labels))

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Action Recognition — Training Report", fontsize=14)

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    cm = confusion_matrix(all_true, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels,
                ax=axes[2], cmap="Blues")
    axes[2].set_title("Confusion Matrix (Val)")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True")

    plt.tight_layout()
    report_path = os.path.join(MODEL_DIR, "training_report.png")
    plt.savefig(report_path, dpi=120)
    print(f"[INFO] Training report saved to '{report_path}'")


if __name__ == "__main__":
    train()
