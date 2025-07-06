from itertools import cycle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import os
import random
from detector import FakeNewsDetector
from selector import Selector, build_selector_state
from text_cnn_encoder import TextCNNEncoder
# ====================== 配置 ======================
VOCAB_PATH     = "data2/merged_title_vocab.json"
EMBEDDING_PATH = "data2/title_dsg.npy"
VAL_PATH       = "data2/val_weak_processed.json"
WEAK_PATH      = "data2/train_weak_processed.json"
SEQ_LENGTH     = 23
BATCH_SIZE     = 100
EPOCHS         = 100
LEARNING_RATE  = 1e-4
BAG_SIZE       = 100
NUM_BAGS       = 200
BETA           = 1.0
PATIENCE       = 10
os.makedirs("checkpoints", exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载有标签数据，输出title和label
def load_labeled_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    inputs = [item["Title"] for item in data]
    labels = [item["label"] for item in data]
    return inputs, labels

# 加载弱标签数据，输出title和weak_label
def load_weak_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    inputs = [item["Title"] for item in data]
    labels = [item["weak_label"] for item in data]
    return inputs, labels


# 创建PyTorch数据加载器，分成batch
def make_dataloader(inputs, labels, batch_size, shuffle=True):
    inputs_tensor = torch.tensor(inputs, dtype=torch.long).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
    dataset = TensorDataset(inputs_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train():
    embedding_matrix = np.load(EMBEDDING_PATH)
    vocab_size, embedding_dim = embedding_matrix.shape


    val_inputs, val_labels = load_weak_data(VAL_PATH)
    train_weak_inputs, train_weak_labels = load_weak_data(WEAK_PATH)

    # 转换为PyTorch数据加载器
    val_loader = make_dataloader(val_inputs, val_labels, batch_size=BATCH_SIZE, shuffle=False)

    # 检测器初始化
    detector = FakeNewsDetector(vocab_size, embedding_dim, embedding_matrix).to(device)
    detector_optimizer = optim.Adam(detector.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # 检测器训练

    pseudo_loader = make_dataloader(train_weak_inputs, train_weak_labels, BATCH_SIZE) 
        
    d_best_f1 = 0
    d_patience_counter = 0
    for epoch in range(EPOCHS):
        detector.train()

        for x_pseudo, y_pseudo in pseudo_loader:
            y_pseudo = y_pseudo.view(-1, 1)

            detector_optimizer.zero_grad()
            y_pred_pseudo = detector(x_pseudo)
            loss_pseudo = criterion(y_pred_pseudo, y_pseudo)

            loss_pseudo.backward()
            detector_optimizer.step()
        
        detector.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = detector(inputs)
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy())
        
        d_preds_prob = np.array(all_preds)
        d_preds_bin = (d_preds_prob >= 0.5).astype(int)
        d_y_true = np.array(all_labels)

        acc = accuracy_score(d_y_true, d_preds_bin)
        auc = roc_auc_score(d_y_true, d_preds_prob)
        f1 = f1_score(d_y_true, d_preds_bin)
        print(f"========== EPOCH {epoch+1} ==========")
        print(f"Accuracy: {acc:.4f}")
        print(f"AUC-ROC : {auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Classification Report:\n", classification_report(d_y_true, d_preds_bin, digits=4))


        if f1 > d_best_f1:
            d_best_f1 = f1
            d_patience_counter = 0
            os.makedirs("saved_models", exist_ok=True)
            torch.save(detector.state_dict(), "saved_models/weak_detector2.pt")
        else:
            d_patience_counter += 1
            if d_patience_counter >= PATIENCE:
                print(f"[Early Stop] detector 第 {epoch+1} 轮停止，best_f1={d_best_f1:.4f}")
                detector.load_state_dict(torch.load("saved_models/weak_detector2.pt"))
                break

    # 5.6 检测器结果保存
    torch.save(detector.state_dict(), f"saved_models/weak_detector2.pt")
    
if __name__ == "__main__":
    train()


# Accuracy: 0.7480
# AUC-ROC : 0.6651
# F1 Score: 0.3696
# Classification Report:
#                precision    recall  f1-score   support

#          0.0     0.8021    0.8872    0.8425      5131
#          1.0     0.4629    0.3076    0.3696      1622