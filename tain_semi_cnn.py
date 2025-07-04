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
VOCAB_PATH     = "new-data/merged_title_vocab.json"
EMBEDDING_PATH = "new-data/title_dsg.npy"
LABELED_PATH   = "new-data/labeled_train_processed.json"
VAL_PATH       = "new-data/val_labeled_processed.json"
UNLABELED_PATH      = "new-data/unlabeled_processed.json"
SEQ_LENGTH     = 23
BATCH_SIZE     = 100
EPOCHS         = 100
LEARNING_RATE  = 1e-4
BAG_SIZE       = 100
NUM_BAGS       = 200
BETA           = 1.0
PATIENCE       = 10
os.makedirs("checkpoints", exist_ok=True)

# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ===================== 数据加载函数 =====================
# 加载有标签数据
def load_labeled_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    inputs = [item["Title"] for item in data]
    labels = [item["label"] for item in data]
    return inputs, labels


def load_unlabeled_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    inputs = [item["Title"] for item in data]
    return inputs

def entropy_minimization_loss(y_pred):
    # 防止 log(0)
    eps = 1e-6
    entropy = -y_pred * torch.log(y_pred + eps) - (1 - y_pred) * torch.log(1 - y_pred + eps)
    return torch.mean(entropy)

# 创建PyTorch数据加载器
def make_dataloader(inputs, labels, batch_size, shuffle=True):
    inputs_tensor = torch.tensor(inputs, dtype=torch.long).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
    dataset = TensorDataset(inputs_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train():
    embedding_matrix = np.load(EMBEDDING_PATH)
    vocab_size, embedding_dim = embedding_matrix.shape

    train_inputs, train_labels = load_labeled_data(LABELED_PATH)
    val_inputs, val_labels = load_labeled_data(VAL_PATH)
    train_unlabeled_inputs = load_unlabeled_data(UNLABELED_PATH)

    # 转换为PyTorch数据加载器
    val_loader = make_dataloader(val_inputs, val_labels, batch_size=BATCH_SIZE, shuffle=False)

    # 检测器
    detector = FakeNewsDetector(vocab_size, embedding_dim, embedding_matrix).to(device)
    detector_optimizer = optim.Adam(detector.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # 5.5 检测器训练
    sup_loader = make_dataloader(train_inputs, train_labels, BATCH_SIZE)
    unsup_loader = make_dataloader(train_unlabeled_inputs, [0]*len(train_unlabeled_inputs), BATCH_SIZE) 
        
    d_best_f1 = 0
    d_patience_counter = 0
    for epoch in range(EPOCHS):
        detector.train()
        sup_iter = cycle(sup_loader)  # 无限循环 labeled 数据

        for x_unsup, _ in unsup_loader:  
            x_sup, y_sup = next(sup_iter)
            
            y_sup = y_sup.view(-1, 1)
            
            detector_optimizer.zero_grad()
            
            # 有标签损失
            y_pred_sup = detector(x_sup)
            loss_sup = criterion(y_pred_sup, y_sup)
            
        
            # 无标签损失（熵最小化）
            y_pred_unsup = detector(x_unsup)
            loss_unsup = entropy_minimization_loss(y_pred_unsup)
            
            # 总损失（权重比 1:0.1）
            total_loss = loss_sup + 0.1 * loss_unsup
            total_loss.backward()
            detector_optimizer.step()
        
        detector.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = detector(inputs)
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy())
        
        d_preds_prob = np.array(all_preds)
        d_preds_bin = (d_preds_prob >= 0.55).astype(int)
        d_y_true = np.array(all_labels)
        val_f1=f1_score(d_y_true, d_preds_bin, pos_label=1)

        if val_f1 > d_best_f1:
            d_best_f1 = val_f1
            d_patience_counter = 0
        else:
            d_patience_counter += 1
            if d_patience_counter >= PATIENCE:
                print(f"[Early Stop] detector 第 {epoch+1} 轮停止，best_f1={d_best_f1:.4f}")
                break

    # 5.6 检测器结果保存
    torch.save(detector.state_dict(), f"checkpoints/semi_cnn_detector.pt")
    
if __name__ == "__main__":
    train()