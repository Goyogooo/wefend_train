import json
import jieba
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from aggregation_cell import AggregationCell
from annotator import Annotator
from text_cnn_encoder import TextCNNEncoder

# ====================== 参数配置（仅保留测试所需） ======================
VOCAB_PATH     = "new-data/vocab.json"
EMBEDDING_PATH = "new-data/dsg_embedding.npy"
TEST_JSON_PATH = "new-data/test_grouped.json"
WEIGHTS_PATH   = "saved_models/best_annotator2.pt"
SEQ_LENGTH     = 58
BATCH_SIZE     = 100
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================== 测试所需工具函数 ========================
def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_grouped_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    inputs = [item["reports"] for item in data]
    labels = [item["label"] for item in data]
    return inputs, labels

class NewsDataset(Dataset):
    def __init__(self, grouped_inputs, labels):
        self.flat_inputs = [report for reports in grouped_inputs for report in reports]
        self.lengths = [len(reports) for reports in grouped_inputs]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        start = sum(self.lengths[:idx])
        end = start + self.lengths[idx]
        reports = self.flat_inputs[start:end]
        reports = [r + [0] * (SEQ_LENGTH - len(r)) if len(r) < SEQ_LENGTH else r[:SEQ_LENGTH] for r in reports]
        return torch.tensor(reports, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float), self.lengths[idx]

def collate_fn(batch):
    flat_reports = []
    labels = []
    lengths = []
    for reports, label, length in batch:
        flat_reports.extend(reports)
        labels.append(label)
        lengths.append(length)
    return torch.stack(flat_reports), torch.tensor(labels), torch.tensor(lengths)

# ==================== 测试评估流程 =======================
def evaluate():

    # 构建模型
    embedding_matrix = np.load(EMBEDDING_PATH)
    vocab_size, embedding_dim = embedding_matrix.shape
    text_cnn = TextCNNEncoder(
        vocab_size, embedding_dim, embedding_matrix,
        filter_sizes=[1,2,3,4,5,6], num_filters=40, feature_dim=40
    )
    agg_cell = AggregationCell(output_dim=20)
    model = Annotator(text_cnn, agg_cell)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"[INFO] 成功加载模型: {WEIGHTS_PATH}")

    test_inputs, test_labels = load_grouped_json(TEST_JSON_PATH)
    test_set = NewsDataset(test_inputs, test_labels)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    all_probs = []
    all_preds = []
    y_true = []
    with torch.no_grad():
        for x, y_batch, lens in test_loader:
            y_true.extend(y_batch.cpu().numpy()) 
            x = x.to(DEVICE)
            lens = lens.to(DEVICE)
            preds = model(x, lens).squeeze()
            all_probs.extend(preds.cpu().numpy())
            all_preds.extend((preds >= 0.5).int().cpu().numpy())

    final_preds = np.array(all_preds)
    preds_prob = np.array(all_probs)

    print("Accuracy:", accuracy_score(y_true, final_preds))
    print("AUC-ROC: ", roc_auc_score(y_true, preds_prob))
    print(classification_report(y_true, final_preds, digits=4))

# =================== 执行入口 ========================
if __name__ == "__main__":
    evaluate()


# Accuracy: 0.8034390750074119
# AUC-ROC:  0.7834371024536216
#               precision    recall  f1-score   support

#          0.0     0.9125    0.8515    0.8809      8642
#          1.0     0.3754    0.5220    0.4367      1477



# Accuracy: 0.8099614586421583
# AUC-ROC:  0.7844934525644076
#               precision    recall  f1-score   support

#          0.0     0.9121    0.8604    0.8855      8642
#          1.0     0.3866    0.5146    0.4415      1477


# Accuracy: 0.8088743947030339
# AUC-ROC:  0.8049475589369484
#               precision    recall  f1-score   support

#          0.0     0.9140    0.8569    0.8845      8642
#          1.0     0.3867    0.5281    0.4465      1477