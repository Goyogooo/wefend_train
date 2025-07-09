import json
import numpy as np
from detector import FakeNewsDetector
import jieba
import pandas as pd
from collections import defaultdict
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from text_cnn_encoder import TextCNNEncoder


# [INFO] 基于Title确定max_seq_len: 23
EMBEDDING_PATH = "data2/title_dsg.npy"
TEST_PROCESSED_PATH = "data2/test_labeled_processed.json"
# MODEL_PATH = "checkpoints/semi_cnn_detector.pt"
MODEL_PATH = "saved_models/wefend_detector2.pt"
# BASE_MODEL_PATH = "checkpoints/detector_round_0.pt"

SEQ_LENGTH = 23
BATCH_SIZE = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 加载有标签数据
def load_labeled_title_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    inputs = [item["Title"] for item in data]
    labels = [item["label"] for item in data]
    return inputs, labels

def make_dataloader(inputs, labels, batch_size, shuffle=False):
    """创建PyTorch数据加载器"""
    inputs_tensor = torch.tensor(inputs, dtype=torch.long).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
    dataset = TensorDataset(inputs_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def save_results_to_file(results, model_path, test_size, seq_length, save_path="test_results.txt"):
    """将评估指标保存到文件"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"模型性能评估: {model_path}\n")
        f.write("=" * 60 + "\n")
        f.write(f"测试集大小: {test_size}\n")
        f.write(f"序列长度: {seq_length}\n")
        f.write(f"准确率 (Accuracy): {results['accuracy']:.4f}\n")
        f.write(f"AUC-ROC: {results['auc_roc']:.4f}\n")
        f.write(f"F1分数: {results['f1_score']:.4f}\n")
        f.write("\n分类报告:\n")
        f.write(results['classification_report'])
    print(f"\n评估结果已保存到 {save_path}")
  
def evaluate_model(model, test_loader):
    """在测试集上评估模型性能"""
    model.eval()  
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            
            # 收集预测结果
            all_probs.extend(outputs.cpu().numpy().flatten())
            all_preds.extend((outputs >= 0.5).int().cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())
    
    # 转换为numpy数组
    probs = np.array(all_probs)
    preds = np.array(all_preds)
    true_labels = np.array(all_labels)
    
    # 计算评估指标
    accuracy = accuracy_score(true_labels, preds)
    auc_roc = roc_auc_score(true_labels, probs)
    f1 = f1_score(true_labels, preds)
    report = classification_report(true_labels, preds, digits=4)
    
    # 打印结果
    print("=" * 60)
    print(f"模型性能评估: {MODEL_PATH}")
    print("=" * 60)
    print(f"测试集大小: {len(true_labels)}")
    print(f"序列长度: {SEQ_LENGTH}")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"F1分数: {f1:.4f}")
    print("\n分类报告:")
    print(report)
    
    # 返回结果
    return {
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "f1_score": f1,
        "classification_report": report
    }


def test_wrapper(model_path):
    
    embedding_matrix = np.load(EMBEDDING_PATH)
    vocab_size, embedding_dim = embedding_matrix.shape

    model = FakeNewsDetector(vocab_size, embedding_dim, embedding_matrix).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型: {model_path}")
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {model_path}")
        return
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return
    
    test_inputs, test_labels = load_labeled_title_data(TEST_PROCESSED_PATH)
    
    test_loader = make_dataloader(test_inputs, test_labels, BATCH_SIZE)
    print(f"测试集大小: {len(test_inputs)}")

    results = evaluate_model(model, test_loader)
    
    save_results_to_file(
        results,
        model_path=model_path,
        test_size=len(test_inputs),
        seq_length=SEQ_LENGTH,
        save_path=(f"{model_path}_results2.txt")
    )

if __name__ == "__main__":
    model_path = MODEL_PATH
    test_wrapper(model_path)

# weak_cnn
# 准确率 (Accuracy): 0.8435
# AUC-ROC: 0.7484
# F1分数: 0.4585

# 分类报告:
#               precision    recall  f1-score   support

#          0.0     0.9068    0.9103    0.9085      8659
#          1.0     0.4638    0.4534    0.4585      1482

# wefend
# 准确率 (Accuracy): 0.9080
# AUC-ROC: 0.8888
# F1分数: 0.6421

# 分类报告:
#               precision    recall  f1-score   support

#          0.0     0.9285    0.9667    0.9472      8659
#          1.0     0.7440    0.5648    0.6421      1482

# 准确率 (Accuracy): 0.9019
# AUC-ROC: 0.8876
# F1分数: 0.6198

# 分类报告:
#               precision    recall  f1-score   support

#          0.0     0.9255    0.9626    0.9437      8659
#          1.0     0.7145    0.5472    0.6198      1482

# wefend——
# 准确率 (Accuracy): 0.9125
# AUC-ROC: 0.9065
# F1分数: 0.6324

# 分类报告:
#               precision    recall  f1-score   support

#          0.0     0.9219    0.9806    0.9504      8659
#          1.0     0.8195    0.5148    0.6324      1482