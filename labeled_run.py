# 完全等价的 PyTorch 实现 AnnotatorWrapper
import json
import jieba
import numpy as np
import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from aggregation_cell import AggregationCell
from annotator import Annotator
from text_cnn_encoder import TextCNNEncoder
import json
import random

VOCAB_PATH = "new-data/vocab.json"
EMBEDDING_PATH = "new-data/dsg_embedding.npy"
WEIGHTS_PATH = "saved_models/best_annotator.pt"
UNLABELED_CSV = "data-all/data/unlabeled data/news.csv"
SAVE_DATA_PATH = "data2/weak_labeled_data_with_title.json"

class AnnotatorWrapper:
    """标注器封装：仅对未标注数据生成弱标签和概率，保留Title等字段供后续使用"""
    def __init__(self, vocab_path, embedding_path, weights_path, seq_length=58):
        self.seq_length = seq_length
        self.vocab = self._load_vocab(vocab_path)
        self.embedding_matrix = self._load_embedding(embedding_path)
        self.model = self._build_model()
        self._load_weights(weights_path)
        self.keep_fields = ['Title', 'Official Account Name', 'News Url', 'Image Url']

    def _load_vocab(self, path):
        """加载词表"""
        with open(path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        print(f"[INFO] 词表加载完成，大小: {len(vocab)}")
        return vocab

    def _load_embedding(self, path):
        """加载嵌入矩阵"""
        embedding_matrix = np.load(path)
        print(f"[INFO] 嵌入矩阵加载完成，形状: {embedding_matrix.shape}")
        return embedding_matrix

    def _build_model(self):
        """构建标注器模型"""
        vocab_size, embedding_dim = self.embedding_matrix.shape
        text_cnn = TextCNNEncoder(vocab_size, embedding_dim, self.embedding_matrix)
        agg_cell = AggregationCell(output_dim=20)
        model = Annotator(text_cnn, agg_cell)
        model.eval()
        return model

    def _load_weights(self, path):
        """加载模型权重"""
        try:
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            print(f"[INFO] 模型权重加载完成: {path}")
        except Exception as e:
            raise ValueError(f"模型权重加载失败: {str(e)}")

    def _tokens_to_ids(self, text):
        """文本转ID序列（处理空文本）"""
        if not text or pd.isna(text):
            text = ""
        words = list(jieba.cut(str(text).strip()))
        unk_id = self.vocab.get("<UNK>", 1)
        pad_id = self.vocab.get("<PAD>", 0)
        ids = [self.vocab.get(w, unk_id) for w in words] if words else []
        if len(ids) > self.seq_length:
            ids = ids[:self.seq_length]
        else:
            ids += [pad_id] * (self.seq_length - len(ids))
        return ids

    def process_unlabeled_data(self, csv_path, threshold=0.55):
        """处理未标注数据，生成弱标签和概率，保留Title等字段"""
        df = pd.read_csv(csv_path)
        for field in self.keep_fields + ['Report Content']:
            if field not in df.columns:
                df[field] = ""
                print(f"[WARN] 缺失字段 '{field}'，已补充为空值")

        weak_labeled_data = []
        skipped_indices = []

        for idx, row in df.iterrows():
            report_content = str(row['Report Content']).strip() if not pd.isna(row['Report Content']) else ""
            reports = [r.strip() for r in report_content.split('##') if r.strip()]
            if not reports:
                skipped_indices.append(idx)
                continue
            encoded_reports = [self._tokens_to_ids(r) for r in reports]
            if not encoded_reports:
                skipped_indices.append(idx)
                continue
            item = {field: str(row[field]).strip() if not pd.isna(row[field]) else "" for field in self.keep_fields}
            weak_labeled_data.append({**item, "encoded_reports": encoded_reports})

        print(f"[INFO] 原始数据 {len(df)} 条，有效样本 {len(weak_labeled_data)} 条，跳过 {len(skipped_indices)} 条")
        if not weak_labeled_data:
            raise ValueError("无有效未标注样本可处理")

        # 构建批量输入
        inputs = [item["encoded_reports"] for item in weak_labeled_data]
        lens = [len(x) for x in inputs]
        flat_inputs = [r for x in inputs for r in x]
        input_tensor = torch.tensor(flat_inputs, dtype=torch.long)
        lens_tensor = torch.tensor(lens, dtype=torch.long)
        with torch.no_grad():
            probs_tensor = self.model(input_tensor, lens_tensor)  # (batch, 1)
        pred_probs = probs_tensor.squeeze().numpy()
        weak_labels = (pred_probs >= threshold).astype(int)

        for i in range(len(weak_labeled_data)):
            weak_labeled_data[i]["weak_label"] = int(weak_labels[i])
            weak_labeled_data[i]["pred_prob"] = float(pred_probs[i])
            del weak_labeled_data[i]["encoded_reports"]

        print(f"[INFO] 弱标注完成，跳过样本索引: {skipped_indices[:5]}...")
        return weak_labeled_data, pred_probs

    def save_weak_results(self, weak_labeled_data, save_data_path):
        """保存弱标注结果（含Title、weak_label和pred_prob）"""
        dir_path = os.path.dirname(save_data_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(save_data_path, 'w', encoding='utf-8') as f:
            json.dump(weak_labeled_data, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 弱标注数据（含Title、weak_label、pred_prob）已保存至: {save_data_path}")

def split():
    # 加载弱标注数据
    with open("data2/weak_labeled_data_with_title.json", 'r', encoding='utf-8') as f:
        weak_data = json.load(f)

    # 划分两类数据
    weak_fake = [x for x in weak_data if x["weak_label"] == 1]  # weakly fake data
    weak_real = [x for x in weak_data if x["weak_label"] == 0]  # weakly real data

    random.seed(42)
    sample_size_fake = int(0.1 * len(weak_fake))
    sample_size_real = int(0.1 * len(weak_real))

    val_fake = random.sample(weak_fake, sample_size_fake)  # 10% weak fake
    val_real = random.sample(weak_real, sample_size_real)  # 10% weak real
    new_val_set = val_fake + val_real  # 新验证集

    # 剩余数据用于训练
    train_weak_fake = [x for x in weak_fake if x not in val_fake]
    train_weak_real = [x for x in weak_real if x not in val_real]
    train_weak = train_weak_fake + train_weak_real  # 剩余90%弱标注数据

    # 保存划分结果
    with open("data2/new_val_set.json", 'w', encoding='utf-8') as f:
        json.dump(new_val_set, f, ensure_ascii=False, indent=2)
    with open("data2/train_weak_data.json", 'w', encoding='utf-8') as f:
        json.dump(train_weak, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 新验证集构建完成：共{len(new_val_set)}条（fake {sample_size_fake}条，real {sample_size_real}条）")
    print(f"[INFO] 剩余弱标注训练数据：{len(train_weak)}条")

def labeled():
    annotator = AnnotatorWrapper(
        vocab_path=VOCAB_PATH,
        embedding_path=EMBEDDING_PATH,
        weights_path=WEIGHTS_PATH,
        seq_length=58
    )
    # 处理未标注数据
    weak_data, probs = annotator.process_unlabeled_data(
        csv_path=UNLABELED_CSV,
        threshold=0.55
    )
    # 保存结果
    annotator.save_weak_results(
        weak_labeled_data=weak_data,
        save_data_path=SAVE_DATA_PATH,
    )
    print("[INFO] 未标注数据弱标注完成")

if __name__ == "__main__":
    # labeled()
    split()