# Q
import json
import numpy as np

def convert_vec_to_npy():
    vec_path = "new-data/title_dsg.vec"
    vocab_path = "new-data/merged_title_vocab.json"
    npy_out_path = "new-data/title_dsg.npy"

    # 载入vocab
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    # 载入.vec
    vec_map = {}
    with open(vec_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        if len(header.strip().split()) != 2:
            f.seek(0)
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            vec_map[word] = vec

    # 构造矩阵
    emb_dim = len(next(iter(vec_map.values())))
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, emb_dim), dtype=np.float32)

    missing = 0
    for word, idx in vocab.items():
        if word in vec_map:
            embedding_matrix[idx] = vec_map[word]
        else:
            # 随机初始化
            embedding_matrix[idx] = np.random.uniform(-0.05, 0.05, emb_dim)
            missing += 1

    print(f"[INFO] Embedding shape = {embedding_matrix.shape}")
    print(f"[INFO] Randomly initialized missing words: {missing}")

    # 4保存
    np.save(npy_out_path, embedding_matrix)
    print(f"[INFO] Saved to {npy_out_path}")

if __name__ == "__main__":
    convert_vec_to_npy()
