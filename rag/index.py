import json
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import faiss
import numpy as np


from flashrag.retriever.encoder import STEncoder


corpus_path = "./emoji_corpus.jsonl"
index_dir = "./emoji_index_db"


def build_index():
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)

    print(f"[INFO] Loading STEncoder...")


    encoder = STEncoder(
        model_name="e5-base-v2",
        model_path="intfloat/e5-base-v2",
        max_length=512,
        use_fp16=False,
        instruction=""
    )

    # 3. 读取语料
    print("[INFO] Reading corpus...")
    contents = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            contents.append(json.loads(line)['contents'])

    # 4. 向量化 (Encoding)
    print(f"[INFO] Encoding {len(contents)} documents...")
    embeddings = encoder.encode(contents)

    # 5. 构建并保存 FAISS 索引
    print("[INFO] Building FAISS index...")
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss_path = os.path.join(index_dir, "index.faiss")
    faiss.write_index(index, faiss_path)

    print(f"[SUCCESS] Index built and saved at: {faiss_path}")


if __name__ == "__main__":
    build_index()