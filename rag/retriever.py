import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from flashrag.config import Config
from flashrag.retriever import DenseRetriever


class EmojiRetriever:
    def __init__(self, model_path, index_path, corpus_path):
        config_dict = {
            "retrieval_method": "e5",
            "model_path": model_path,
            "index_path": index_path,
            "corpus_path": corpus_path,
            "pooling_method": "mean",
            "max_length": 512,
            "instruction": "query: ",
            "topk": 2,
            "device": "cpu"
        }
        self.retriever = DenseRetriever(Config(config_dict=config_dict))

    def get_knowledge(self, query):
        docs = self.retriever.search(query)
        return "\n".join([f"- {d['contents']}" for d in docs])


if __name__ == '__main__':
    r = EmojiRetriever('intfloat/e5-base-v2', 'emoji_index_db/index.faiss', 'emoji_corpus.jsonl')
    k = r.get_knowledge("heart")
    print(k)
