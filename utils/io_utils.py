import csv
import os


FIELDNAMES = [
    "index",
    "target_sentence", "context_a", "context_b",
    "harmful_category", "context_type", "emoji_category", "tactic",
    "explanation_a", "explanation_b",
    "raw_a", "raw_b",
    "pred_a", "pred_b",
    "gold_a", "gold_b",
    "correct_a", "correct_b",
]


class CsvStore:
    """管理 predictions CSV：列定义、断点续跑、增量 flush。

    用法：
        store = CsvStore("results/predictions_sa.csv")
        with store:
            store.write(row)
    """

    def __init__(self, path: str):
        self.path = path
        self._file = None
        self._writer = None
        self.processed: set[int] = self._load_processed()

    # ---------- 断点续跑 ----------
    def _load_processed(self) -> set[int]:
        if not os.path.exists(self.path):
            return set()
        with open(self.path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            indices: set[int] = set()
            for row in reader:
                try:
                    indices.add(int(row["index"]))
                except (KeyError, ValueError):
                    continue
            return indices

    # ---------- 上下文管理 ----------
    def __enter__(self) -> "CsvStore":
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        file_exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
        mode = "a" if file_exists else "w"
        self._file = open(self.path, mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        if not file_exists:
            self._writer.writeheader()
            self._file.flush()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()

    # ---------- 写入 ----------
    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self._file.flush()