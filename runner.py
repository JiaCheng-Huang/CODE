import time
from tqdm import tqdm

from method.base import BaseMode
from utils.io_utils import CsvStore
from idle_checker import (
    is_idle_time,
    seconds_until_idle,
    status_text,
    STATUS_REFRESH_INTERVAL,
)


class Runner:
    """把「数据集 × 模式 × CSV 存储」粘起来。

    可选：check_idle=True 时，每次处理记录前检查 DeepSeek 空闲时段，
    非空闲则阻塞等待，直到进入空闲时段再继续。
    """

    def __init__(
        self,
        client,
        mode: BaseMode,
        store: CsvStore,
        n_runs: int = 1,
        check_idle: bool = False,
    ):
        self.client = client
        self.mode = mode
        self.store = store
        self.n_runs = n_runs
        self.check_idle = check_idle

    # ------------------------------------------------------------------
    def run(self, records: list[dict]) -> None:
        total = len(records)
        already = len(self.store.processed)
        remaining = total - already
        print(f"Total records: {total}, already processed: {already}, remaining: {remaining}")
        if remaining <= 0:
            print("All records have been processed. Nothing to do.")
            return

        # 启动时打印一次当前状态
        if self.check_idle:
            print(f"deepseek空闲检查已开启")

        pbar = tqdm(
            enumerate(records),
            total=total,
            desc=f"Processing[{self.mode.name}]",
            dynamic_ncols=True,   # 让 postfix 随终端宽度自适应
        )
        try:
            for idx, record in pbar:
                if idx in self.store.processed:
                    continue

                # ---------- 每次请求前检查空闲 ----------
                if self.check_idle:
                    self._wait_until_idle(pbar)
                    pbar.set_postfix_str(status_text())

                self._process_one(idx, record)

                # 处理完再刷一次状态，保证 postfix 始终是最新
                if self.check_idle:
                    pbar.set_postfix_str(status_text())
        finally:
            pbar.close()

    # ------------------------------------------------------------------
    def _wait_until_idle(self, pbar) -> None:
        """阻塞直到进入空闲时段，期间持续刷新控制台状态。"""
        if is_idle_time():
            return
        while not is_idle_time():
            wait_s = seconds_until_idle()
            pbar.set_postfix_str(status_text())
            pbar.refresh()
            # 分片 sleep，每 STATUS_REFRESH_INTERVAL 秒刷新一次状态
            time.sleep(min(wait_s, STATUS_REFRESH_INTERVAL))

    # ------------------------------------------------------------------
    def _call_mode(self, target_sentence: str, context: str) -> tuple[str, str]:
        if self.mode.multi_run and self.n_runs > 1:
            return self.mode.run_avg(
                self.client, target_sentence, context, n_runs=self.n_runs
            )
        return self.mode.run(self.client, target_sentence, context)

    def _process_one(self, idx: int, record: dict) -> None:
        target_sentence = record.get("target_sentence", "")
        context_a = record.get("context_a", "")
        context_b = record.get("context_b", "")

        raw_a, pred_a = self._call_mode(target_sentence, context_a)
        raw_b, pred_b = self._call_mode(target_sentence, context_b)

        gold_a, gold_b = "benign", "toxic"
        self.store.write({
            "index": idx,
            "target_sentence": target_sentence,
            "context_a": context_a,
            "context_b": context_b,
            "harmful_category": record.get("harmful_category", ""),
            "context_type": record.get("context_type", ""),
            "emoji_category": record.get("emoji_category", ""),
            "tactic": record.get("tactic", ""),
            "explanation_a": record.get("explanation_a", ""),
            "explanation_b": record.get("explanation_b", ""),
            "raw_a": raw_a,
            "raw_b": raw_b,
            "pred_a": pred_a,
            "pred_b": pred_b,
            "gold_a": gold_a,
            "gold_b": gold_b,
            "correct_a": pred_a == gold_a,
            "correct_b": pred_b == gold_b,
        })