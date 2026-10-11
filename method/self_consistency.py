from collections import Counter

from method.base import BaseMode
from prompts import cot_evaluate
from parsers import extract_final_answer

DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_K = None
DEFAULT_NUM_PATHS = 5
DEFAULT_N_RUNS = 10

_VALID_LABELS = ("toxic", "benign")


class SelfConsistencyMode(BaseMode):
    name = "self_consistency"
    multi_run = True  # 支持 §3.2 的 10-run 评估协议

    def __init__(
            self,
            num_paths: int = DEFAULT_NUM_PATHS,
            temperature: float = DEFAULT_TEMPERATURE,
            top_k: int | None = DEFAULT_TOP_K,
    ):
        self.num_paths = num_paths
        self.temperature = temperature
        self.top_k = top_k

    # ------------------------------------------------------------------
    # 单次运行（论文 §3.2 "in each run"）
    # ------------------------------------------------------------------
    def run(self, client, target_sentence, context):
        prompt = cot_evaluate(target_sentence, context)

        samples: list[tuple[str, str]] = []
        for _ in range(self.num_paths):
            # 论文要求「independently sample from the decoder」：
            #   - 不传 seed，让解码器真正独立采样
            #   - temperature > 0 打开多项式采样
            raw = client.generate(
                prompt,
                temperature=self.temperature,
                top_k=self.top_k,
            )
            samples.append((raw, extract_final_answer(raw)))

        final_answer = self._majority_vote([a for _, a in samples])
        return self._merge_logs(samples, final_answer), final_answer

    # ------------------------------------------------------------------
    # §3.2 的评估协议：10 次运行，取平均
    #   - 每次运行独立采样 40 条路径 → 多数投票 → 该运行的答案
    #   - 10 次运行的答案再取多数投票，作为该样本的最终预测
    #   - 同时返回每次运行的预测，便于计算论文 Figure 2 的 mean/std
    # ------------------------------------------------------------------
    def run_avg(self, client, target_sentence, context, n_runs: int = DEFAULT_N_RUNS):
        per_run: list[str] = []
        logs: list[str] = []
        for r in range(n_runs):
            raw, ans = self.run(client, target_sentence, context)
            per_run.append(ans)
            logs.append(f"===== Run {r + 1} | prediction: {ans or 'N/A'} =====\n{raw}")

        final_answer = self._majority_vote(per_run)
        merged = (
                "\n\n".join(logs)
                + f"\n\n=== Per-run predictions (n={n_runs}) ===\n{per_run}"
                + f"\n=== Majority over runs ===\n{final_answer}"
        )
        return merged, final_answer

    # ------------------------------------------------------------------
    # 论文 §2：argmax_a Σ_i 1(a_i == a)
    # ------------------------------------------------------------------
    @staticmethod
    def _majority_vote(answers: list[str]) -> str:
        valid = [a for a in answers if a in _VALID_LABELS]
        if not valid:
            return ""
        counter = Counter(valid)
        top = counter.most_common()
        # 二分类平票时保守选 toxic（该决策独立于论文，可改为 benign 或 "")
        if len(top) > 1 and top[0][1] == top[1][1]:
            return "toxic"
        return top[0][0]

    # ------------------------------------------------------------------
    # 把全部推理路径写入 raw_a / raw_b，便于事后复核
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_logs(samples: list[tuple[str, str]], final_answer: str) -> str:
        blocks = [
            f"--- Path {i} (extracted: {ans or 'N/A'}) ---\n{raw}"
            for i, (raw, ans) in enumerate(samples, 1)
        ]
        blocks.append(f"=== Majority Vote (over {len(samples)} paths) ===\n{final_answer}")
        return "\n\n".join(blocks)
