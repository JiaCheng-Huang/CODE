from method.base import BaseMode
from prompts import (
    mad_debate_topic,
    mad_meta_prompt,
    mad_affirmative_prompt,
    mad_negative_prompt,
    mad_judge_discriminative_prompt,
    mad_judge_extractive_prompt,
)
from parsers import (
    parse_debater_answer,
    parse_judge_discriminative,
    parse_judge_extractive,
)


# 论文 §3.2: 采样温度设置为 0（greedy decode）
MAD_TEMPERATURE = 0.0

# 论文 §4.3: 最大辩论迭代轮数 = 3
MAD_MAX_ITERATIONS = 3


class MadMode(BaseMode):
    """Multi-Agent Debate (Liang et al., EMNLP 2024)。

    严格按论文 §2 的三组件实现：
      - Meta Prompts（含 level-2 'tit for tat'）
      - 2 个 Debater（affirmative + negative），固定顺序发言
      - 1 个 Judge，含 Discriminative / Extractive 两种模式

    流程（与论文 Figure 2 一致）：
        iter = 1 .. MAX_ITERATIONS
            Affirmative(H)  -> h_a
            H += h_a
            Negative(H, h_a) -> h_n
            H += h_n
            Judge_d(H) -> (obtained?, answer?)
            H += judge_output
            if obtained: return answer
        如果循环结束仍未 obtained：
            Judge_e(H) -> a
            return a

    采样温度恒为 0，与论文 §3.2 一致。
    """

    name = "mad"
    multi_run = False

    def __init__(self, max_iterations: int = MAD_MAX_ITERATIONS):
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------
    def run(self, client, target_sentence, context):
        debate_topic = mad_debate_topic(target_sentence, context)
        meta_prompt = mad_meta_prompt(debate_topic)

        history = ""
        log_blocks: list[str] = []

        for iteration in range(1, self.max_iterations + 1):
            # ---------- Affirmative speaks ----------
            aff_prompt = mad_affirmative_prompt(meta_prompt, history)
            aff_raw = client.generate(aff_prompt, temperature=MAD_TEMPERATURE)
            aff_answer = parse_debater_answer(aff_raw)

            history += f"\n=== Iteration {iteration} | Affirmative ===\n{aff_raw}\n"
            log_blocks.append(
                f"=== Iteration {iteration} | Affirmative "
                f"(extracted: {aff_answer or 'N/A'}) ===\n{aff_raw}"
            )

            # ---------- Negative speaks ----------
            neg_prompt = mad_negative_prompt(meta_prompt, history, aff_raw)
            neg_raw = client.generate(neg_prompt, temperature=MAD_TEMPERATURE)
            neg_answer = parse_debater_answer(neg_raw)

            history += f"\n=== Iteration {iteration} | Negative ===\n{neg_raw}\n"
            log_blocks.append(
                f"=== Iteration {iteration} | Negative "
                f"(extracted: {neg_answer or 'N/A'}) ===\n{neg_raw}"
            )

            # ---------- Judge – Discriminative ----------
            judge_prompt = mad_judge_discriminative_prompt(meta_prompt, debate_topic, history)
            judge_raw = client.generate(judge_prompt, temperature=MAD_TEMPERATURE)
            obtained, answer = parse_judge_discriminative(judge_raw)

            history += f"\n=== Iteration {iteration} | Judge (Discriminative) ===\n{judge_raw}\n"
            log_blocks.append(
                f"=== Iteration {iteration} | Judge (Discriminative) "
                f"(obtained: {obtained}, answer: {answer or 'N/A'}) ===\n{judge_raw}"
            )

            # 论文 §2：J_d(H) = True 则辩论结束
            if obtained and answer:
                log_blocks.append(f"=== Adaptive Break at iteration {iteration} ===\nFinal: {answer}")
                return "\n\n".join(log_blocks), answer

        # ---------- Judge – Extractive (fallback) ----------
        judge_prompt = mad_judge_extractive_prompt(meta_prompt, debate_topic, history)
        judge_raw = client.generate(judge_prompt, temperature=MAD_TEMPERATURE)
        answer = parse_judge_extractive(judge_raw)

        log_blocks.append(
            f"=== Judge (Extractive, after {self.max_iterations} iterations) "
            f"(answer: {answer or 'N/A'}) ===\n{judge_raw}"
        )
        return "\n\n".join(log_blocks), answer