from method.base import BaseMode
from prompts import (
    cfmad_abduction,
    cfmad_critic,
    cfmad_agent_defense,
    cfmad_judge,
)
from parsers import parse_cfmad_judge


CFMAD_TEMPERATURE = 0.0
CFMAD_DEBATE_ROUNDS = 1

# 二分类任务直接使用两个互斥标签作为预设立场（§B.2 的处理方式）
CFMAD_PRESET_ANSWERS = ("toxic", "benign")


class CfMadMode(BaseMode):
    """CounterFactual Multi-Agent Debate (Fang et al., COLING 2025)。

    严格按论文 §3 两阶段实现：

    Stage 1 — Abduction Generation（§3.1）
        对每个预设立场 a_i，让模型假设 a_i 正确并生成理由 r_i。
        即便 a_i 是错误的，模型也会按指令做反事实推理产出看似合理的理由。

    Stage 2 — Counterfactual Debate（§3.2）
        对每个 (agent g_i, critic c_i) 对，进行 1 轮对抗辩论：
            critic: 读 r_i → 反驳
            agent : 读 r_i 与 critic 反驳 → 自辩
        所有 pair 的辩论过程送入第三方 Judge 做最终裁决。

    与论文的对应：
        - 预设立场数量：二分类任务用 2 个（= toxic / benign），与 §B.2 一致
        - 辩论轮数：默认 1（§4.4 + Fig.6B）
        - 温度：0.2（§4.1）
    """

    name = "cfmad"
    multi_run = False

    def __init__(
        self,
        preset_answers: tuple[str, ...] = CFMAD_PRESET_ANSWERS,
        debate_rounds: int = CFMAD_DEBATE_ROUNDS,
    ):
        self.preset_answers = list(preset_answers)
        self.debate_rounds = debate_rounds

    # ------------------------------------------------------------------
    def run(self, client, target_sentence, context):
        # ================= Stage 1: Abduction Generation =================
        abductions: dict[str, str] = {}
        for stance in self.preset_answers:
            prompt = cfmad_abduction(target_sentence, context, stance)
            raw = client.generate(prompt, temperature=CFMAD_TEMPERATURE)
            abductions[stance] = raw

        # ================= Stage 2: Counterfactual Debate =================
        # 每个 stance 单独和 critic 进行 debate_rounds 轮对话
        pair_logs: list[dict] = []
        for stance in self.preset_answers:
            current_agent_reply = abductions[stance]
            critic_replies: list[str] = []
            agent_defenses: list[str] = []

            for _ in range(self.debate_rounds):
                critic_prompt = cfmad_critic(target_sentence, context, current_agent_reply)
                critic_raw = client.generate(critic_prompt, temperature=CFMAD_TEMPERATURE)
                critic_replies.append(critic_raw)

                defense_prompt = cfmad_agent_defense(
                    target_sentence, context, current_agent_reply, critic_raw
                )
                defense_raw = client.generate(defense_prompt, temperature=CFMAD_TEMPERATURE)
                agent_defenses.append(defense_raw)
                current_agent_reply = defense_raw

            pair_logs.append({
                "stance": stance,
                "abduction": abductions[stance],
                "critic_replies": critic_replies,
                "agent_defenses": agent_defenses,
            })

        # ================= Judge =================
        debate_text = self._format_debate_logs(pair_logs)
        judge_prompt = cfmad_judge(target_sentence, context, debate_text)
        judge_raw = client.generate(judge_prompt, temperature=CFMAD_TEMPERATURE)
        final_answer = parse_cfmad_judge(judge_raw)

        # ================= Merge logs =================
        merged = self._merge_logs(pair_logs, judge_raw, final_answer)
        return merged, final_answer

    # ------------------------------------------------------------------
    @staticmethod
    def _format_debate_logs(pair_logs: list[dict]) -> str:
        """把每个 pair 的辩论过程拼成 judge 输入（对应论文 Fig.3 底部）。"""
        blocks = []
        for pair in pair_logs:
            b = [f"=== Preset stance: {pair['stance']} ==="]
            b.append(f"[Abduction]\n{pair['abduction']}")
            for r, (c, d) in enumerate(
                zip(pair["critic_replies"], pair["agent_defenses"]), 1
            ):
                b.append(f"[Round {r} - Critic]\n{c}")
                b.append(f"[Round {r} - Agent defense]\n{d}")
            blocks.append("\n".join(b))
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    @staticmethod
    def _merge_logs(pair_logs, judge_raw, final_answer) -> str:
        blocks = ["=== Stage 1: Abduction Generation ==="]
        for pair in pair_logs:
            blocks.append(f"--- Stance: {pair['stance']} ---\n{pair['abduction']}")

        blocks.append("\n=== Stage 2: Counterfactual Debate ===")
        for pair in pair_logs:
            blocks.append(f"--- Stance: {pair['stance']} ---")
            for r, (c, d) in enumerate(
                zip(pair["critic_replies"], pair["agent_defenses"]), 1
            ):
                blocks.append(f"[Round {r} - Critic]\n{c}")
                blocks.append(f"[Round {r} - Agent defense]\n{d}")

        blocks.append(f"\n=== Judge ===\n{judge_raw}")
        blocks.append(f"=== Final answer ===\n{final_answer}")
        return "\n\n".join(blocks)