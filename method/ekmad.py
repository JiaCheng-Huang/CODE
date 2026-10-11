from method.base import BaseMode
from parsers import parse_debater_answer, parse_ragmad_judge
from prompts import (
    mad_debate_topic,
    mad_affirmative_prompt,
    mad_negative_prompt,
    mad_judge_extractive_prompt,
    ragmad_meta_prompt,
)
from rag.retriever import EmojiRetriever
from utils.emoji_utils import get_emoji_name, extract_emojis

# 论文 §4.2: 采样温度恒为 0（greedy decode），与 MAD / CFMAD 一致
RAGMAD_TEMPERATURE = 0.0

# 论文 §4.2: 默认辩论轮数 R = 2（消融 §5.3 讨论 R 的影响）
RAGMAD_DEBATE_ROUNDS = 2


class EKMadMode(BaseMode):
    name = "ekmad"
    multi_run = False

    def __init__(
            self,
            debate_rounds: int = RAGMAD_DEBATE_ROUNDS,
            retriever_model_path: str = "intfloat/e5-base-v2",
            retriever_index_path: str = "rag/emoji_index_db/index.faiss",
            retriever_corpus_path: str = "rag/emoji_corpus.jsonl",
    ):
        self.debate_rounds = debate_rounds
        self.retriever = EmojiRetriever(
            model_path=retriever_model_path,
            index_path=retriever_index_path,
            corpus_path=retriever_corpus_path,
        )

    # ------------------------------------------------------------------
    def run(self, client, target_sentence, context):
        # ---------- Stage 1: RAG ----------
        emojis = extract_emojis(target_sentence)
        emoji_name = get_emoji_name(emojis)
        knowledge = self.retriever.get_knowledge(emoji_name)

        # 复用 MAD 的 debate topic，并构造带知识的 meta prompt
        debate_topic = mad_debate_topic(target_sentence, context)
        meta_prompt = ragmad_meta_prompt(debate_topic, knowledge)

        history = ""
        log_blocks = [f"=== Stage 1 | Retrieved emoji knowledge ===\n{knowledge}"]

        # ---------- Stage 2: Debate（完全复用 MAD 的辩手 prompt） ----------
        for r in range(1, self.debate_rounds + 1):
            # Toxic advocate = affirmative side
            affirmative_prompt = mad_affirmative_prompt(meta_prompt, history)
            affirmative_raw = client.generate(
                affirmative_prompt, temperature=RAGMAD_TEMPERATURE
            )
            affirmative_ans = parse_debater_answer(affirmative_raw)
            history += f"\n=== Round {r} | Affirmative (Toxic) ===\n{affirmative_raw}\n"
            log_blocks.append(
                f"=== Round {r} | Affirmative (Toxic) "
                f"(extracted: {affirmative_ans or 'N/A'}) ===\n{affirmative_raw}"
            )

            # Benign advocate = negative side
            negative_prompt = mad_negative_prompt(
                meta_prompt, history, affirmative_raw
            )
            negative_raw = client.generate(
                negative_prompt, temperature=RAGMAD_TEMPERATURE
            )
            negative_ans = parse_debater_answer(negative_raw)
            history += f"\n=== Round {r} | Negative (Benign) ===\n{negative_raw}\n"
            log_blocks.append(
                f"=== Round {r} | Negative (Benign) "
                f"(extracted: {negative_ans or 'N/A'}) ===\n{negative_raw}"
            )

        # ---------- Stage 3: Judge（完全复用 MAD 的 Judge prompt） ----------
        judge_raw = client.generate(
            mad_judge_extractive_prompt(meta_prompt, debate_topic, history),
            temperature=RAGMAD_TEMPERATURE,
        )
        final_answer = parse_ragmad_judge(judge_raw)

        log_blocks.append(f"=== Stage 3 | Judge ===\n{judge_raw}")
        log_blocks.append(f"=== Final answer ===\n{final_answer}")
        return "\n\n".join(log_blocks), final_answer
