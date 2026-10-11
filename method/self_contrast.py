from method.base import BaseMode
from prompts import (
    sc_generate_perspectives,
    sc_evaluate_with_perspective,
    sc_contrast_and_checklist,
    sc_reflect_final,
)
from parsers import (
    parse_perspectives,
    parse_reasoning_answer,
    parse_contrast,
    parse_reflection_json,
)
from settings import temperature, seed, top_p

DEFAULT_PERSPECTIVES = [
    "You are a literal evaluator. Please focus on the direct meaning of the words "
    "without considering the context. Judge whether the sentence itself contains "
    "explicit threats or harmful content.",
    "You are a context-aware evaluator. Please consider the given context carefully, "
    "and judge whether the sentence becomes benign under this specific context.",
    "You are a harm evaluator. Please focus on the potential harm the sentence may "
    "cause, including physical, psychological, or social harm.",
]


class SelfContrastMode(BaseMode):
    """Self-Contrast（论文 2024.acl-long.197 附录 D）：
        D.1 Self-Curated Process → 逐视角判断
        D.2 Contrasting Process → 逐对差异 + checklist
        D.3 Reflection Stage   → JSON 反思输出
    """

    name = "self_contrast"

    def run(self, client, target_sentence, context):
        # ---------- Stage 1: Self-Curated Process ----------
        perspectives_raw = client.generate(
            sc_generate_perspectives(target_sentence, context),
            temperature=temperature, seed=seed, top_p=top_p,
        )
        perspectives = parse_perspectives(perspectives_raw) or DEFAULT_PERSPECTIVES

        results = []
        for persp in perspectives:
            raw = client.generate(
                sc_evaluate_with_perspective(target_sentence, context, persp),
                temperature=temperature, seed=seed, top_p=top_p,
            )
            reasoning, answer = parse_reasoning_answer(raw)
            results.append({"perspective": persp, "reasoning": reasoning, "answer": answer})

        # ---------- Stage 2: Contrast + Checklist ----------
        contrast_raw = client.generate(
            sc_contrast_and_checklist(target_sentence, context, results),
            temperature=temperature, seed=seed, top_p=top_p,
        )
        discrepancies, checklist = parse_contrast(contrast_raw, len(results))

        # ---------- Stage 3: Reflection ----------
        final_raw = client.generate(
            sc_reflect_final(target_sentence, context, results, discrepancies, checklist),
            temperature=temperature, seed=seed, top_p=top_p,
        )
        final_answer = parse_reflection_json(final_raw)

        return self._merge_logs(perspectives_raw, results, contrast_raw, final_raw), final_answer

    @staticmethod
    def _merge_logs(perspectives_raw, results, contrast_raw, final_raw) -> str:
        individual = "\n".join(
            f"Perspective: {r['perspective']}\n"
            f"Reasoning: {r['reasoning']}\n"
            f"Answer: {r['answer']}"
            for r in results
        )
        return (
            f"=== D.1 Self-Curated Prompts ===\n{perspectives_raw}\n\n"
            f"=== D.1 Individual Results ===\n{individual}\n\n"
            f"=== D.2 Contrast & Checklist ===\n{contrast_raw}\n\n"
            f"=== D.3 Reflection ===\n{final_raw}"
        )
