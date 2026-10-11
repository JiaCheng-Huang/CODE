from method.base import BaseMode
from prompts import cot_evaluate
from parsers import extract_final_answer
from settings import temperature, seed, top_p


class CotMode(BaseMode):
    name = "cot"

    def run(self, client, target_sentence, context):
        raw = client.generate(
            cot_evaluate(target_sentence, context),
            temperature=temperature, seed=seed, top_p=top_p,
        )
        return raw, extract_final_answer(raw)