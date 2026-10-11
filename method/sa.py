from method.base import BaseMode
from prompts import sa_evaluate
from parsers import clean
from settings import temperature, seed, top_p


class SaMode(BaseMode):
    name = "sa"

    def run(self, client, target_sentence, context):
        raw = client.generate(
            sa_evaluate(target_sentence, context),
            temperature=temperature, seed=seed, top_p=top_p,
        )
        return raw, clean(raw)