import json


# ============ 原有单轮模式 ============

def sa_evaluate(target_sentence, context):
    return (
        f"I’ll give you a text with Emoji and some additional context, which provides information closely related to the text. "
        f"Please answer my question based on the text and the context.\n"
        f"Context: \"{context}\"\n"
        f"Sentence: \"{target_sentence}\"\n"
        f"Question: Based on the context, determine whether this sentence belongs to a toxic context including sexual harassment, hate speech, illegal dealing, and terrorism. "
        f"If it is toxic, answer 'toxic'. If it is benign and not harmful, answer 'benign'. "
        f"Output only one of these two words in lowercase."
    )


def cot_evaluate(target_sentence, context):
    return (
        f"I’ll give you a text with Emoji and some additional context, which provides information closely related to the text. "
        f"Please answer my question based on the text and the context.\n"
        f"Context: \"{context}\"\n"
        f"Sentence: \"{target_sentence}\"\n"
        f"Question: Based on the context, determine whether this sentence belongs to a toxic context including sexual harassment, hate speech, illegal dealing, and terrorism. "
        f"If it is toxic, answer 'toxic'. If it is benign and not harmful, answer 'benign'.\n\n"
        f"Please think step by step. First provide your reasoning, then give the final answer in the exact format below:\n"
        f"Reasoning: <your step-by-step analysis>\n"
        f"Final answer: <toxic or benign>\n"
        f"Important: The final answer must be exactly one lowercase word, either 'toxic' or 'benign'. "
        f"Do not output anything else after the final answer."
    )


# ============ Self-Contrast: D.1 Self-Curated Process ============

def sc_generate_perspectives(target_sentence, context):
    return (
        f"You are a toxicity detection specialist who specializes in detecting toxic content from diverse perspectives. "
        f"Given a sentence with emoji and a context, you need to carefully analyze the sentence and context, and dynamically generate "
        f"several useful prompt instructions. These prompt instructions should be diverse and also useful for toxicity detection. "
        f"These prompt instructions are used to guide the language model to think in different ways, attention to different emphases, "
        f"and reason from different perspectives for more accurate toxicity judgment.\n\n"
        f"For instance, you can adopt multi-faceted thinking (literal meaning, metaphorical meaning, social implication, potential harm, etc.), "
        f"different reasoning perspectives (e.g., word-level, sentence-level, context-level), and different emphases of concern "
        f"(entity words, emojis, harmful categories such as sexual harassment, hate speech, illegal dealing, terrorism, etc.) "
        f"for the input sentence in prompt instruction.\n\n"
        f"Here are some guidance rules for Prompt Generation:\n"
        f"1. Tone Requirement: Please generate prompt instructions in the third person.\n"
        f"2. Content Requirement: Each prompt instruction should adopt a different way of thinking, or focus on a different perspective, "
        f"or different emphases to detect toxicity.\n"
        f"3. Number Requirement: Dynamically generate the most valuable 2 to 9 prompt instructions based on the input sentence and context.\n"
        f"4. Format Requirement: Each prompt instruction should start with ### and end with @@@.\n"
        f"5. Others: Prompt instructions should focus on toxicity detection. So don't ask any other irrelevant questions in the prompt.\n\n"
        f"Here is an example: The sentence is: \"I will kill you\" with context: \"Two friends are playing a violent video game.\"\n"
        f"Output:\n"
        f"### You are a literal evaluator. Please focus on the direct meaning of the words without considering the context. "
        f"Judge whether the sentence itself contains explicit threats or harmful content. @@@\n"
        f"### You are a context-aware evaluator. Please consider the given context carefully, and judge whether the sentence becomes "
        f"benign under this specific context. @@@\n"
        f"### You are a harm evaluator. Please focus on the potential harm the sentence may cause, including physical, psychological, "
        f"or social harm. @@@\n\n"
        f"The sentence is: \"{target_sentence}\" with context: \"{context}\". Please generate the most suitable prompts:"
    )


def sc_evaluate_with_perspective(target_sentence, context, perspective):
    """Stage 1 后续：用单个视角提示生成判断"""
    return (
        f"{perspective}\n\n"
        f"Context: \"{context}\"\n"
        f"Sentence: \"{target_sentence}\"\n"
        f"Based on the above perspective, determine whether the sentence is toxic in the given context. "
        f"If it is toxic, answer 'toxic'. If it is benign and not harmful, answer 'benign'.\n\n"
        f"Output format:\n"
        f"Reasoning: <your reasoning>\n"
        f"Answer: <toxic or benign>"
    )


# ============ Self-Contrast: D.2 Contrasting Process ============

def sc_contrast_and_checklist(target_sentence, context, results):
    results_text = ", ".join([f"Result{i}" for i in range(1, len(results) + 1)])
    results_content = "\n".join([
        f"Result{i}: Reasoning: {r['reasoning']} Answer: {r['answer']}"
        for i, r in enumerate(results, 1)
    ])

    # 生成逐对比较的输出格式说明
    pair_lines = []
    n = len(results)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            pair_lines.append(f"For Result{i} and Result{j}: {{Difference{i}_{j}}}")
    pair_format = "\n".join(pair_lines)

    return (
        f"You are a toxicity detection specialist who specializes in detecting toxic content. "
        f"Given some candidate judgments for a sentence under a specific context, you should carefully compare the difference "
        f"for each two judgments in their reasoning and final answers.\n\n"
        f"When you compare, you need to consider the following questions:\n"
        f"1: Do the two judgments have different final answers (toxic or benign) and different reasoning?\n"
        f"2: Where are the differences in their reasoning and judgment?\n"
        f"3: Why are the answers of the two judgments different?\n\n"
        f"After contrasting, you should generate a checklist based on these differences between candidate judgments. "
        f"You should carefully consider each discrepancy and the reasons behind it, summarizing them into a few checking instructions "
        f"in the checklist. This checklist can guide others to re-examine the input sentence and context and these candidate judgments "
        f"to eliminate these discrepancies.\n\n"
        f"Input Format: The sentence is \"{target_sentence}\". The context is \"{context}\". All results: {results_text}.\n"
        f"Detailed results:\n{results_content}\n\n"
        f"Output Format:\n"
        f"{pair_format}\n"
        f"Checklist: {{Directive1, Directive2, ...}}"
    )


# ============ Self-Contrast: D.3 Reflection Stage ============

def sc_reflect_final(target_sentence, context, results, discrepancies, checklist):
    """
    results: list of {"reasoning": ..., "answer": ...}
    discrepancies: list of {"source": int, "target": int, "relation": str}
    checklist: list of str
    """
    candidate_json = {
        "Candidate": {
            f"result_{i}": {"answer": r["answer"], "solution": r["reasoning"]}
            for i, r in enumerate(results, 1)
        },
        "Discrepancy": {
            f"difference_{d['source']}_{d['target']}": {
                "source": f"result_{d['source']}",
                "target": f"result_{d['target']}",
                "relation": d["relation"],
            }
            for d in discrepancies
        },
    }
    candidate_str = json.dumps(candidate_json, ensure_ascii=False, indent=2)
    checklist_str = ", ".join(checklist)

    return (
        f"Given a sentence with emoji, a context, multiple inconsistent judgments, their differences in their reasoning processes, "
        f"and a checklist. You should revise the inconsistent reasoning for each judgment, eliminate the differences, and output a new "
        f"judgment for each candidate.\n\n"
        f"Guidance Rules for Reflection:\n"
        f"1. Please check carefully according to the requirements on the checklist. It helps you to resolve conflicts between different judgments.\n"
        f"2. When you finish revising inconsistent judgments, please ensure all revised judgments should have the same answer. "
        f"If not, please revise again until all inconsistencies are removed, and all candidates are consistent.\n"
        f"3. Please output all revised judgments in JSON format as input, without any other text.\n\n"
        f"The sentence is \"{target_sentence}\". The context is \"{context}\".\n"
        f"The candidate judgments and their discrepancy are as follows:\n"
        f"{candidate_str}\n"
        f"Checklist: {checklist_str}\n"
        f"Please revise each inconsistent judgment."
    )


# ============ Multi-Agent Debate (Liang et al., EMNLP 2024) ============

def mad_debate_topic(target_sentence, context):
    """把任务转换成辩论主题（论文 §3.2 的 <debate topic> 占位）。"""
    return (
        f"Based on the given context, determine whether the following sentence is toxic, "
        f"including sexual harassment, hate speech, illegal dealing, and terrorism. "
        f"If it is toxic, answer 'toxic'. If it is benign and not harmful, answer 'benign'.\n"
        f"Sentence: \"{target_sentence}\"\n"
        f"Context: \"{context}\""
    )


def mad_meta_prompt(debate_topic):
    """论文 §2 "Meta Prompts"，采用 Table 11 默认 level-2 'tit for tat'。"""
    return (
        f"You are a debater. Hello and welcome to the debate competition. "
        f"It's not necessary to fully agree with each other's perspectives, as our objective is to find the correct answer. "
        f"The debate topic is stated as follows: {debate_topic}"
    )


def mad_affirmative_prompt(meta_prompt, history):
    """论文 §2 Affirmative Debater prompt。"""
    return (
        f"{meta_prompt}\n\n"
        f"Debate history:\n{history if history else '(No previous rounds.)'}\n\n"
        f"You are affirmative side. Please express your viewpoints."
    )


def mad_negative_prompt(meta_prompt, history, affirmative_latest):
    """论文 §2 Negative Debater prompt。"""
    return (
        f"{meta_prompt}\n\n"
        f"Debate history:\n{history if history else '(No previous rounds.)'}\n\n"
        f"Affirmative side's latest argument:\n{affirmative_latest}\n\n"
        f"You are negative side. You disagree with the affirmative side's points. "
        f"Provide your reasons and answer."
    )


def mad_judge_discriminative_prompt(meta_prompt, debate_topic, history):
    """论文 §2 Judge – Discriminative Mode J_d(H) ∈ {True, False}。"""
    return (
        f"{meta_prompt}\n\n"
        f"You are a moderator. There will be two debaters involved in a debate competition. "
        f"They will present their answers and discuss their perspectives on the following topic:\n"
        f"{debate_topic}\n\n"
        f"Debate history:\n{history}\n\n"
        f"At the end of this round, please evaluate both sides' answers and decide which one is correct.\n"
        f"If a correct solution has been obtained, output exactly:\n"
        f"Solution obtained: True\n"
        f"Final answer: <toxic or benign>\n\n"
        f"Otherwise, output exactly:\n"
        f"Solution obtained: False"
    )


def mad_judge_extractive_prompt(meta_prompt, debate_topic, history):
    """论文 §2 Judge – Extractive Mode J_e(H) = a。"""
    return (
        f"{meta_prompt}\n\n"
        f"You are a moderator. There will be two debaters involved in a debate competition. "
        f"They will present their answers and discuss their perspectives on the following topic:\n"
        f"{debate_topic}\n\n"
        f"Debate history:\n{history}\n\n"
        f"The maximum number of debate iterations has been reached without a clear consensus. "
        f"Please extract the final solution based on the whole debate history.\n\n"
        f"Output format:\n"
        f"Final answer: <toxic or benign>"
    )


# ============ CFMAD (Fang et al., COLING 2025) ============

def cfmad_abduction(target_sentence, context, preset_answer):
    """附录 D.2 Abduction Generation 结构：让模型假设某标签是对的，产出理由。"""
    return (
        f"You are a toxicity detection specialist. Given a sentence with emoji and its context, "
        f"try to explain why the sentence's label might be \"{preset_answer}\".\n"
        f"Sentence: \"{target_sentence}\"\n"
        f"Context: \"{context}\"\n\n"
        f"Output Format:\n"
        f"Judgement: The answer is {preset_answer}.\n"
        f"Reasoning: [Your reasoning here]"
    )


def cfmad_critic(target_sentence, context, agent_reply):
    """附录 D.2 Counterfactual Debate for Critic。"""
    return (
        f"You are a toxicity detection specialist.\n"
        f"Sentence: \"{target_sentence}\"\n"
        f"Context: \"{context}\"\n"
        f"Assistant: {agent_reply}\n\n"
        f"The Assistant's answer may be wrong. Please persuade the assistant that his answer may be wrong."
    )


def cfmad_agent_defense(target_sentence, context, agent_reply, critic_reply):
    """附录 D.2 Counterfactual Debate for Assistant。"""
    return (
        f"You are a toxicity detection specialist.\n"
        f"Sentence: \"{target_sentence}\"\n"
        f"Context: \"{context}\"\n"
        f"Assistant: {agent_reply}\n"
        f"Critic: {critic_reply}\n\n"
        f"As assistant, please refute the critic's answer and persuade the critic that your answer is correct."
    )


def cfmad_judge(target_sentence, context, debate_logs):
    """附录 D.2 Judge。"""
    return (
        f"You are a toxicity detection specialist.\n"
        f"Sentence: \"{target_sentence}\"\n"
        f"Context: \"{context}\"\n\n"
        f"The results of the analysis for each of the possible labels are as follows:\n"
        f"{debate_logs}\n\n"
        f"After seeing the debate process above, do you think which label is the most appropriate answer "
        f"for the question? Please only give a correct answer and no other replies.\n"
        f"Output format:\n"
        f"Judgement: The correct answer is [toxic/benign].\n"
        f"Reasoning steps: [Your precise reasoning steps here]"
    )

# ============ RAG-MAD (Retrieval-Augmented Multi-Agent Debate) ============

def _ragmad_shared_knowledge(knowledge):
    """仅负责将检索到的知识文本包装为可追加的片段，不假设内部格式。"""
    return (
        f"Retrieved external emoji knowledge (from an emoji knowledge base):\n"
        f"{knowledge}"
    )


def ragmad_meta_prompt(debate_topic, knowledge):
    """复用 MAD 的 meta prompt，仅追加检索到的 emoji 知识。"""
    return (
        f"{mad_meta_prompt(debate_topic)}\n\n"
        f"{_ragmad_shared_knowledge(knowledge)}"
    )