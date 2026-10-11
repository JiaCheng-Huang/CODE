import json
import re


def clean(text: str) -> str:
    """归一化为小写并去除尾部标点"""
    if not text:
        return ""
    text = text.strip().lower()
    if text.endswith((".", ",", ";")):
        text = text[:-1]
    return text


def extract_final_answer(text: str) -> str:
    """从 CoT / SC 输出中提取 'Final answer:' 后的标签"""
    if not text:
        return ""
    m = re.search(r"final answer:\s*(toxic|benign)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    matches = re.findall(r"\b(toxic|benign)\b", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    return clean(text)


def parse_perspectives(raw: str) -> list[str]:
    """解析 D.1 的 ### ... @@@ 格式，最多保留 9 个"""
    perspectives = re.findall(r"###\s*(.*?)\s*@@@", raw, re.DOTALL)
    return [p.strip() for p in perspectives if p.strip()][:9]


def parse_reasoning_answer(raw: str) -> tuple[str, str]:
    """解析单视角输出的 'Reasoning: ... Answer: ...'"""
    m = re.search(
        r"Reasoning:\s*(.*?)\s*Answer:\s*(toxic|benign)",
        raw, re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip(), m.group(2).lower()
    return raw.strip(), extract_final_answer(raw)


def parse_contrast(raw: str, n: int) -> tuple[list[dict], list[str]]:
    """解析 D.2：逐对差异 + checklist"""
    discrepancies: list[dict] = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            m = re.search(
                rf"For Result{i} and Result{j}:\s*(.*?)(?=For Result|Checklist:|$)",
                raw, re.DOTALL,
            )
            if m:
                discrepancies.append(
                    {"source": i, "target": j, "relation": m.group(1).strip()}
                )

    m_cl = re.search(r"Checklist:\s*(.*)", raw, re.DOTALL)
    checklist = (
        [c.strip(" -•\t") for c in re.split(r"[\n,]", m_cl.group(1)) if c.strip(" -•\t")]
        if m_cl else []
    )
    if not checklist:
        checklist = ["Re-examine the sentence and context carefully."]
    return discrepancies, checklist


def parse_reflection_json(raw: str) -> str:
    """解析 D.3 的 JSON 输出，提取最终标签"""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return extract_final_answer(raw)
        cand = (
                obj.get("Candidate")
                or obj.get("Revised Candidate")
                or obj.get("Revised Candidates")
                or {}
        )
        for key in sorted(cand.keys()):
            ans = str(cand[key].get("answer", "")).lower()
            if ans in ("toxic", "benign"):
                return ans
    return extract_final_answer(raw)


def parse_debater_answer(raw: str) -> str:
    """从辩手的自然语言论证中抽取标签。"""
    if not raw:
        return ""
    m = re.search(
        r"(?:final\s+answer|answer|conclusion|verdict)\s*[:\-]\s*(toxic|benign)",
        raw, re.IGNORECASE,
    )
    if m:
        return m.group(1).lower()
    matches = re.findall(r"\b(toxic|benign)\b", raw, re.IGNORECASE)
    return matches[-1].lower() if matches else ""


def parse_judge_discriminative(raw: str) -> tuple[bool, str]:
    """解析 Judge 的 Discriminative Mode 输出。

    返回 (solution_obtained, answer)。
    """
    if not raw:
        return False, ""
    m = re.search(
        r"solution\s+obtained\s*[:\-]\s*(true|false|yes|no)",
        raw, re.IGNORECASE,
    )
    if not m:
        return False, ""
    obtained = m.group(1).lower() in ("true", "yes")
    if not obtained:
        return False, ""
    return True, extract_final_answer(raw)


def parse_judge_extractive(raw: str) -> str:
    """解析 Judge 的 Extractive Mode 输出。"""
    return extract_final_answer(raw)


def parse_cfmad_judge(raw: str) -> str:
    """解析 CFMAD judge 输出，提取最终标签。

    优先匹配 'Judgement: The correct answer is X'；匹配失败则回退到
    extract_final_answer。
    """
    if not raw:
        return ""
    m = re.search(
        r"judgement\s*:\s*the\s+correct\s+answer\s+is\s+[\[\(]?\s*(toxic|benign)",
        raw, re.IGNORECASE,
    )
    if m:
        return m.group(1).lower()
    return extract_final_answer(raw)


def parse_ragmad_judge(text):
    # 优先匹配 Final answer: xxx
    m = re.search(r"Final answer:\s*(toxic|benign)", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # 否则取最后出现的 toxic / benign
    tokens = re.findall(r"\b(toxic|benign)\b", text, re.IGNORECASE)
    return tokens[-1].lower() if tokens else None
