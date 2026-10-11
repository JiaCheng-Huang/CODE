import pandas as pd
import re

df = pd.read_csv("results/predictions_cot.csv")

def extract_emoji(s):
    if pd.isna(s):
        return None
    emojis = re.findall(
        r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]",
        s
    )
    return emojis[0] if emojis else None

df["emoji"] = df["target_sentence"].apply(extract_emoji)

target_emojis = ['🧨', '🍑', '🌿', '🦟']

rows = []
for emoji in target_emojis:
    group = df[df["emoji"] == emoji]
    n_pairs = len(group)
    if n_pairs == 0:
        print(f"{emoji}: no data")
        continue

    pred_a = group["pred_a"].tolist()
    pred_b = group["pred_b"].tolist()

    toxic_in_benign = sum(1 for p in pred_a if p == "toxic")
    benign_in_benign = sum(1 for p in pred_a if p == "benign")
    toxic_in_toxic = sum(1 for p in pred_b if p == "toxic")
    benign_in_toxic = sum(1 for p in pred_b if p == "benign")

    total_toxic = toxic_in_benign + toxic_in_toxic
    total_benign = benign_in_benign + benign_in_toxic
    cat = group["harmful_category"].value_counts().idxmax()

    rows.append({
        "emoji": emoji,
        "category": cat,
        "n_pairs": n_pairs,
        "total_toxic": total_toxic,
        "total_benign": total_benign,
    })

print("=== Per-emoji stats ===")
for r in rows:
    print(f"{r['emoji']} ({r['category']}) n_pairs={r['n_pairs']} "
          f"total_toxic={r['total_toxic']} total_benign={r['total_benign']}")

print("\n=== LaTeX rows ===")
print("Ground truth & " + " & ".join([f"{r['n_pairs']}:{r['n_pairs']}" for r in rows]) + " \\\\")
print("  & " + " & ".join([f"{r['total_toxic']}:{r['total_benign']}" for r in rows]) + " \\\\")