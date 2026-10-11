import csv
import argparse
from collections import defaultdict


TRUE_VALUES = {"true", "1", "yes", "y", "t"}


def parse_bool(value):
    """把 CSV 中的布尔值解析成 Python bool。"""
    if value is None:
        return False
    return str(value).strip().lower() in TRUE_VALUES


def main():
    parser = argparse.ArgumentParser(
        description="按 harmful_category 分类统计预测准确率（百分比，保留一位小数，不带百分号）"
    )
    parser.add_argument(
        "--input",
        default="results/predictions_ekmad.csv",
        help="预测结果 CSV 路径，默认 results/predictions_sa.csv",
    )
    parser.add_argument(
        "--output",
        default="",
        help="可选：把分类统计结果写入指定 CSV",
    )
    parser.add_argument(
        "--category-col",
        default="harmful_category",
        help="类别列名，默认 harmful_category",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8-sig", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames or []
        rows = [
            row for row in reader
            if row and any(v not in (None, "") for v in row.values())
        ]

    if not rows:
        print(f"No data found in {args.input}")
        return

    # 检查必要列
    for col in ["correct_a", "correct_b"]:
        if col not in fieldnames:
            raise SystemExit(f"缺少列 {col!r}。实际列名: {fieldnames}")

    category_col = args.category_col
    if category_col not in fieldnames:
        # 兼容一些常见别名
        for alt in ["harmful_category", "category", "harmful_cat", "label_category"]:
            if alt in fieldnames:
                category_col = alt
                break
        else:
            raise SystemExit(
                f"缺少类别列 {args.category_col!r}。实际列名: {fieldnames}"
            )

    # 分组统计容器
    stats = defaultdict(lambda: {
        "N": 0,
        "correct_a": 0,
        "correct_b": 0,
        "correct_both": 0,
        "correct_total": 0,
    })

    # 总体统计
    total = {
        "N": 0,
        "correct_a": 0,
        "correct_b": 0,
        "correct_both": 0,
        "correct_total": 0,
    }

    for row in rows:
        category = (row.get(category_col) or "").strip() or "unknown"
        correct_a = parse_bool(row.get("correct_a"))
        correct_b = parse_bool(row.get("correct_b"))

        for target in (stats[category], total):
            target["N"] += 1
            target["correct_a"] += int(correct_a)
            target["correct_b"] += int(correct_b)
            target["correct_both"] += int(correct_a and correct_b)
            target["correct_total"] += int(correct_a) + int(correct_b)

    def calc_metrics(s):
        """返回百分比数值（0-100）。"""
        n = s["N"]
        if n == 0:
            return {**s, "Acc_a": 0.0, "Acc_b": 0.0, "Acc_p": 0.0, "Acc_q": 0.0}
        return {
            **s,
            "Acc_a": s["correct_a"] / n * 100,
            "Acc_b": s["correct_b"] / n * 100,
            "Acc_p": s["correct_both"] / n * 100,
            "Acc_q": s["correct_total"] / (2 * n) * 100,
        }

    total_calc = calc_metrics(total)

    print(f"Total instances: {total_calc['N']}")
    print(f"Acc_p = {total_calc['Acc_p']:.1f}  (both predictions correct)")
    print(f"Acc_q = {total_calc['Acc_q']:.1f}  (average per-prediction accuracy)")
    print(f"Acc_a = {total_calc['Acc_a']:.1f}")
    print(f"Acc_b = {total_calc['Acc_b']:.1f}")
    print()

    header = (
        f"{'harmful_category':<20} "
        f"{'N':>6} "
        f"{'Acc_p':>8} "
        f"{'Acc_q':>8} "
        f"{'Acc_a':>8} "
        f"{'Acc_b':>8}"
    )
    print(header)
    print("-" * len(header))

    # 按样本数从多到少排序
    ordered = sorted(stats.items(), key=lambda kv: (-kv[1]["N"], kv[0]))
    output_rows = []

    for category, s in ordered:
        c = calc_metrics(s)
        print(
            f"{category:<20} "
            f"{c['N']:>6d} "
            f"{c['Acc_p']:>8.1f} "
            f"{c['Acc_q']:>8.1f} "
            f"{c['Acc_a']:>8.1f} "
            f"{c['Acc_b']:>8.1f}"
        )
        output_rows.append({
            "harmful_category": category,
            "N": c["N"],
            "correct_a": c["correct_a"],
            "correct_b": c["correct_b"],
            "correct_both": c["correct_both"],
            "correct_total": c["correct_total"],
            "Acc_a": round(c["Acc_a"], 1),
            "Acc_b": round(c["Acc_b"], 1),
            "Acc_p": round(c["Acc_p"], 1),
            "Acc_q": round(c["Acc_q"], 1),
        })

    # 可选输出 CSV
    if args.output:
        with open(args.output, "w", encoding="utf-8-sig", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "harmful_category",
                    "N",
                    "correct_a",
                    "correct_b",
                    "correct_both",
                    "correct_total",
                    "Acc_a",
                    "Acc_b",
                    "Acc_p",
                    "Acc_q",
                ],
            )
            writer.writeheader()
            writer.writerows(output_rows)

        print(f"\n分类统计已写入: {args.output}")


if __name__ == "__main__":
    main()
