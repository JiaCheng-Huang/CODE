import argparse
import json

from llms import create_client
from method import available_modes, get_mode
from utils.io_utils import CsvStore
from runner import Runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", type=str, choices=available_modes(), default="sa",
        help="推理模式：sa / cot / sc / self_consistency / mad / cfmad / ekmad",
    )
    parser.add_argument("--data", type=str, default="data/benchmark.jsonl")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--n_runs", type=int, default=1,
    )
    parser.add_argument(
        "--check_idle", action="store_true",
        help="开启后每次请求前检查 DeepSeek 空闲时段；未开启则直接调用 API",
    )
    return parser.parse_args()


def load_records(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    args = parse_args()
    method = get_mode(args.method)
    records = load_records(args.data)

    csv_path = f"{args.results_dir}/predictions_{method.name}.csv"
    print(f"Mode: {method.name}, n_runs: {args.n_runs}, check_idle: {args.check_idle}")
    print(f"Output: {csv_path}")

    client = create_client("deepseek")
    with CsvStore(csv_path) as store:
        Runner(
            client, method, store,
            n_runs=args.n_runs,
            check_idle=args.check_idle,
        ).run(records)

    print("Done! All remaining records processed.")


if __name__ == "__main__":
    main()