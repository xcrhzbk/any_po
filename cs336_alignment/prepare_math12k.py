import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any

import pyarrow.parquet as pq


DEFAULT_DATASET = "hiyouga/math12k"
DEFAULT_OUTPUT_DIR = "/home/bkzhu/storage/assignment5-alignment/data/math12k"
DEFAULT_PROMPT_PATH = "/home/bkzhu/storage/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
BASE_URL = "https://datasets-server.huggingface.co"


def http_get_json(url: str, timeout: int = 60) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "cs336-alignment/prepare-math12k"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def query_splits(dataset: str) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({"dataset": dataset})
    url = f"{BASE_URL}/splits?{params}"
    data = http_get_json(url)
    return data["splits"]


def query_repo_tree(dataset: str) -> list[dict[str, Any]]:
    url = f"https://huggingface.co/api/datasets/{dataset}/tree/main?recursive=1"
    return http_get_json(url)


def find_parquet_files(tree: list[dict[str, Any]]) -> tuple[str, str]:
    train_path = ""
    test_path = ""
    for item in tree:
        if item.get("type") != "file":
            continue
        path = str(item.get("path", ""))
        if path.endswith(".parquet") and "train" in os.path.basename(path):
            train_path = path
        if path.endswith(".parquet") and "test" in os.path.basename(path):
            test_path = path
    if not train_path or not test_path:
        raise RuntimeError("Cannot find train/test parquet files in dataset repo tree.")
    return train_path, test_path


def download_file(url: str, output_path: str, timeout: int = 120) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "cs336-alignment/prepare-math12k"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(output_path, "wb") as f:
        f.write(resp.read())


def load_parquet_rows(path: str) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    return table.to_pylist()


def fetch_rows(
    dataset: str,
    config: str,
    split: str,
    num_rows: int,
    chunk_size: int = 100,
    sleep_s: float = 0.05,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while offset < num_rows:
        length = min(chunk_size, num_rows - offset)
        params = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            }
        )
        url = f"{BASE_URL}/rows?{params}"
        data = http_get_json(url)
        for item in data["rows"]:
            rows.append(item["row"])
        offset += length
        if sleep_s > 0:
            time.sleep(sleep_s)
    return rows


def normalize_answer_for_anypo(answer: str) -> str:
    answer = answer.strip()
    if answer.startswith("####"):
        return answer
    return f"#### {answer}"


def build_anypo_jsonl(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        problem = str(row.get("problem", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not problem or not answer:
            continue
        out.append({"question": problem, "answer": normalize_answer_for_anypo(answer)})
    return out


def build_ei_processed_jsonl(rows: list[dict[str, Any]], prompt_template: str) -> list[dict[str, str]]:
    out = []
    for row in rows:
        problem = str(row.get("problem", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not problem or not answer:
            continue
        prompt = prompt_template.format(question=problem)
        response = f"<think> </think> <answer> {answer} </answer>"
        out.append({"prompt": prompt, "response": response})
    return out


def dump_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess math12k for AnyPO/EI.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt_path", type=str, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--sleep_s", type=float, default=0.02)
    args = parser.parse_args()

    with open(args.prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    tree = query_repo_tree(args.dataset)
    train_parquet, test_parquet = find_parquet_files(tree)
    train_url = f"https://huggingface.co/datasets/{args.dataset}/resolve/main/{train_parquet}"
    test_url = f"https://huggingface.co/datasets/{args.dataset}/resolve/main/{test_parquet}"

    raw_parquet_dir = os.path.join(args.output_dir, "raw_parquet")
    local_train_parquet = os.path.join(raw_parquet_dir, os.path.basename(train_parquet))
    local_test_parquet = os.path.join(raw_parquet_dir, os.path.basename(test_parquet))
    print(f"Downloading train parquet: {train_url}")
    download_file(train_url, local_train_parquet)
    print(f"Downloading test parquet: {test_url}")
    download_file(test_url, local_test_parquet)

    train_rows = load_parquet_rows(local_train_parquet)
    test_rows = load_parquet_rows(local_test_parquet)

    train_anypo = build_anypo_jsonl(train_rows)
    test_anypo = build_anypo_jsonl(test_rows)
    train_ei = build_ei_processed_jsonl(train_rows, prompt_template)

    dump_jsonl(os.path.join(args.output_dir, "train_raw.jsonl"), train_rows)
    dump_jsonl(os.path.join(args.output_dir, "test_raw.jsonl"), test_rows)
    dump_jsonl(os.path.join(args.output_dir, "train.jsonl"), train_anypo)
    dump_jsonl(os.path.join(args.output_dir, "test.jsonl"), test_anypo)
    dump_jsonl(os.path.join(args.output_dir, "processed_train.jsonl"), train_ei)

    print(f"Saved: {os.path.join(args.output_dir, 'train.jsonl')} ({len(train_anypo)} rows)")
    print(f"Saved: {os.path.join(args.output_dir, 'test.jsonl')} ({len(test_anypo)} rows)")
    print(f"Saved: {os.path.join(args.output_dir, 'processed_train.jsonl')} ({len(train_ei)} rows)")


if __name__ == "__main__":
    main()
