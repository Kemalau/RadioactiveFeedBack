from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.split())


def read_tasks(path: str | Path) -> list[dict]:
    tasks = []
    ids = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"task row {line_number} must be a JSON object")
        for name in ("task_id", "prompt"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"task row {line_number} requires nonempty {name}")
        if row["task_id"] in ids:
            raise ValueError(f"duplicate task_id on row {line_number}")
        ids.add(row["task_id"])
        task = {"task_id": row["task_id"], "prompt": row["prompt"]}
        if "carrier_id" in row:
            if not isinstance(row["carrier_id"], str) or not row["carrier_id"]:
                raise ValueError(f"invalid carrier_id on row {line_number}")
            task["carrier_id"] = row["carrier_id"]
        tasks.append(task)
    if not tasks:
        raise ValueError("task file is empty")
    return tasks


def prepare_tasks(tasks: list[dict], audit_tasks: list[dict], seed: int) -> tuple[list[dict], dict]:
    audit_prompts = {normalize_prompt(row["prompt"]) for row in audit_tasks}
    audit_ids = {row["task_id"] for row in audit_tasks}
    seen = set()
    retained = []
    overlap_count = duplicate_count = 0
    for row in tasks:
        normalized = normalize_prompt(row["prompt"])
        if normalized in audit_prompts or row["task_id"] in audit_ids:
            overlap_count += 1
        elif normalized in seen:
            duplicate_count += 1
        else:
            seen.add(normalized)
            retained.append(dict(row))
    if not retained:
        raise ValueError("no training prompts remain after deduplication")
    random.Random(seed).shuffle(retained)
    metadata = {
        "source_prompts": len(tasks), "retained_prompts": len(retained),
        "removed_audit_overlap": overlap_count, "removed_train_duplicates": duplicate_count,
        "audit_prompts": len(audit_tasks), "audit_panel_supplied": bool(audit_tasks),
        "source_sha256": digest_json(tasks), "audit_sha256": digest_json(audit_tasks),
        "trajectory_sha256": digest_json(retained),
    }
    return retained, metadata


def task_batches(tasks: list[dict], size: int):
    if size <= 0:
        raise ValueError("prompt batch size must be positive")
    for start in range(0, len(tasks), size):
        yield tasks[start:start + size]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
