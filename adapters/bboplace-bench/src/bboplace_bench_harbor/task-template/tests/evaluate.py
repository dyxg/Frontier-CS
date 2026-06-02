#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from urllib import error, request

SOLUTION_PATH = Path("/app/solution.py")
REWARD_TXT = Path("/logs/verifier/reward.txt")
REWARD_JSON = Path("/logs/verifier/reward.json")
EVALUATION_JSON = Path("/logs/verifier/evaluation_result.json")
AGENT_SUBMISSIONS_LOG = Path("/logs/agent/submissions.jsonl")
VERIFIER_SUBMISSIONS_LOG = Path("/logs/verifier/submissions.jsonl")
JUDGE_URL = "http://judge:8082"
JUDGE_TIMEOUT_SECONDS = 10800


def best_submission() -> dict | None:
    records: list[dict] = []
    try:
        with request.urlopen(f"{JUDGE_URL}/submissions", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        http_records = payload.get("submissions", [])
        if isinstance(http_records, list):
            records.extend(record for record in http_records if isinstance(record, dict))
    except Exception as exc:
        print(f"WARN: failed to fetch judge submissions: {exc}")

    seen: set[str] = set()
    best: dict | None = None
    VERIFIER_SUBMISSIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with VERIFIER_SUBMISSIONS_LOG.open("w", encoding="utf-8") as dst:
        for record in records:
            key = str(record.get("submission_uuid") or json.dumps(record, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record.get("status") != "done":
                continue
            try:
                reward = float(record.get("reward", 0.0))
            except (TypeError, ValueError):
                continue
            if best is None or reward > float(best.get("reward", 0.0)):
                best = record
    return best


def copy_agent_submissions_log() -> None:
    if AGENT_SUBMISSIONS_LOG.exists():
        target = Path("/logs/verifier/agent_submissions.jsonl")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(AGENT_SUBMISSIONS_LOG, target)
        except OSError as exc:
            print(f"WARN: failed to copy agent submissions log: {exc}")


def post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=JUDGE_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"judge HTTP {exc.code}: {detail}") from exc


def fetch_info() -> dict:
    with request.urlopen(f"{JUDGE_URL}/info", timeout=JUDGE_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") != "ok":
        raise RuntimeError(str(payload.get("error") or payload))
    return payload["info"]


def evaluate_final_solution() -> dict:
    import importlib.util
    import numpy as np

    info = fetch_info()
    module_name = f"bboplace_final_solution_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, SOLUTION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load solution from {SOLUTION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "solve"):
        value = module.solve(info)
    elif hasattr(module, "generate"):
        value = module.generate(info)
    elif hasattr(module, "CANDIDATES"):
        value = module.CANDIDATES
    elif hasattr(module, "CANDIDATE"):
        value = module.CANDIDATE
    else:
        raise RuntimeError("solution must define solve(info), generate(info), CANDIDATES, or CANDIDATE")

    arr = np.asarray(value.get("candidates", value.get("candidate")) if isinstance(value, dict) else value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"candidate output must be 1D or 2D, got shape {arr.shape}")
    if arr.shape[1] != int(info["dim"]):
        raise ValueError(f"candidate dimension mismatch: expected {info['dim']}, got {arr.shape[1]}")
    if arr.shape[0] < 1:
        raise ValueError("at least one candidate is required")
    max_candidates = int(info.get("max_candidates_per_submission", 256))
    if arr.shape[0] > max_candidates:
        raise ValueError(f"too many candidates: max {max_candidates}, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("candidate values must be finite")

    result = post_json(
        f"{JUDGE_URL}/evaluate",
        {"submission_uuid": f"final-{uuid.uuid4()}", "candidates": arr.tolist()},
    )
    if result.get("status") != "done":
        raise RuntimeError(str(result.get("message") or result.get("error") or result))
    return result


def write_reward(reward: float, detail: str = "", extra: dict | None = None) -> None:
    REWARD_TXT.parent.mkdir(parents=True, exist_ok=True)
    reward = max(0.0, float(reward))
    REWARD_TXT.write_text(str(reward), encoding="utf-8")
    numeric_payload = {"reward": reward, "score": reward}
    sidecar = {"reward": reward, "detail": detail}
    if extra:
        for key, value in extra.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_payload[key] = value
            sidecar[key] = value
    REWARD_JSON.write_text(json.dumps(numeric_payload, indent=2), encoding="utf-8")
    EVALUATION_JSON.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")


def write_best_submission_reward(best: dict | None, reason: str) -> bool:
    if best is None:
        return False
    reward = float(best.get("reward", 0.0))
    print(f"Using best iterative submission after {reason}: reward={reward:.6f}")
    write_reward(
        reward,
        f"Using best iterative submission after {reason}",
        {
            "hpwl": best.get("hpwl"),
            "overlap_rate": best.get("overlap_rate"),
            "best_submission_reward": reward,
            "used_best_submission": 1,
        },
    )
    return True


def main() -> None:
    copy_agent_submissions_log()
    best = best_submission()

    if not SOLUTION_PATH.exists():
        print("ERROR: /app/solution.py not found")
        if write_best_submission_reward(best, "solution.py not found"):
            return
        write_reward(0.0, "solution.py not found")
        return
    if not SOLUTION_PATH.read_text(encoding="utf-8", errors="replace").strip():
        print("ERROR: /app/solution.py is empty")
        if write_best_submission_reward(best, "solution.py is empty"):
            return
        write_reward(0.0, "solution.py is empty")
        return

    try:
        final_result = evaluate_final_solution()
    except Exception as exc:
        print(f"WARN: final solution evaluation failed: {exc}")
        if write_best_submission_reward(best, "final solution evaluation failed"):
            return
        write_reward(0.0, f"final solution evaluation failed: {exc}")
        return

    final_reward = float(final_result.get("reward", 0.0))
    best_reward = float(best.get("reward", 0.0)) if best is not None else -1.0
    if best is not None and best_reward > final_reward:
        write_best_submission_reward(best, "best iterative submission beats final solution")
        return

    write_reward(
        final_reward,
        "Using final solution evaluation",
        {
            "hpwl": final_result.get("hpwl"),
            "overlap_rate": final_result.get("overlap_rate"),
            "final_solution_reward": final_reward,
            "best_submission_reward": best_reward if best is not None else None,
            "used_best_submission": 0,
        },
    )


if __name__ == "__main__":
    main()
