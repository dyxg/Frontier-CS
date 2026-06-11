#!/usr/bin/env python3
"""
Run Harbor trials for algorithmic problems in parallel.

Examples:
    uv run python scripts/run_algorithmic_harbor_trials.py -m gpt-5.5 -j 72
    uv run python scripts/run_algorithmic_harbor_trials.py --problems 0 1 2 -j 8 --agent-timeout 180
    uv run python scripts/run_algorithmic_harbor_trials.py -j 72 -- --force-build --env FOO=bar
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBLEMS_DIR = REPO_ROOT / "algorithmic" / "problems"


@dataclass
class TrialResult:
    problem_id: str
    command: list[str]
    returncode: int
    success: bool
    attempts: int
    started_at: str
    finished_at: str
    duration_seconds: float
    stdout_log: str
    stderr_log: str
    summary: dict[str, Any] | None


def natural_key(value: str) -> tuple[int, int | str]:
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def list_algorithmic_problems() -> list[str]:
    if not PROBLEMS_DIR.is_dir():
        raise SystemExit(f"Problem directory not found: {PROBLEMS_DIR}")

    problem_ids = [
        path.name
        for path in PROBLEMS_DIR.iterdir()
        if path.is_dir() and (path / "config.yaml").exists()
    ]
    return sorted(problem_ids, key=natural_key)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_name(problem_id: str) -> str:
    return problem_id.replace("/", "_").replace(" ", "_")


def parse_json_from_stdout(stdout_path: Path) -> dict[str, Any] | None:
    if not stdout_path.exists():
        return None

    text = stdout_path.read_text(errors="replace").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)

    return objects[-1] if objects else None


def build_trial_command(
    frontier_command: list[str],
    problem_id: str,
    agent: str,
    model: str | None,
    emit_json: bool,
    extra_args: list[str],
) -> list[str]:
    command = [
        *frontier_command,
        "harbor",
        "trial",
        "algorithmic",
        problem_id,
        "-a",
        agent,
    ]
    if model:
        command.extend(["-m", model])
    if emit_json and "--json" not in extra_args:
        command.append("--json")
    command.extend(extra_args)
    return command


def run_one_trial(
    problem_id: str,
    command: list[str],
    log_dir: Path,
    retries: int,
    dry_run: bool,
) -> TrialResult:
    problem_log_dir = log_dir / safe_name(problem_id)
    problem_log_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    started_at = utc_now()
    returncode = 0
    stdout_log = problem_log_dir / "stdout.log"
    stderr_log = problem_log_dir / "stderr.log"

    if dry_run:
        finished_at = utc_now()
        return TrialResult(
            problem_id=problem_id,
            command=command,
            returncode=0,
            success=True,
            attempts=0,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=0.0,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            summary=None,
        )

    attempts = 0
    for attempt in range(1, retries + 2):
        attempts = attempt
        stdout_log = problem_log_dir / (
            "stdout.log" if attempt == 1 else f"stdout.attempt_{attempt}.log"
        )
        stderr_log = problem_log_dir / (
            "stderr.log" if attempt == 1 else f"stderr.attempt_{attempt}.log"
        )
        with stdout_log.open("w") as stdout_file, stderr_log.open("w") as stderr_file:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                check=False,
            )
        returncode = completed.returncode
        if returncode == 0:
            break

    finished_at = utc_now()
    duration = time.monotonic() - started
    summary = parse_json_from_stdout(stdout_log)
    return TrialResult(
        problem_id=problem_id,
        command=command,
        returncode=returncode,
        success=returncode == 0,
        attempts=attempts,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=round(duration, 3),
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        summary=summary,
    )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run `frontier harbor trial algorithmic <id>` for many problems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Any unknown arguments are forwarded to `frontier harbor trial`. "
            "Use `--` before forwarded arguments when a flag name is ambiguous."
        ),
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Maximum number of concurrent Harbor trials.",
    )
    parser.add_argument(
        "-a",
        "--agent",
        default="codex",
        help="Harbor agent name passed to `frontier harbor trial`.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gpt-5.5",
        help="Model name passed to `frontier harbor trial`.",
    )
    parser.add_argument(
        "--frontier-command",
        default="uv run frontier",
        help="Command prefix used to invoke the Frontier-CS CLI.",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        help="Problem IDs to run. Defaults to every algorithmic problem.",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="Problem IDs to skip after selection.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for per-problem logs and the JSONL summary.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="JSONL summary path. Defaults to <log-dir>/results.jsonl.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Retry count for failed trials.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Cancel queued work after the first failed trial.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not add --json to the trial command automatically.",
    )

    args, extra_args = parser.parse_known_args()
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    return args, extra_args


def main() -> int:
    args, extra_args = parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.retries < 0:
        raise SystemExit("--retries must be non-negative")

    selected = args.problems if args.problems else list_algorithmic_problems()
    skip = set(args.skip)
    problem_ids = [problem_id for problem_id in selected if problem_id not in skip]
    if not problem_ids:
        raise SystemExit("No algorithmic problems selected.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir or (
        REPO_ROOT / ".frontier-cs" / "harbor" / f"algorithmic_batch_{timestamp}"
    )
    summary_file = args.summary_file or (log_dir / "results.jsonl")
    frontier_command = shlex.split(args.frontier_command)

    commands = {
        problem_id: build_trial_command(
            frontier_command=frontier_command,
            problem_id=problem_id,
            agent=args.agent,
            model=args.model,
            emit_json=not args.no_json,
            extra_args=extra_args,
        )
        for problem_id in problem_ids
    }

    print(f"Problems: {len(problem_ids)}")
    print(f"Workers: {args.workers}")
    if extra_args:
        print(f"Extra trial args: {shlex.join(extra_args)}")

    if args.dry_run:
        print("Dry run: no commands will be executed and no logs will be written.")
        for problem_id in problem_ids:
            print(shlex.join(commands[problem_id]))
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Logs: {log_dir}")
    print(f"Summary: {summary_file}")

    completed_count = 0
    failed_count = 0
    started_at = time.monotonic()

    with summary_file.open("w") as summary, concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {
            executor.submit(
                run_one_trial,
                problem_id,
                commands[problem_id],
                log_dir,
                args.retries,
                args.dry_run,
            ): problem_id
            for problem_id in problem_ids
        }

        for future in concurrent.futures.as_completed(futures):
            problem_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - keep batch runner alive.
                failed_count += 1
                completed_count += 1
                record = {
                    "problem_id": problem_id,
                    "success": False,
                    "error": repr(exc),
                }
                summary.write(json.dumps(record, sort_keys=True) + "\n")
                summary.flush()
                print(f"[{completed_count}/{len(problem_ids)}] FAIL {problem_id}: {exc}")
            else:
                completed_count += 1
                if not result.success:
                    failed_count += 1
                summary.write(json.dumps(asdict(result), sort_keys=True) + "\n")
                summary.flush()
                status = "OK" if result.success else f"FAIL rc={result.returncode}"
                reward = ""
                if result.summary and "reward" in result.summary:
                    reward = f" reward={result.summary['reward']}"
                print(
                    f"[{completed_count}/{len(problem_ids)}] {status} "
                    f"{problem_id} ({result.duration_seconds:.1f}s){reward}"
                )

            if args.stop_on_failure and failed_count:
                for pending in futures:
                    pending.cancel()
                break

    elapsed = time.monotonic() - started_at
    print(
        f"Done: {completed_count} completed, {failed_count} failed, "
        f"{elapsed:.1f}s elapsed."
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    sys.exit(main())
