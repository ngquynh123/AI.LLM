#!/usr/bin/env python3
"""Asynchronous load tester for the AI.LLM retriever API.

The script is designed for quick local benchmarks.  By default it runs in
"dry-run" mode so it can be linted / invoked without hitting the server or
requiring optional dependencies.  Pass ``--no-dry-run`` to actually issue
requests (requires ``aiohttp``).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_URL = "http://localhost:8001/query"


@dataclass(slots=True)
class QuestionBatch:
    index: int
    question: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Asynchronous load generator for the retriever API")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Target endpoint (default: {DEFAULT_URL})")
    parser.add_argument("--questions", default="questions.csv", help="CSV file with a 'question' column")
    parser.add_argument("--total", type=int, default=100, help="Total number of requests to send")
    parser.add_argument("--concurrency", type=int, default=20, help="Maximum in-flight requests")
    parser.add_argument("--session-id", default="load-test", help="Session id attached to each request")
    parser.add_argument("--http-timeout", type=float, default=650.0, help="Per-request timeout (seconds)")
    parser.add_argument("--max-retries", type=int, default=0, help="Automatic retries per request on failure")
    parser.add_argument(
        "--save-responses",
        type=Path,
        default=None,
        help="Optional JSONL file to dump each response payload.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Run input validation only (default)",
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Send actual requests (requires aiohttp)",
    )
    parser.set_defaults(dry_run=True)
    return parser.parse_args(argv)


def load_questions_csv(path: Path, limit: Optional[int] = None) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Question file is empty: {path}")
    header = rows[0]
    data_rows = rows[1:] if header else rows
    if header and len(header) == 1:
        column_index = 0
    elif header and "question" in [h.strip().lower() for h in header]:
        column_index = [h.strip().lower() for h in header].index("question")
    else:
        column_index = 0
    questions: List[str] = []
    for row in data_rows:
        if not row:
            continue
        try:
            raw = row[column_index]
        except IndexError:
            continue
        question = (raw or "").strip()
        if question:
            questions.append(question)
        if limit and len(questions) >= limit:
            break
    if not questions:
        raise ValueError("No questions found in the provided CSV")
    return questions


def build_batches(questions: Sequence[str], total: int) -> List[QuestionBatch]:
    batches: List[QuestionBatch] = []
    if not questions:
        return batches
    for idx in range(total):
        question = questions[idx % len(questions)]
        batches.append(QuestionBatch(index=idx + 1, question=question))
    return batches


async def _send_single(
    session: "aiohttp.ClientSession",
    sem: asyncio.Semaphore,
    batch: QuestionBatch,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    payload = {"question": batch.question, "session_id": args.session_id}
    attempt = 0
    error: Optional[str] = None
    while attempt <= args.max_retries:
        attempt += 1
        async with sem:
            start = time.perf_counter()
            try:
                async with session.post(args.url, json=payload) as resp:
                    text = await resp.text()
                    elapsed = time.perf_counter() - start
                    try:
                        body = json.loads(text)
                    except json.JSONDecodeError:
                        body = text
                    return {
                        "index": batch.index,
                        "status": resp.status,
                        "elapsed_ms": round(elapsed * 1000, 2),
                        "body": body,
                    }
            except Exception as exc:  # pragma: no cover - network dependent
                error = repr(exc)
        if attempt > args.max_retries:
            break
        await asyncio.sleep(0)  # allow event loop to breathe before retry
    return {"index": batch.index, "status": None, "elapsed_ms": None, "error": error}


async def run_load_test(args: argparse.Namespace, batches: Sequence[QuestionBatch]) -> Dict[str, Any]:
    import aiohttp  # imported lazily so dry-run works without dependency

    timeout = aiohttp.ClientTimeout(total=args.http_timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    sem = asyncio.Semaphore(args.concurrency)
    start_time = time.perf_counter()
    results: List[Dict[str, Any]] = []

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [asyncio.create_task(_send_single(session, sem, batch, args)) for batch in batches]
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)

    duration = time.perf_counter() - start_time
    return {"results": results, "duration": duration}


def summarise(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    statuses = Counter()
    elapsed_times: List[float] = []
    busy_count = 0
    errors = 0
    for item in results:
        status = item.get("status")
        if status is None:
            errors += 1
        else:
            statuses[status] += 1
        elapsed = item.get("elapsed_ms")
        if isinstance(elapsed, (int, float)):
            elapsed_times.append(float(elapsed))
        body = item.get("body")
        if isinstance(body, dict):
            answer = body.get("answer") or {}
            if isinstance(answer, dict) and answer.get("type") == "busy":
                busy_count += 1
            elif isinstance(body.get("details"), dict):
                reason = body["details"].get("reason")
                if reason == "llm_queue_timeout":
                    busy_count += 1
    elapsed_stats = {
        "p50_ms": round(_percentile(elapsed_times, 0.50), 2) if elapsed_times else None,
        "p90_ms": round(_percentile(elapsed_times, 0.90), 2) if elapsed_times else None,
        "max_ms": round(max(elapsed_times), 2) if elapsed_times else None,
    }
    return {
        "status_counts": dict(statuses),
        "busy_responses": busy_count,
        "errors": errors,
        "latency": elapsed_stats,
    }


def _percentile(data: Sequence[float], percentile: float) -> float:
    if not data:
        return math.nan
    ordered = sorted(data)
    k = (len(ordered) - 1) * percentile
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    d0 = ordered[int(f)] * (c - k)
    d1 = ordered[int(c)] * (k - f)
    return d0 + d1


def save_jsonl(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    questions = load_questions_csv(Path(args.questions))
    batches = build_batches(questions, args.total)
    if args.dry_run:
        print("Dry-run: no requests will be sent.")
        print(f"Loaded {len(questions)} unique questions from {args.questions}")
        print(f"Prepared {len(batches)} requests across concurrency={args.concurrency}")
        print("Use --no-dry-run to execute and optionally --save-responses output.jsonl to capture payloads.")
        return 0
    outcome = asyncio.run(run_load_test(args, batches))
    results = outcome["results"]
    duration = outcome["duration"]
    summary = summarise(results)
    print(json.dumps({"duration_s": round(duration, 2), **summary}, ensure_ascii=False, indent=2))
    if args.save_responses:
        save_jsonl(Path(args.save_responses), results)
        print(f"Responses written to {args.save_responses}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
