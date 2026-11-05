"""Lightweight CLI wrapper for AI.LLM load tests.

This command lets you trigger the `/load-test/async-run` endpoint without
opening the web UI. It is designed for local runs that bypass Dev Tunnels or
other proxies that might terminate long-running HTTP requests.

Example usage:
    python -m tools.load_test_cli --api-base http://localhost:8001 \
        --target-url http://localhost:8011/query --total 50 --concurrency 10
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any, Dict, Optional

import requests

DEFAULT_API_BASE = "http://localhost:8011"
DEFAULT_TARGET_URL = "http://localhost:8011/query"
DEFAULT_QUESTIONS = "questions.csv"
DEFAULT_SAVE_PATH = "out/load.jsonl"


class CLIError(RuntimeError):
    """Raised when the CLI fails to perform the requested operation."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trigger AI.LLM load test runs without the web UI.",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=(
            "Base URL of the AI.LLM API (default: %(default)s). "
            "The CLI will POST to {base}/load-test/async-run."
        ),
    )
    parser.add_argument(
        "--target-url",
        default=DEFAULT_TARGET_URL,
        help=(
            "Retriever endpoint to stress-test (default: %(default)s). "
            "Should point to the POST /query endpoint exposed by your LLM service."
        ),
    )
    parser.add_argument(
        "--questions",
        default=DEFAULT_QUESTIONS,
        help=(
            "Path to the CSV file containing a 'question' column (default: %(default)s). "
            "Path is resolved relative to the AI.LLM workspace on the server."
        ),
    )
    parser.add_argument(
        "--total",
        type=int,
        default=100,
        help="Total number of requests to send (default: %(default)s).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Maximum concurrent requests (default: %(default)s).",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Override session id. Defaults to load-test-<timestamp> when omitted.",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=650.0,
        help="HTTP timeout per request in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Number of retries for failed requests (default: %(default)s).",
    )
    parser.add_argument(
        "--save-responses",
        default=DEFAULT_SAVE_PATH,
        help="Where to store JSONL responses (default: %(default)s).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist JSONL results on the server.",
    )
    parser.add_argument(
        "--return-results",
        action="store_true",
        help="Return the entire results array in the HTTP response (default: false).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CSV/config without sending traffic.",
    )
    parser.add_argument(
        "--client-timeout",
        type=float,
        default=900.0,
        help="Timeout in seconds for the CLI's HTTP request (default: %(default)s).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the raw JSON response instead of a friendly summary.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification when calling the API base URL.",
    )
    return parser


def _build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    session_id = args.session_id or f"load-test-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    save_responses: Optional[str] = None if args.no_save else args.save_responses
    payload: Dict[str, Any] = {
        "url": args.target_url,
        "questions": args.questions,
        "total": args.total,
        "concurrency": args.concurrency,
        "session_id": session_id,
        "http_timeout": args.http_timeout,
        "max_retries": args.max_retries,
        "dry_run": bool(args.dry_run),
        "save_responses": save_responses,
        "return_results": bool(args.return_results),
    }
    return payload


def _post(url: str, payload: Dict[str, Any], timeout: float, verify: bool) -> Dict[str, Any]:
    try:
        response = requests.post(url, json=payload, timeout=timeout, verify=verify)
    except requests.RequestException as exc:  # pragma: no cover - network call
        raise CLIError(f"Không thể kết nối tới {url}: {exc}") from exc

    if response.status_code != 200:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise CLIError(
            f"Load test thất bại với HTTP {response.status_code}: {detail}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise CLIError("Phản hồi không phải JSON hợp lệ") from exc


def _print_summary(data: Dict[str, Any]) -> None:
    status = data.get("status")
    config = data.get("config", {})
    summary = data.get("summary")
    results_path = data.get("results_path")

    print("=== Load test result ===")
    print(f"Trạng thái       : {status}")
    if data.get("duration_s") is not None:
        print(f"Thời gian        : {data['duration_s']} giây")
    print(f"Tổng request     : {config.get('total')} (batch = {data.get('batches')})")
    print(f"Concurrency      : {config.get('concurrency')}")
    print(f"Session ID       : {config.get('session_id')}")
    print(f"Questions file   : {config.get('questions_file')}")
    print(f"Save responses   : {results_path or 'Không lưu'}")

    if summary:
        latency = summary.get("latency") or {}
        print("--- Thống kê ---")
        print(f"HTTP 200/err     : {summary.get('status_counts')}")
        print(f"Busy responses   : {summary.get('busy_responses')}")
        print(f"Errors           : {summary.get('errors')}")
        if latency:
            print(
                "Latency (ms)    : p50={p50} · p90={p90} · max={mx}".format(
                    p50=latency.get("p50_ms"),
                    p90=latency.get("p90_ms"),
                    mx=latency.get("max_ms"),
                )
            )

    results = data.get("results")
    if results is not None:
        print("--- Ảnh mẫu kết quả ---")
        preview = results[:3]
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        if len(results) > len(preview):
            print(f"(Đã cắt bớt {len(results) - len(preview)} record)")

    if status == "dry-run":
        print("Dry-run thành công. Không có request nào được gửi.")


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    payload = _build_payload(args)
    api_endpoint = args.api_base.rstrip("/") + "/load-test/async-run"
    verify_tls = not args.insecure

    try:
        data = _post(api_endpoint, payload, timeout=args.client_timeout, verify=verify_tls)
    except CLIError as exc:
        parser.exit(status=1, message=f"Lỗi: {exc}\n")

    if args.raw:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        _print_summary(data)

    if not payload["return_results"] and not payload["dry_run"]:
        results_path = data.get("results_path")
        if results_path:
            print(
                "Ghi chú: Response đã trả summary. Dữ liệu đầy đủ nằm ở file JSONL "
                f"{results_path}. Dùng /load-results để xem chi tiết."
            )
        else:
            print(
                "Ghi chú: return_results=false và không lưu file, không có record để tải lại."
            )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
