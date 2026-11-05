import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SITE_PACKAGES = PROJECT_ROOT / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if SITE_PACKAGES.exists():
    sys.path.insert(0, str(SITE_PACKAGES))

if "fastapi" in sys.modules:
    module = sys.modules["fastapi"]
    module_file = getattr(module, "__file__", "")
    if module_file and "tests/fastapi.py" in module_file:
        sys.modules.pop("fastapi")

from starlette.testclient import TestClient

from retriever import api_server


def _prepare_test_jsonl(filename: str, entries):
    target_dir = Path(api_server.BASE_DIR) / "out"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def test_load_results_endpoint_returns_paginated_data():
    entries = [
        {"index": 1, "status": 200, "elapsed_ms": 120.0, "body": {"answer": {"type": "semantic"}}},
        {"index": 2, "status": 200, "elapsed_ms": 240.0, "body": {"answer": {"type": "busy"}}},
        {"index": 3, "status": None, "elapsed_ms": None, "error": "timeout"},
    ]
    file_path = _prepare_test_jsonl("test_load_results.jsonl", entries)

    startup_backup = list(api_server.app.router.on_startup)
    shutdown_backup = list(api_server.app.router.on_shutdown)
    api_server.app.router.on_startup.clear()
    api_server.app.router.on_shutdown.clear()

    try:
        with TestClient(api_server.app) as client:
            response = client.get(
                "/load-results",
                params={
                    "path": "out/" + file_path.name,
                    "offset": 1,
                    "limit": 1,
                    "include_summary": True,
                },
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == len(entries)
        assert payload["offset"] == 1
        assert payload["limit"] == 1
        assert payload["count"] == 1
        assert payload["items"][0]["index"] == 2
        assert payload["summary"]["busy_responses"] == 1
        assert payload["summary"]["errors"] >= 0
        status_counts = payload["summary"]["status_counts"]
        assert status_counts.get("200") == 2
    finally:
        api_server.app.router.on_startup[:] = startup_backup
        api_server.app.router.on_shutdown[:] = shutdown_backup
        if file_path.exists():
            file_path.unlink()
