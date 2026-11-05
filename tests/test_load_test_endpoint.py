import json
import sys
import types
from pathlib import Path

from starlette.testclient import TestClient

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) in sys.path:
    sys.path.remove(str(TESTS_DIR))
PACKAGE_ROOT = TESTS_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

if "sentence_transformers" not in sys.modules:
    fake_st = types.ModuleType("sentence_transformers")

    class _DummySentenceTransformer:  # pragma: no cover - test stub
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, *args, **kwargs):
            return []

    fake_st.SentenceTransformer = _DummySentenceTransformer
    sys.modules["sentence_transformers"] = fake_st

if "requests" not in sys.modules:
    fake_requests = types.ModuleType("requests")

    class _DummyResponse:  # pragma: no cover - test stub
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {}

    class _DummyRequestException(Exception):
        pass

    def _dummy_post(*args, **kwargs):
        return _DummyResponse()

    fake_requests.post = _dummy_post
    fake_requests.exceptions = types.SimpleNamespace(RequestException=_DummyRequestException)
    sys.modules["requests"] = fake_requests

if "langchain_core.prompts" not in sys.modules:
    fake_langchain = types.ModuleType("langchain_core")
    fake_prompts = types.ModuleType("langchain_core.prompts")

    class _DummyChatPromptTemplate:  # pragma: no cover - test stub
        @classmethod
        def from_messages(cls, messages):  # noqa: D401 - simple stub
            return cls()

    class _DummyMessagesPlaceholder:  # pragma: no cover - test stub
        def __init__(self, *args, **kwargs):
            pass

    fake_prompts.ChatPromptTemplate = _DummyChatPromptTemplate
    fake_prompts.MessagesPlaceholder = _DummyMessagesPlaceholder
    fake_runnables = types.ModuleType("langchain_core.runnables")
    fake_history = types.ModuleType("langchain_core.runnables.history")

    class _DummyRunnableWithMessageHistory:  # pragma: no cover - test stub
        def __init__(self, *args, **kwargs):
            pass

        def with_types(self, *args, **kwargs):
            return self

    fake_history.RunnableWithMessageHistory = _DummyRunnableWithMessageHistory
    fake_runnables.history = fake_history
    fake_langchain.prompts = fake_prompts
    fake_langchain.runnables = fake_runnables
    fake_chat_history = types.ModuleType("langchain_core.chat_history")

    class _DummyBaseChatMessageHistory:  # pragma: no cover - test stub
        def __init__(self, *args, **kwargs):
            pass

        def add_message(self, *args, **kwargs):
            return None

    fake_chat_history.BaseChatMessageHistory = _DummyBaseChatMessageHistory
    fake_langchain.chat_history = fake_chat_history
    sys.modules["langchain_core"] = fake_langchain
    sys.modules["langchain_core.prompts"] = fake_prompts
    sys.modules["langchain_core.runnables"] = fake_runnables
    sys.modules["langchain_core.runnables.history"] = fake_history
    sys.modules["langchain_core.chat_history"] = fake_chat_history

if "langchain_community.chat_models" not in sys.modules:
    fake_langchain_community = types.ModuleType("langchain_community")
    fake_chat_models = types.ModuleType("langchain_community.chat_models")

    class _DummyChatOllama:  # pragma: no cover - test stub
        def __init__(self, *args, **kwargs):
            pass

        def bind(self, *args, **kwargs):
            return self

        def with_structured_output(self, *args, **kwargs):
            return self

    fake_chat_models.ChatOllama = _DummyChatOllama
    fake_chat_histories = types.ModuleType("langchain_community.chat_message_histories")

    class _DummyChatMessageHistory:  # pragma: no cover - test stub
        def __init__(self, *args, **kwargs):
            self.messages = []

        def add_message(self, message):
            self.messages.append(message)

        def clear(self):
            self.messages.clear()

    fake_chat_histories.ChatMessageHistory = _DummyChatMessageHistory
    fake_langchain_community.chat_models = fake_chat_models
    fake_langchain_community.chat_message_histories = fake_chat_histories
    sys.modules["langchain_community"] = fake_langchain_community
    sys.modules["langchain_community.chat_models"] = fake_chat_models
    sys.modules["langchain_community.chat_message_histories"] = fake_chat_histories

from retriever import api_server


def _suppress_startup():
    startup_backup = list(api_server.app.router.on_startup)
    shutdown_backup = list(api_server.app.router.on_shutdown)
    api_server.app.router.on_startup.clear()
    api_server.app.router.on_shutdown.clear()
    return startup_backup, shutdown_backup


def _restore_startup(startup_backup, shutdown_backup):
    api_server.app.router.on_startup[:] = startup_backup
    api_server.app.router.on_shutdown[:] = shutdown_backup


def _write_questions(filename: str, questions):
    target = Path(api_server.BASE_DIR) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write("question\n")
        for q in questions:
            handle.write(q + "\n")
    return target


def test_load_test_endpoint_dry_run():
    questions_file = _write_questions("out/test_async_questions.csv", ["Câu hỏi 1", "Câu hỏi 2"])

    startup_backup, shutdown_backup = _suppress_startup()
    try:
        with TestClient(api_server.app) as client:
            response = client.post(
                "/load-test/async-run",
                json={
                    "questions": "out/test_async_questions.csv",
                    "dry_run": True,
                    "return_results": True,
                    "total": 5,
                    "concurrency": 2,
                },
            )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "dry-run"
        assert payload["batches"] == 5
        assert payload["config"]["questions_count"] == 2
        assert payload["results"] is None
    finally:
        _restore_startup(startup_backup, shutdown_backup)
        questions_file.unlink(missing_ok=True)


def test_load_test_endpoint_executes(monkeypatch):
    questions_file = _write_questions("out/test_async_questions_exec.csv", ["Q1"])
    results_path = Path("out/test_async_results.jsonl")

    async def fake_run_load_test(args, batches):
        assert args.total == 3
        return {
            "results": [
                {"index": 1, "status": 200, "elapsed_ms": 123.0, "body": {"answer": {"type": "semantic"}}}
            ],
            "duration": 1.234,
        }

    saved = {}

    def fake_save_jsonl(path, results):
        saved["path"] = Path(path)
        saved["results"] = results

    monkeypatch.setattr(api_server.async_batch_runner, "run_load_test", fake_run_load_test)
    monkeypatch.setattr(api_server.async_batch_runner, "save_jsonl", fake_save_jsonl)

    startup_backup, shutdown_backup = _suppress_startup()
    try:
        with TestClient(api_server.app) as client:
            response = client.post(
                "/load-test/async-run",
                json={
                    "questions": "out/test_async_questions_exec.csv",
                    "total": 3,
                    "concurrency": 1,
                    "save_responses": str(results_path),
                    "return_results": True,
                },
            )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "completed"
        assert payload["duration_s"] == 1.23
        assert payload["summary"]["status_counts"]["200"] == 1
        assert payload["results"][0]["index"] == 1
        assert saved["path"].as_posix().endswith(results_path.as_posix())
        assert saved["results"][0]["status"] == 200
    finally:
        _restore_startup(startup_backup, shutdown_backup)
        questions_file.unlink(missing_ok=True)
        (Path(api_server.BASE_DIR) / results_path).unlink(missing_ok=True)
