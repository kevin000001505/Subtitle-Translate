import asyncio
import re

import pytest

import main
import translate_gemma

SAMPLE_SRT = """1
00:00:01,000 --> 00:00:02,000
Hello

2
00:00:03,000 --> 00:00:04,000
Goodbye
"""

_LINE_RE = re.compile(r"^\s*\[(\d+)\] (.*)$", re.MULTILINE)


class FakeResponse:
    def __init__(self, status_code, content=""):
        self.status_code = status_code
        self.text = content
        self._content = content

    def json(self):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "test",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": self._content},
            }],
        }


class FakeLlamaSwap:
    """Stands in for llama-swap's /v1/chat/completions over HTTP.

    Answers each `[ID] text` input line with `[ID] <target_lang_code>: text`,
    so a test can see which language the request asked for."""

    def __init__(self, status_code=200):
        self.status_code = status_code
        self.requests = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.requests.append({"url": url, "json": json})
        if self.status_code != 200:
            return FakeResponse(self.status_code, "server error")
        item = json["messages"][0]["content"][0]
        body = item["text"].split("INPUT:", 1)[1]
        target = item["target_lang_code"]
        reply = "\n".join(f"[{i}] {target}: {t}" for i, t in _LINE_RE.findall(body))
        return FakeResponse(200, reply)


class FakeGemini:
    """Stands in for google-genai's async client: client.aio.models.generate_content."""

    def __init__(self):
        self.prompts = []
        self.models_used = []
        self.aio = self
        self.models = self

    async def generate_content(self, model, contents):
        self.prompts.append(contents)
        self.models_used.append(model)
        body = contents.split("INPUT:", 1)[1]
        reply = "\n".join(f"[{i}] gemini: {t}" for i, t in _LINE_RE.findall(body))

        class _Resp:
            text = reply
        return _Resp()


@pytest.fixture(autouse=True)
def isolated_app(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DB_FILE", str(tmp_path / "queue.db"))
    main.init_db()

    async def no_sleep(_seconds):
        return None
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    yield


@pytest.fixture
def llama(monkeypatch):
    fake = FakeLlamaSwap()
    monkeypatch.setattr(translate_gemma.requests, "post", fake.post)
    return fake


@pytest.fixture
def gemini(monkeypatch):
    fake = FakeGemini()
    monkeypatch.setattr(main, "client", fake)
    return fake


@pytest.fixture
def media(tmp_path):
    folder = tmp_path / "media" / "Movie (2020)"
    folder.mkdir(parents=True)
    return folder
