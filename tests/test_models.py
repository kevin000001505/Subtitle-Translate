import asyncio

import pytest
import requests

import translate_gemma
from conftest import SAMPLE_SRT
from test_translate import lines, run_process_movie
import main


def test_a_non_translategemma_model_gets_a_chat_prompt_naming_the_language(media, llama):
    (media / "Movie.en.srt").write_text(SAMPLE_SRT)

    run_process_movie(media, target_lang="ja", translator=main.Translator("llama-swap", "qwen3-14b"))

    request = llama.requests[0]["json"]
    assert request["model"] == "qwen3-14b"
    assert "Japanese (ja)" in request["messages"][0]["content"]
    assert request["messages"][1]["content"].startswith("INPUT:\n[0] Hello")
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert lines(media / "Movie.ja.srt")[0] == (500, 1500, "ja: Hello")


def test_a_translategemma_model_gets_the_structured_request(media, llama):
    (media / "Movie.en.srt").write_text(SAMPLE_SRT)

    run_process_movie(media, target_lang="ja", translator=main.Translator("llama-swap", "TranslateGemma-4B"))

    content = llama.requests[0]["json"]["messages"][0]["content"]
    assert content[0]["target_lang_code"] == "ja" and content[0]["source_lang_code"] == "en"


def test_a_reasoning_models_think_block_is_dropped(media, llama):
    llama.prefix = "<think>[0] not this\n[1] nor this</think>\n"
    (media / "Movie.en.srt").write_text(SAMPLE_SRT)

    run_process_movie(media, translator=main.Translator("llama-swap", "qwen3-14b"))

    assert [t for _, _, t in lines(media / "Movie.zh-TW.srt")] == ["zh-TW: Hello", "zh-TW: Goodbye"]


class FakeModelsHttp:
    def __init__(self, body=None, error=None):
        self.body, self.error, self.calls = body, error, []

    def get(self, url, headers=None, timeout=None):
        self.calls.append({"url": url, "headers": headers})
        if self.error:
            raise self.error

        class _Resp:
            def raise_for_status(_self):
                pass

            def json(_self):
                return self.body
        return _Resp()


LLAMA_SWAP_MODELS = {"object": "list", "data": [
    {"id": "qwen3-14b", "object": "model", "owned_by": "llama-swap", "status": {"value": "unloaded"}},
    {"id": "translategemma-4b", "object": "model", "name": "TranslateGemma 4B",
     "status": {"value": "loaded"}, "meta": {"llamaswap": {"type": "model", "aliases": ["tg"]}}},
    {"id": "tg", "object": "model", "status": {"value": "loaded"},
     "meta": {"llamaswap": {"type": "alias", "modelID": "translategemma-4b"}}},
]}


def test_models_are_listed_from_llama_swap_without_aliases():
    http = FakeModelsHttp(LLAMA_SWAP_MODELS)

    models = translate_gemma.list_models(http)

    assert http.calls[0]["url"] == "http://localhost:8080/v1/models"
    assert http.calls[0]["headers"]["Authorization"] == "Bearer random_key"
    assert models == [
        {"id": "qwen3-14b", "name": "qwen3-14b", "loaded": False},
        {"id": "translategemma-4b", "name": "TranslateGemma 4B", "loaded": True},
    ]


def test_an_unreachable_llama_swap_raises():
    with pytest.raises(requests.RequestException):
        translate_gemma.list_models(FakeModelsHttp(error=requests.ConnectionError("down")))
