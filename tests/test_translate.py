import asyncio

import pysubs2
from fastapi.testclient import TestClient

import main
from conftest import SAMPLE_SRT


def translate(path, **fields):
    client = TestClient(main.app)
    return client.post("/translate", json={"path": str(path), **fields})


def lines(path):
    return [(s.start, s.end, s.text) for s in pysubs2.load(str(path))]


def test_translating_a_folder_writes_zh_tw_next_to_the_english_subtitle(media, llama):
    (media / "Movie.en.srt").write_text(SAMPLE_SRT)

    translate(media)

    assert lines(media / "Movie.zh-TW.srt") == [
        (500, 1500, "zh-TW: Hello"),
        (2500, 3500, "zh-TW: Goodbye"),
    ]


def test_translating_uses_the_translategemma_model_from_env_by_default(media, llama):
    (media / "Movie.en.srt").write_text(SAMPLE_SRT)

    translate(media)

    assert [r["json"]["model"] for r in llama.requests] == ["translatemgemma"]
    assert llama.requests[0]["url"] == "http://localhost:8080/v1/chat/completions"


def test_translating_a_single_english_subtitle_file(media, llama):
    (media / "Movie.en.srt").write_text(SAMPLE_SRT)

    translate(media / "Movie.en.srt")

    assert lines(media / "Movie.zh-TW.srt")[0] == (500, 1500, "zh-TW: Hello")


def test_an_existing_translation_is_skipped(media, llama):
    (media / "Movie.en.srt").write_text(SAMPLE_SRT)
    (media / "Movie.zh-TW.srt").write_text("keep me")

    translate(media)

    assert llama.requests == []
    assert (media / "Movie.zh-TW.srt").read_text() == "keep me"


def test_only_untranslated_episodes_of_a_series_are_translated(media, llama):
    (media / "S01E01.en.srt").write_text(SAMPLE_SRT)
    (media / "S01E01.zh-TW.srt").write_text("keep me")
    (media / "S01E02.en.srt").write_text(SAMPLE_SRT)

    translate(media)

    assert (media / "S01E01.zh-TW.srt").read_text() == "keep me"
    assert lines(media / "S01E02.zh-TW.srt")[1] == (2500, 3500, "zh-TW: Goodbye")


def test_a_failing_local_model_falls_back_to_gemini_by_default(media, llama, gemini):
    llama.status_code = 500
    (media / "Movie.en.srt").write_text(SAMPLE_SRT)

    translate(media)

    assert lines(media / "Movie.zh-TW.srt")[0] == (500, 1500, "gemini: Hello")
    assert "Translate to zh-TW" in gemini.prompts[0]
    assert gemini.models_used == ["gemini-2.5-flash-lite"]


def run_process_movie(path, **kwargs):
    asyncio.run(main.process_movie(str(path), **kwargs))


def test_translating_to_another_language_writes_that_language_next_to_the_video(media, llama):
    (media / "Movie.en.srt").write_text(SAMPLE_SRT)

    run_process_movie(media, target_lang="ja")

    assert (media / "Movie.ja.srt").exists()
    assert lines(media / "Movie.ja.srt") == [
        (500, 1500, "ja: Hello"),
        (2500, 3500, "ja: Goodbye"),
    ]
    assert not (media / "Movie.zh-TW.srt").exists()


def test_an_existing_translation_in_another_language_does_not_block_translating(media, llama):
    (media / "Movie.en.srt").write_text(SAMPLE_SRT)
    (media / "Movie.zh-TW.srt").write_text("keep me")

    run_process_movie(media, target_lang="ja")

    assert (media / "Movie.ja.srt").exists()
    assert (media / "Movie.zh-TW.srt").read_text() == "keep me"


def test_an_existing_translation_in_the_requested_language_is_skipped(media, llama):
    (media / "Movie.en.srt").write_text(SAMPLE_SRT)
    (media / "Movie.ja.srt").write_text("keep me")

    run_process_movie(media, target_lang="ja")

    assert llama.requests == []
    assert (media / "Movie.ja.srt").read_text() == "keep me"


def test_translating_a_single_english_subtitle_file_to_another_language(media, llama):
    (media / "Movie.en.srt").write_text(SAMPLE_SRT)

    run_process_movie(media / "Movie.en.srt", target_lang="ja")

    assert (media / "Movie.ja.srt").exists()
    assert lines(media / "Movie.ja.srt")[0] == (500, 1500, "ja: Hello")


def test_a_video_without_an_english_srt_is_extracted_and_translated_to_the_requested_language(media, llama, make_mkv):
    make_mkv(media / "S01E01.mkv")
    (media / "S01E01.zh-TW.srt").write_text("keep me")

    run_process_movie(media, target_lang="ja")

    assert (media / "S01E01.ja.srt").exists()
    assert lines(media / "S01E01.ja.srt")[0] == (500, 1500, "ja: Hello")


def test_a_chosen_gemini_model_is_asked_for_the_requested_language(media, llama, gemini):
    (media / "Movie.en.srt").write_text(SAMPLE_SRT)

    run_process_movie(media, target_lang="ja",
                      translator=main.Translator(backend="gemini", model="gemini-2.5-pro"))

    assert gemini.models_used == ["gemini-2.5-pro"]
    assert "Translate to ja." in gemini.prompts[0]
    assert llama.requests == []
    assert lines(media / "Movie.ja.srt")[0] == (500, 1500, "gemini: Hello")


def test_a_single_subtitle_file_already_translated_to_the_requested_language_is_skipped(media, llama):
    (media / "Movie.en.srt").write_text(SAMPLE_SRT)
    (media / "Movie.ja.srt").write_text("keep me")

    run_process_movie(media / "Movie.en.srt", target_lang="ja")

    assert llama.requests == []
    assert (media / "Movie.ja.srt").read_text() == "keep me"


def test_the_translate_endpoint_accepts_a_target_language(media, llama):
    (media / "Movie.en.srt").write_text(SAMPLE_SRT)

    translate(media, target_lang="ja")

    assert (media / "Movie.ja.srt").exists()


def test_the_translate_endpoint_rejects_a_language_code_that_is_not_a_language(media, llama):
    (media / "Movie.en.srt").write_text(SAMPLE_SRT)

    response = translate(media, target_lang="zh TW")

    assert response.status_code == 422
    assert llama.requests == []
