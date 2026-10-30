import asyncio
import subprocess

import pytest
from fastapi.testclient import TestClient

import jobs
import main
from conftest import SAMPLE_SRT, FakeJellyfinHttp, jellyfin_routes
from media_servers import Jellyfin, PathMap


@pytest.fixture
def library(tmp_path, monkeypatch, llama):
    root = tmp_path / "lib"
    movie = root / "Movies" / "Arrival (2016)"
    show = root / "TV" / "Severance"
    movie.mkdir(parents=True)
    show.mkdir(parents=True)
    for video in (movie / "Arrival.mkv", show / "S01E01.mkv", show / "S01E02.mkv"):
        video.write_bytes(b"")
        video.with_suffix(".en.srt").write_text(SAMPLE_SRT)
    http = FakeJellyfinHttp(jellyfin_routes())
    server = Jellyfin("http://jf:8096", "secret-key", PathMap(f"/data/media:{root}"), http)
    monkeypatch.setattr(main, "SERVERS", {"jellyfin": server})
    monkeypatch.setattr(main, "_library_cache", {})

    class Library:
        pass
    lib = Library()
    lib.root, lib.movie, lib.show, lib.http, lib.client = root, movie, show, http, TestClient(main.app)
    return lib


def statuses(lib, lang="zh-TW"):
    titles = lib.client.get(f"/library/jellyfin?lang={lang}").json()["titles"]
    return {t["name"]: (t["status"], t.get("done"), t.get("total")) for t in titles}


def queue(lib, **body):
    return lib.client.post("/jobs", json={"server": "jellyfin", **body}).json()


def run_all():
    while asyncio.run(main.run_next_job()):
        pass


def test_the_library_lists_posters_and_subtitle_status(library):
    titles = library.client.get("/library/jellyfin").json()["titles"]

    assert [(t["name"], t["kind"], t["year"], t["poster"]) for t in titles] == [
        ("Arrival", "movie", 2016, "/poster/jellyfin/m1"),
        ("Elsewhere", "movie", 2020, "/poster/jellyfin/m2"),
        ("Severance", "show", 2022, "/poster/jellyfin/s1"),
    ]
    assert statuses(library) == {"Arrival": ("missing", None, None),
                                 "Elsewhere": ("unavailable", None, None),
                                 "Severance": ("missing", 0, 2)}


def test_badges_follow_the_files_and_the_language(library):
    (library.movie / "Arrival.zh-TW.srt").write_text(SAMPLE_SRT)
    (library.show / "S01E01.zh-TW.srt").write_text(SAMPLE_SRT)

    assert statuses(library)["Arrival"][0] == "done"
    assert statuses(library)["Severance"] == ("partial", 1, 2)
    assert statuses(library, "ja")["Arrival"][0] == "missing"


def test_posters_come_through_the_backend_without_the_key(library):
    response = library.client.get("/poster/jellyfin/m1")

    assert response.content == b"poster-bytes"
    assert "secret-key" not in library.client.get("/library/jellyfin").text


def test_an_unconfigured_server_is_reported(library):
    assert library.client.get("/library/plex").status_code == 404


def test_queued_movie_is_translated_and_refreshed_in_the_server(library):
    assert queue(library, items=[{"kind": "movie", "id": "m1"}])["created"] == 1
    assert statuses(library)["Arrival"][0] == "running"

    run_all()

    assert (library.movie / "Arrival.zh-TW.srt").exists()
    job = library.client.get("/jobs").json()["jobs"][0]
    assert (job["status"], job["progress"], job["title"]) == ("done", 100, "Arrival")
    assert ("POST", "/Items/m1/Refresh") in [(c["method"], c["path"]) for c in library.http.calls]


def test_the_same_movie_is_not_queued_twice(library):
    queue(library, items=[{"kind": "movie", "id": "m1"}])

    assert queue(library, items=[{"kind": "movie", "id": "m1"}]) == \
        {"created": 0, "already_queued": 1, "unavailable": []}


def test_a_show_expands_to_one_job_per_episode_with_a_file(library):
    result = queue(library, lang="ja", items=[{"kind": "show", "id": "s1"}])

    assert result == {"created": 2, "already_queued": 0, "unavailable": ["Severance S02E01"]}
    run_all()
    assert (library.show / "S01E01.ja.srt").exists() and (library.show / "S01E02.ja.srt").exists()


def test_a_single_episode_can_be_queued(library):
    queue(library, items=[{"kind": "episode", "id": "e2", "show_id": "s1"}])
    run_all()

    assert [p.name for p in sorted(library.show.glob("*.zh-TW.srt"))] == ["S01E02.zh-TW.srt"]


def test_an_existing_subtitle_is_skipped_unless_overwrite(library, llama):
    (library.movie / "Arrival.zh-TW.srt").write_text("keep me")
    queue(library, items=[{"kind": "movie", "id": "m1"}])
    run_all()
    assert (library.movie / "Arrival.zh-TW.srt").read_text() == "keep me"
    assert library.client.get("/jobs").json()["jobs"][0]["status"] == "skipped"

    queue(library, overwrite=True, items=[{"kind": "movie", "id": "m1"}])
    run_all()
    assert "zh-TW: Hello" in (library.movie / "Arrival.zh-TW.srt").read_text()


def test_the_chosen_model_translates_with_no_gemini_fallback(library, llama, gemini):
    llama.status_code = 500
    queue(library, model="qwen3-14b", items=[{"kind": "movie", "id": "m1"}])
    run_all()

    job = library.client.get("/jobs").json()["jobs"][0]
    assert job["status"] == "failed" and "qwen3-14b" in job["message"]
    assert {r["json"]["model"] for r in llama.requests} == {"qwen3-14b"}
    assert gemini.prompts == []


def test_a_failed_job_does_not_stop_the_next(library):
    (library.movie / "Arrival.mkv").unlink()
    queue(library, items=[{"kind": "movie", "id": "m1"}, {"kind": "episode", "id": "e1", "show_id": "s1"}])
    run_all()

    assert [j["status"] for j in library.client.get("/jobs").json()["jobs"]] == ["done", "failed"]


def test_gemini_jobs_wait_while_the_daily_limit_is_used(library, gemini, monkeypatch):
    monkeypatch.setattr(main, "get_calls_last_24h", lambda: main.DAILY_LIMIT)
    queue(library, backend="gemini", model="gemini-2.5-flash-lite", items=[{"kind": "movie", "id": "m1"}])
    queue(library, items=[{"kind": "episode", "id": "e1", "show_id": "s1"}])

    run_all()

    assert {j["title"]: j["status"] for j in library.client.get("/jobs").json()["jobs"]} == {
        "Arrival": "queued", "Severance S01E01": "done"}


def test_a_job_running_at_restart_is_marked_failed(library):
    queue(library, items=[{"kind": "movie", "id": "m1"}])
    jobs.claim_next(main.DB_FILE)

    jobs.init(main.DB_FILE)

    job = library.client.get("/jobs").json()["jobs"][0]
    assert (job["status"], job["message"]) == ("failed", "Interrupted by a restart")


def test_shift_and_sync_jobs_act_on_the_chosen_language(library, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    (library.movie / "Arrival.ja.srt").write_text(SAMPLE_SRT)

    queue(library, op="shift", lang="ja", offset_ms=1000, items=[{"kind": "movie", "id": "m1"}])
    queue(library, op="sync", lang="ja", items=[{"kind": "movie", "id": "m1"}])
    queue(library, op="shift", lang="ko", items=[{"kind": "movie", "id": "m1"}])
    run_all()

    import pysubs2
    assert [s.start for s in pysubs2.load(str(library.movie / "Arrival.ja.srt"))] == [2000, 4000]
    ja, en = str(library.movie / "Arrival.ja.srt"), str(library.movie / "Arrival.en.srt")
    assert calls == [["ffsubsync", en, "-i", ja, "-o", ja]]
    assert [j["status"] for j in library.client.get("/jobs").json()["jobs"]] == ["skipped", "done", "done"]


def test_models_list_llama_swap_and_gemini(library, monkeypatch, gemini):
    monkeypatch.setattr(main, "list_models", lambda: [{"id": "qwen3-14b", "name": "qwen3-14b", "loaded": True}])

    body = library.client.get("/models").json()

    assert body["local"] == [{"backend": "llama-swap", "id": "qwen3-14b", "name": "qwen3-14b", "loaded": True}]
    assert body["gemini"]["id"] == "gemini-2.5-flash-lite" and body["local_error"] is None


def test_models_report_an_unreachable_llama_swap(library, monkeypatch):
    import requests

    def down():
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(main, "list_models", down)

    body = library.client.get("/models").json()

    assert body["local"] == [] and "unreachable" in body["local_error"] and body["gemini"] is None


def test_config_lists_servers_and_languages(library):
    body = library.client.get("/config").json()

    assert body["servers"] == [{"name": "jellyfin", "label": "Jellyfin"}]
    assert [l["code"] for l in body["languages"]] == ["zh-TW", "zh-CN", "ja", "ko", "en"]
    assert body["default_lang"] == "zh-TW"


def test_show_page_lists_seasons_and_episode_status(library):
    (library.show / "S01E02.zh-TW.srt").write_text(SAMPLE_SRT)

    seasons = library.client.get("/library/jellyfin/shows/s1").json()["seasons"]

    assert [(s["season"], [(e["number"], e["status"]) for e in s["episodes"]]) for s in seasons] == [
        (1, [(1, "missing"), (2, "done")]), (2, [(1, "unavailable")])]
