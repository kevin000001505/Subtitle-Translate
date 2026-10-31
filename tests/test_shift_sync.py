import json
import subprocess

import pysubs2
from fastapi.testclient import TestClient

import main
from conftest import SAMPLE_SRT


def starts(path):
    return [s.start for s in pysubs2.load(str(path))]


def sync_and_wait(client, **body):
    job_id = client.post("/sync", json=body).json()["job_id"]
    stream = client.get(f"/sync/progress/{job_id}").text
    events = [json.loads(line[len("data: "):]) for line in stream.splitlines() if line.startswith("data: ")]
    return events[-1]


def test_shifting_a_folder_moves_only_zh_tw_subtitles_by_default(media):
    (media / "Movie.zh-TW.srt").write_text(SAMPLE_SRT)
    (media / "Movie.ja.srt").write_text(SAMPLE_SRT)

    result = TestClient(main.app).post(
        "/shift", json={"path": str(media), "offset_ms": 1000}
    ).json()

    assert starts(media / "Movie.zh-TW.srt") == [2000, 4000]
    assert starts(media / "Movie.ja.srt") == [1000, 3000]
    assert result["total"] == 1


def test_syncing_a_folder_aligns_zh_tw_against_the_english_subtitle_by_default(media, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    (media / "Movie.mkv").write_bytes(b"")
    (media / "Movie.en.srt").write_text(SAMPLE_SRT)
    (media / "Movie.zh-TW.srt").write_text(SAMPLE_SRT)

    final = sync_and_wait(TestClient(main.app), path=str(media))

    en, zh = str(media / "Movie.en.srt"), str(media / "Movie.zh-TW.srt")
    assert calls == [["ffsubsync", en, "-i", zh, "-o", zh]]
    assert final["status"] == "done"


def test_syncing_a_folder_without_zh_tw_subtitles_reports_an_error(media, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: None)
    (media / "Movie.mkv").write_bytes(b"")
    (media / "Movie.ja.srt").write_text(SAMPLE_SRT)

    final = sync_and_wait(TestClient(main.app), path=str(media))

    assert final == {
        "status": "error", "total": 0, "current": 0, "files": [],
        "message": "No zh-TW.srt files found to sync",
        "finished_at": final["finished_at"],
    }


def test_shifting_a_folder_moves_only_the_requested_language(media):
    (media / "Movie.zh-TW.srt").write_text(SAMPLE_SRT)
    (media / "Movie.ja.srt").write_text(SAMPLE_SRT)

    result = TestClient(main.app).post(
        "/shift", json={"path": str(media), "offset_ms": 1000, "target_lang": "ja"}
    ).json()

    assert starts(media / "Movie.ja.srt") == [2000, 4000]
    assert starts(media / "Movie.zh-TW.srt") == [1000, 3000]
    assert result["total"] == 1


def test_syncing_a_folder_aligns_the_requested_language(media, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    (media / "Movie.mkv").write_bytes(b"")
    (media / "Movie.en.srt").write_text(SAMPLE_SRT)
    (media / "Movie.ja.srt").write_text(SAMPLE_SRT)
    (media / "Movie.zh-TW.srt").write_text(SAMPLE_SRT)

    final = sync_and_wait(TestClient(main.app), path=str(media), target_lang="ja")

    en, ja = str(media / "Movie.en.srt"), str(media / "Movie.ja.srt")
    assert calls == [["ffsubsync", en, "-i", ja, "-o", ja]]
    assert final["status"] == "done"


def test_syncing_a_folder_that_only_has_the_requested_language(media, monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    (media / "Movie.mkv").write_bytes(b"")
    (media / "Movie.ja.srt").write_text(SAMPLE_SRT)

    final = sync_and_wait(TestClient(main.app), path=str(media), target_lang="ja")

    assert final["status"] == "done"
    assert final["synced"] == ["Movie.mkv"]


def test_shifting_rejects_a_language_code_that_is_not_a_language(media):
    response = TestClient(main.app).post(
        "/shift", json={"path": str(media), "offset_ms": 1000, "target_lang": "zh TW"}
    )

    assert response.status_code == 422


def test_syncing_rejects_a_language_code_that_is_not_a_language(media):
    response = TestClient(main.app).post(
        "/sync", json={"path": str(media), "target_lang": "../ja"}
    )

    assert response.status_code == 422
