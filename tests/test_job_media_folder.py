"""A job that stalls on the media folder holds only itself.

Production mounts the media folder as a hard NFS mount, so a file check on it
can wait forever and never fails by itself. A job may wait that long, but only
that job waits: the job runners make their media-folder checks in the media
folder's own thread pool, so the API keeps answering, the job list still shows
the job as running, and queueing new work (a media-server lookup, no file check)
still answers. See the stalled media folder harness in conftest.
"""
import asyncio
import subprocess

import pysubs2

import main
from conftest import (SAMPLE_SRT, REQUEST_DEADLINE_SECONDS, asgi_client, media_folder_is_stuck,
                      request_within, run_scenario)


async def queue_job(client, **body):
    """Queue one job through the HTTP API."""
    response = await request_within(client, "POST", "/jobs", json={"server": "jellyfin", **body})
    assert response.status_code == 200, response.text
    assert response.json()["created"] == 1, response.json()


async def start_job(stalled):
    """Claim the queued job as a task on this loop, and wait until its first
    media-folder check reaches the stalled media folder."""
    task = asyncio.create_task(main.run_next_job())
    assert await media_folder_is_stuck(stalled), "the job never reached the media folder"
    return task


async def job_is_listed_as_running(client, expected):
    """The job list answers while a job is stuck, and shows it as running."""
    response = await request_within(client, "GET", "/jobs")
    assert response.status_code == 200, response.text
    assert [(j["op"], j["title"], j["status"]) for j in response.json()["jobs"]] == expected


def test_a_translate_job_that_waits_for_the_media_folder_holds_only_itself(media_stall, library):
    """Stories 14, 15: while a translate job waits on the media folder, /jobs
    answers within the deadline and lists that job as running, and /config and
    /models answer with it. When the folder answers again, the job finishes with
    its usual result."""
    media_stall.stall()

    async def scenario():
        async with asgi_client() as client:
            await queue_job(client, items=[{"kind": "movie", "id": "m1"}])
            job = await start_job(media_stall)

            await job_is_listed_as_running(client, [("translate", "Arrival", "running")])
            for path in ("/config", "/models"):
                response = await request_within(client, "GET", path)
                assert response.status_code == 200, path

            media_stall.release()
            assert await asyncio.wait_for(job, REQUEST_DEADLINE_SECONDS) is True

    run_scenario(scenario())

    # The job's usual result, read back through the job list and the file it wrote.
    finished = library.client.get("/jobs").json()["jobs"][0]
    assert (finished["status"], finished["message"]) == ("done", "Wrote Arrival.zh-TW.srt")
    assert [s.text for s in pysubs2.load(str(library.movie / "Arrival.zh-TW.srt"))] == \
        ["zh-TW: Hello", "zh-TW: Goodbye"]


def test_a_shift_job_that_waits_for_the_media_folder_holds_only_itself(media_stall, library):
    """Story 16: a shift job stuck on the media folder leaves the API answering
    and stays visible as running. When the folder answers, it shifts the
    subtitle as it always has."""
    (library.movie / "Arrival.zh-TW.srt").write_text(SAMPLE_SRT)
    media_stall.stall()

    async def scenario():
        async with asgi_client() as client:
            await queue_job(client, op="shift", offset_ms=1000, items=[{"kind": "movie", "id": "m1"}])
            job = await start_job(media_stall)

            await job_is_listed_as_running(client, [("shift", "Arrival", "running")])

            media_stall.release()
            assert await asyncio.wait_for(job, REQUEST_DEADLINE_SECONDS) is True

    run_scenario(scenario())

    finished = library.client.get("/jobs").json()["jobs"][0]
    assert (finished["status"], finished["message"]) == ("done", "Shifted +1000 ms")
    assert [s.start for s in pysubs2.load(str(library.movie / "Arrival.zh-TW.srt"))] == [2000, 4000]


def test_a_sync_job_that_waits_for_the_media_folder_holds_only_itself(media_stall, library, monkeypatch):
    """Story 16: the same for a sync job. The ffsubsync call stands in for a real
    one, as in the sync tests."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    (library.movie / "Arrival.zh-TW.srt").write_text(SAMPLE_SRT)
    media_stall.stall()

    async def scenario():
        async with asgi_client() as client:
            await queue_job(client, op="sync", items=[{"kind": "movie", "id": "m1"}])
            job = await start_job(media_stall)

            await job_is_listed_as_running(client, [("sync", "Arrival", "running")])

            media_stall.release()
            assert await asyncio.wait_for(job, REQUEST_DEADLINE_SECONDS) is True

    run_scenario(scenario())

    finished = library.client.get("/jobs").json()["jobs"][0]
    assert (finished["status"], finished["message"]) == ("done", "Synced against the English subtitle")
    assert calls == [["ffsubsync", str(library.movie / "Arrival.en.srt"),
                      "-i", str(library.movie / "Arrival.zh-TW.srt"),
                      "-o", str(library.movie / "Arrival.zh-TW.srt")]]


def test_queueing_a_movie_answers_while_a_job_waits_for_the_media_folder(media_stall, library):
    """Story 18: queueing a movie needs a media-server lookup and no file check,
    so it answers while file checks are stuck. Stuck file checks run in the media
    pool and cannot use up the lookup threads."""
    media_stall.stall()

    async def scenario():
        async with asgi_client() as client:
            await queue_job(client, items=[{"kind": "movie", "id": "m1"}])
            job = await start_job(media_stall)

            main._library_cache.clear()          # queueing again must look Jellyfin up
            lookups = len(library.http.calls)
            created = await request_within(client, "POST", "/jobs",
                                           json={"server": "jellyfin", "lang": "ja",
                                                 "items": [{"kind": "movie", "id": "m1"}]})
            assert created.status_code == 200, created.text
            assert created.json()["created"] == 1, created.json()
            assert len(library.http.calls) > lookups, "queueing did not look the media server up"

            # The stuck job is still stuck, and it is still the only thing that is.
            assert not job.done()
            await job_is_listed_as_running(client, [("translate", "Arrival", "running"),
                                                    ("translate", "Arrival", "queued")])

            media_stall.release()
            assert await asyncio.wait_for(job, REQUEST_DEADLINE_SECONDS) is True

    run_scenario(scenario())

    assert sorted((j["target_lang"], j["status"]) for j in library.client.get("/jobs").json()["jobs"]) == \
        [("ja", "queued"), ("zh-TW", "done")]


def test_page_requests_that_gave_up_on_the_media_folder_do_not_take_the_api_down(media_stall, library,
                                                                                 short_media_wait):
    """The media pool is shared on purpose: a job waits for the folder, and so do
    the library pages that gave up on it. A thread stuck on a hard mount cannot be
    cancelled, so a page that answered 503 leaves its worker waiting there, and at
    most MEDIA_POOL_WORKERS of them can wait at once. That must not take the API
    down, and the job finishes when the folder answers again, without a restart."""
    media_stall.stall()

    async def scenario():
        async with asgi_client() as client:
            await queue_job(client, items=[{"kind": "movie", "id": "m1"}])

            # Each library page waits its media-folder timeout, answers 503, and
            # leaves its media-pool worker waiting on the mount.
            pages = [asyncio.create_task(client.get("/library/jellyfin"))
                     for _ in range(main.MEDIA_POOL_WORKERS)]
            responses = await asyncio.gather(*pages)
            assert [r.status_code for r in responses] == [503] * len(responses), responses

            # With every media worker waiting on the mount, the endpoints that need
            # no file check still answer, and queueing still looks Jellyfin up.
            main._library_cache.clear()
            response = await request_within(client, "GET", "/jobs")
            assert response.status_code == 200, response.text
            created = await request_within(client, "POST", "/jobs",
                                           json={"server": "jellyfin", "lang": "ja",
                                                 "items": [{"kind": "movie", "id": "m1"}]})
            assert created.status_code == 200, created.text
            assert created.json()["created"] == 1, created.json()

            job = asyncio.create_task(main.run_next_job())
            try:
                await asyncio.wait_for(asyncio.shield(job), short_media_wait)
            except asyncio.TimeoutError:
                pass                               # the job is still waiting on the folder
            assert not job.done(), "the job finished while the media folder was stalled"
            await job_is_listed_as_running(client, [("translate", "Arrival", "running"),
                                                    ("translate", "Arrival", "queued")])

            media_stall.release()
            assert await asyncio.wait_for(job, REQUEST_DEADLINE_SECONDS) is True

    run_scenario(scenario())

    jobs_listed = library.client.get("/jobs").json()["jobs"]
    assert sorted((j["target_lang"], j["status"]) for j in jobs_listed) == [("ja", "queued"),
                                                                           ("zh-TW", "done")]
