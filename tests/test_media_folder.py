"""The backend service keeps answering while the media folder stalls.

Production mounts the media folder as a hard NFS mount, so a file check on it
can wait forever and never fails by itself. These tests hold the file checks a
request makes under the test library root on an event, and drive the backend
with one async HTTP client over the ASGI app, so every request shares one event
loop as in production. See the stalled media folder harness in conftest.
"""
import asyncio
import logging

import main
from conftest import (SAMPLE_SRT, REQUEST_DEADLINE_SECONDS, SCENARIO_DEADLINE_SECONDS, asgi_client,
                      backend_answers_while, media_folder_is_stuck, request_within, run_scenario)


def test_the_backend_answers_while_the_library_waits_for_the_media_folder(media_stall):
    """Stories 5-8: the page header and languages (/config), the job list, the
    model list and the posters must not wait for the media folder with the
    library page."""
    media_stall.stall()

    async def scenario():
        async with asgi_client() as client:
            library_request = asyncio.create_task(client.get("/library/jellyfin"))
            await backend_answers_while(client, media_stall, library_request)

            media_stall.release()
            response = await asyncio.wait_for(library_request, REQUEST_DEADLINE_SECONDS)
            assert response.status_code == 200

    run_scenario(scenario())


def test_a_library_page_that_waits_too_long_answers_503_with_the_media_folder(media_stall, caplog,
                                                                              short_media_wait):
    """Story 9: a library page and a show page give up after the media-folder
    timeout, answer 503 naming the media folder, and log a warning. A thread
    waiting on a hard mount cannot be cancelled, so the request answers and the
    worker stays waiting."""
    media_stall.stall()

    async def scenario():
        async with asgi_client() as client:
            for path in ("/library/jellyfin", "/library/jellyfin/shows/s1"):
                response = await request_within(client, "GET", path,
                                                timeout=short_media_wait + REQUEST_DEADLINE_SECONDS)
                assert response.status_code == 503, path
                assert str(media_stall.root) in response.json()["detail"], response.json()

    run_scenario(scenario())

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(str(media_stall.root) in r.getMessage() for r in warnings), caplog.text


def test_the_library_answers_normally_after_the_media_folder_starts_again(media_stall, library,
                                                                         short_media_wait):
    """Story 12: the request that gave up does not poison the backend. Once the
    media folder answers, the next library page is a normal 200 with its badges."""
    media_stall.stall()

    async def scenario():
        async with asgi_client() as client:
            stalled = await request_within(client, "GET", "/library/jellyfin",
                                           timeout=short_media_wait + REQUEST_DEADLINE_SECONDS)
            assert stalled.status_code == 503

            media_stall.release()
            (library.movie / "Arrival.zh-TW.srt").write_text(SAMPLE_SRT)
            (library.show / "S01E01.zh-TW.srt").write_text(SAMPLE_SRT)

            response = await request_within(client, "GET", "/library/jellyfin")
            assert response.status_code == 200
            badges = {t["name"]: (t["status"], t.get("done"), t.get("total"))
                      for t in response.json()["titles"]}
            assert badges == {"Arrival": ("done", None, None),
                              "Elsewhere": ("unmapped", None, None),
                              "Severance": ("partial", 1, 2)}

    run_scenario(scenario())


def test_a_job_that_waits_for_the_media_folder_does_not_hold_the_backend(media_stall):
    """A job's media-folder checks wait in the media folder's own threads, so a
    job stuck on the folder leaves /config, /jobs and /models answering."""
    async def scenario():
        async with asgi_client() as client:
            created = await request_within(client, "POST", "/jobs",
                                           json={"server": "jellyfin", "op": "shift",
                                                 "items": [{"kind": "movie", "id": "m1"}]})
            assert created.status_code == 200, created.text

            media_stall.stall()
            job = asyncio.create_task(main.run_next_job())
            await backend_answers_while(client, media_stall, job, paths=("/config", "/jobs", "/models"))

            media_stall.release()
            assert await asyncio.wait_for(job, REQUEST_DEADLINE_SECONDS) is True

    run_scenario(scenario())


def test_legacy_background_work_waits_for_the_media_folder_while_the_backend_answers(media_stall,
                                                                                     library):
    """/translate and /sync start work that reads the media folder for minutes.
    Those reads wait in the media folder's own threads, so a job stuck on the
    folder leaves /config, the job list, the model list and the posters
    answering, and both finish once the folder answers again."""
    sync_job = "stalled-sync"
    main.sync_jobs[sync_job] = {"status": "starting", "total": 0, "current": 0, "files": []}
    media_stall.stall()

    async def scenario():
        async with asgi_client() as client:
            working = [asyncio.create_task(main.process_movie(str(library.movie))),
                       asyncio.create_task(main.run_ffsubsync_with_progress(
                           sync_job, str(library.movie), "auto"))]
            assert await media_folder_is_stuck(media_stall), \
                "no background read reached the media folder"

            for path in ("/config", "/jobs", "/models", "/poster/jellyfin/m1"):
                response = await request_within(client, "GET", path)
                assert response.status_code == 200, path

            media_stall.release()
            for task in working:
                await asyncio.wait_for(task, SCENARIO_DEADLINE_SECONDS)

    try:
        run_scenario(scenario())
    finally:
        main.sync_jobs.pop(sync_job, None)
