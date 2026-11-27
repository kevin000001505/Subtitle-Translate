"""The legacy endpoints and the legacy queue cannot freeze the API.

/translate, /shift and /sync, and the legacy queue monitor, all work on the
media folder. Production mounts it as a hard NFS mount, so a file check or a
subtitle read there can wait forever and never fails by itself. They do that
work in the media folder's own thread pool, so a request or a queued path that
is stuck there leaves /config, /jobs, /models and the posters answering, and
each flow still finishes with its usual response once the folder answers again.
See the stalled media folder harness in conftest.
"""
import asyncio
import subprocess

import pysubs2

import main
from conftest import (SAMPLE_SRT, REQUEST_DEADLINE_SECONDS, asgi_client, media_folder_is_stuck,
                      request_within, run_scenario)


async def the_api_answers_while(client, stalled, waiting):
    """Probe the endpoints that need no media-folder file, while `waiting` is
    stuck on the stalled media folder."""
    assert await media_folder_is_stuck(stalled), "no media-folder file access happened"
    assert not waiting.done(), "the legacy request answered while the media folder was stalled"
    for path in ("/config", "/jobs", "/models", "/poster/jellyfin/m1"):
        response = await request_within(client, "GET", path)
        assert response.status_code == 200, path


def test_the_api_answers_while_a_legacy_shift_waits_for_the_media_folder(media_stall, library):
    """Story 17: while the legacy /shift endpoint is blocked on the media folder,
    /config (and the job list, the model list and a poster) answer within the
    deadline. The shift waits; nothing else does."""
    target = library.movie / "Arrival.zh-TW.srt"
    target.write_text(SAMPLE_SRT)
    media_stall.stall()

    async def scenario():
        async with asgi_client() as client:
            shift = asyncio.create_task(client.post("/shift",
                                                    json={"path": str(target), "offset_ms": 1000}))
            await the_api_answers_while(client, media_stall, shift)
            media_stall.release()
            return await asyncio.wait_for(shift, REQUEST_DEADLINE_SECONDS)

    response = run_scenario(scenario())
    assert response.status_code == 200, response.text
    assert [s.start for s in pysubs2.load(str(target))] == [2000, 4000]


def test_a_legacy_shift_answers_with_its_usual_response_after_the_media_folder_starts_again(
        media_stall, library):
    """Shifting a whole folder looks the folder up, lists it, then reads and
    writes each subtitle. A stall that ends before the request gives up must not
    change the answer: the same body and the same shifted cues as on a healthy
    media folder."""
    target = library.movie / "Arrival.zh-TW.srt"
    target.write_text(SAMPLE_SRT)
    (library.movie / "Arrival.ja.srt").write_text(SAMPLE_SRT)
    media_stall.stall()

    async def scenario():
        async with asgi_client() as client:
            shift = asyncio.create_task(client.post("/shift",
                                                    json={"path": str(library.movie), "offset_ms": 1000}))
            await the_api_answers_while(client, media_stall, shift)
            media_stall.release()
            return await asyncio.wait_for(shift, REQUEST_DEADLINE_SECONDS)

    response = run_scenario(scenario())

    assert response.status_code == 200, response.text
    assert response.json() == {"status": "done", "offset_ms": 1000,
                               "shifted": [str(target)], "failed": [], "total": 1}
    assert [s.start for s in pysubs2.load(str(target))] == [2000, 4000]
    assert [s.start for s in pysubs2.load(str(library.movie / "Arrival.ja.srt"))] == [1000, 3000]


def test_the_api_answers_while_the_legacy_flows_wait_for_the_media_folder(media_stall, library):
    """Story 17: /translate starts its work in the background, and
    the legacy queue monitor picks up a path queued at the Gemini limit. Both
    read and write the media folder for as long as they need, and only they wait:
    /config, the job list, the model list and a poster keep answering, and both
    finish when the folder answers again.

    The queue monitor runs in the app lifespan, which the ASGI transport does not
    start, so the test drives one pass of it (`run_next_queued_job`, the function
    its loop calls each pass) as a task on the same event loop, as the job tests
    do for the job worker."""
    target = library.movie / "Arrival.zh-TW.srt"
    main.add_to_queue(str(library.show))             # queued at the Gemini limit
    media_stall.stall()

    async def scenario():
        async with asgi_client() as client:
            # The ASGI transport holds a request open until its background task
            # ends (uvicorn would have answered already), so a /translate whose
            # background work is stuck shows up as a request task still pending.
            translate = asyncio.create_task(client.post("/translate", json={"path": str(library.movie)}))
            queued = asyncio.create_task(main.run_next_queued_job())

            await the_api_answers_while(client, media_stall, translate)

            media_stall.release()
            response = await asyncio.wait_for(translate, REQUEST_DEADLINE_SECONDS)
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "processing_async", response.json()
            await asyncio.wait_for(queued, REQUEST_DEADLINE_SECONDS)

    run_scenario(scenario())

    assert target.read_text() != SAMPLE_SRT, "the requested movie never translated"
    assert [s.text for s in pysubs2.load(str(target))] == ["zh-TW: Hello", "zh-TW: Goodbye"]
    episode = library.show / "S01E01.zh-TW.srt"
    assert [s.text for s in pysubs2.load(str(episode))] == ["zh-TW: Hello", "zh-TW: Goodbye"]
    assert main.pop_from_queue() is None, "the queued path is still in the legacy queue"


def test_legacy_flows_never_touch_the_media_folder_from_the_event_loop(media_stall, library,
                                                                       monkeypatch):
    """The rule the reviewer checks: every media-folder file access a legacy flow
    makes happens in a media-pool thread.

    The test runs each legacy flow through the HTTP API (and one pass of the
    legacy queue monitor), lets each stall on the folder, and then reads back
    which threads reached it. The event loop's own thread is the one named
    "scenario": a check, a read or a write made there would hold every request.
    """
    calls = []

    def fake_run(cmd, **kwargs):                     # stands in for ffsubsync
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    (library.movie / "Arrival.zh-TW.srt").write_text(SAMPLE_SRT)
    main.add_to_queue(str(library.show))
    media_stall.stall()
    media_stall.reset_checks()

    async def scenario():
        async with asgi_client() as client:
            flows = [asyncio.create_task(client.post("/shift",
                                                      json={"path": str(library.movie), "offset_ms": 100})),
                     asyncio.create_task(client.post("/sync", json={"path": str(library.movie)})),
                     asyncio.create_task(client.post("/translate", json={"path": str(library.movie)})),
                     asyncio.create_task(main.run_next_queued_job())]

            assert await media_folder_is_stuck(media_stall), "no media-folder file access happened"
            response = await request_within(client, "GET", "/config")
            assert response.status_code == 200, response.text

            media_stall.release()
            for flow in flows:
                await asyncio.wait_for(flow, REQUEST_DEADLINE_SECONDS)

    run_scenario(scenario())

    assert media_stall.checks > 0, "the legacy flows made no media-folder file access at all"
    assert all(name.startswith("media") for name in media_stall.check_threads), \
        f"media-folder file access outside the media pool: {sorted(media_stall.check_threads)}"
    assert calls, "the /sync flow never reached ffsubsync after the media folder answered again"


def test_a_legacy_shift_reads_and_writes_the_subtitle_off_the_event_loop(media_stall, library):
    """Acceptance: the shift endpoint reads and writes every subtitle it shifts,
    and a stalled media folder holds a subtitle read the way it holds an
    existence check.

    `stall(checks=False)` holds the subtitle opens and leaves the folder listing
    answering, so the request is sitting on the subtitles. A read or a write
    there, on the event loop, would hold every other request with it."""
    target = library.movie / "Arrival.zh-TW.srt"
    target.write_text(SAMPLE_SRT)
    media_stall.stall(checks=False)

    async def scenario():
        async with asgi_client() as client:
            shift = asyncio.create_task(client.post("/shift",
                                                    json={"path": str(library.movie), "offset_ms": 1000}))
            await the_api_answers_while(client, media_stall, shift)
            media_stall.release()
            return await asyncio.wait_for(shift, REQUEST_DEADLINE_SECONDS)

    response = run_scenario(scenario())

    assert response.status_code == 200, response.text
    assert response.json()["shifted"] == [str(target)]
    assert [s.start for s in pysubs2.load(str(target))] == [2000, 4000]


def test_a_legacy_translate_reads_and_writes_the_subtitle_off_the_event_loop(media_stall, library):
    """The same for a translation: /translate and the legacy queue monitor load
    the English subtitle and save the translated one. Both are media-folder file
    access, and neither may run on the event loop, where it would hold the other
    endpoints with it. As in the shift test, only the opens are held."""
    target = library.movie / "Arrival.zh-TW.srt"
    main.add_to_queue(str(library.show))
    media_stall.stall(checks=False)

    async def scenario():
        async with asgi_client() as client:
            translate = asyncio.create_task(client.post("/translate", json={"path": str(library.movie)}))
            queued = asyncio.create_task(main.run_next_queued_job())
            await the_api_answers_while(client, media_stall, translate)
            media_stall.release()
            await asyncio.wait_for(translate, REQUEST_DEADLINE_SECONDS)
            await asyncio.wait_for(queued, REQUEST_DEADLINE_SECONDS)

    run_scenario(scenario())

    assert [s.text for s in pysubs2.load(str(target))] == ["zh-TW: Hello", "zh-TW: Goodbye"]
    episode = library.show / "S01E01.zh-TW.srt"
    assert [s.text for s in pysubs2.load(str(episode))] == ["zh-TW: Hello", "zh-TW: Goodbye"]
    assert main.pop_from_queue() is None, "the queued path is still in the legacy queue"
