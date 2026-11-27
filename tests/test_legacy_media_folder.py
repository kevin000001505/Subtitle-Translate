"""The legacy endpoints and the legacy queue cannot freeze the backend service.

/translate, /shift and /sync, and the legacy queue monitor, all work on the
media folder. Production mounts it as a hard NFS mount, so a file check or a
subtitle read there can wait forever and never fails by itself. They do that
work in the media folder's own threads, so a request or a queued path that is
stuck there leaves /config, /jobs, /models and the posters answering, and each
flow still finishes with its usual response once the folder answers again. A
media-folder timeout belongs to the pages that must answer a browser: a legacy
flow has none, and waits as long as the folder takes.

See the stalled media folder harness in conftest.
"""
import asyncio
import json
import subprocess

import pysubs2

import main
from conftest import (SAMPLE_SRT, REQUEST_DEADLINE_SECONDS, asgi_client, backend_answers_while,
                      media_folder_is_stuck, request_within, run_scenario)


def sync_progress(client, job_id) -> dict:
    """The last progress event of a legacy sync job, the way the legacy UI reads it."""
    stream = client.get(f"/sync/progress/{job_id}").text
    events = [json.loads(line[len("data: "):]) for line in stream.splitlines() if line.startswith("data: ")]
    return events[-1]


def test_the_backend_answers_while_a_legacy_shift_waits_for_the_media_folder(media_stall, library):
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
            await backend_answers_while(client, media_stall, shift)
            media_stall.release()
            return await asyncio.wait_for(shift, REQUEST_DEADLINE_SECONDS)

    response = run_scenario(scenario())
    assert response.status_code == 200, response.text
    assert [s.start for s in pysubs2.load(str(target))] == [2000, 4000]


def test_a_legacy_shift_keeps_waiting_past_the_time_a_page_waits_for_the_media_folder(
        media_stall, library, short_media_wait):
    """The media-folder timeout is the pages' (the library page, a show page). A
    legacy request has none: it is still waiting long after a page would have
    answered 503, and it answers with its usual body once the folder does, not
    with an error."""
    target = library.movie / "Arrival.zh-TW.srt"
    target.write_text(SAMPLE_SRT)
    media_stall.stall()

    async def scenario():
        async with asgi_client() as client:
            shift = asyncio.create_task(client.post("/shift",
                                                    json={"path": str(target), "offset_ms": 1000}))
            assert await media_folder_is_stuck(media_stall), "the shift never reached the media folder"

            try:                                    # a page would have given up by now
                await asyncio.wait_for(asyncio.shield(shift), 3 * short_media_wait)
                raise AssertionError("the legacy shift answered while the media folder was stalled")
            except TimeoutError:
                pass
            assert not shift.done(), "the legacy shift gave up on the media folder"

            media_stall.release()
            return await asyncio.wait_for(shift, REQUEST_DEADLINE_SECONDS)

    response = run_scenario(scenario())

    assert response.status_code == 200, response.text
    assert response.json() == {"status": "done", "offset_ms": 1000,
                               "shifted": [str(target)], "failed": [], "total": 1}
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
            await backend_answers_while(client, media_stall, shift)
            media_stall.release()
            return await asyncio.wait_for(shift, REQUEST_DEADLINE_SECONDS)

    response = run_scenario(scenario())

    assert response.status_code == 200, response.text
    assert response.json() == {"status": "done", "offset_ms": 1000,
                               "shifted": [str(target)], "failed": [], "total": 1}
    assert [s.start for s in pysubs2.load(str(target))] == [2000, 4000]
    assert [s.start for s in pysubs2.load(str(library.movie / "Arrival.ja.srt"))] == [1000, 3000]


def test_the_backend_answers_while_the_legacy_flows_wait_for_the_media_folder(media_stall, library):
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

            await backend_answers_while(client, media_stall, translate)

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


def test_every_legacy_flow_waits_for_the_media_folder_while_the_backend_answers(media_stall, library,
                                                                                monkeypatch):
    """/shift, /sync, /translate and one pass of the legacy queue monitor all
    reach the media folder. Each of them waits there, and only it waits: the
    backend answers every other request, and after the folder answers each flow
    gives its usual answer."""
    calls = []

    def fake_run(cmd, **kwargs):                     # stands in for ffsubsync
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    target = library.movie / "Arrival.zh-TW.srt"
    target.write_text(SAMPLE_SRT)
    main.add_to_queue(str(library.show))
    media_stall.stall()

    async def scenario():
        async with asgi_client() as client:
            flows = {"shift": asyncio.create_task(client.post("/shift",
                                                              json={"path": str(library.movie),
                                                                    "offset_ms": 100})),
                     "sync": asyncio.create_task(client.post("/sync", json={"path": str(library.movie)})),
                     "translate": asyncio.create_task(client.post("/translate",
                                                                  json={"path": str(library.movie)})),
                     "legacy queue": asyncio.create_task(main.run_next_queued_job())}

            assert await media_folder_is_stuck(media_stall), "no legacy flow reached the media folder"
            response = await request_within(client, "GET", "/config")
            assert response.status_code == 200, response.text
            assert all(not flow.done() for flow in flows.values()), \
                [name for name, flow in flows.items() if flow.done()]

            media_stall.release()
            answers = {name: await asyncio.wait_for(flow, REQUEST_DEADLINE_SECONDS)
                       for name, flow in flows.items() if name != "legacy queue"}
            await asyncio.wait_for(flows["legacy queue"], REQUEST_DEADLINE_SECONDS)
            return answers

    answers = run_scenario(scenario())

    shift = answers["shift"].json()
    assert (shift["status"], shift["total"], shift["failed"]) == ("done", 1, [])
    assert [s.start for s in pysubs2.load(str(target))] == [1100, 3100]
    assert answers["sync"].json()["status"] == "started", answers["sync"].text
    job_id = answers["sync"].json()["job_id"]
    assert sync_progress(library.client, job_id)["status"] == "done"
    assert calls, "the /sync flow never reached ffsubsync"
    assert answers["translate"].json()["status"] == "processing_async"
    assert [s.text for s in pysubs2.load(str(library.show / "S01E01.zh-TW.srt"))] == \
        ["zh-TW: Hello", "zh-TW: Goodbye"], "the queued path never translated"
    assert main.pop_from_queue() is None, "the queued path is still in the legacy queue"


def test_a_legacy_shift_waits_for_the_media_folder_when_it_reads_and_writes_the_subtitles(
        media_stall, library):
    """The shift endpoint reads and writes every subtitle it shifts, and a
    stalled media folder holds a subtitle read the way it holds an existence
    check.

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
            await backend_answers_while(client, media_stall, shift)
            media_stall.release()
            return await asyncio.wait_for(shift, REQUEST_DEADLINE_SECONDS)

    response = run_scenario(scenario())

    assert response.status_code == 200, response.text
    assert response.json()["shifted"] == [str(target)]
    assert [s.start for s in pysubs2.load(str(target))] == [2000, 4000]


def test_a_legacy_translate_waits_for_the_media_folder_when_it_reads_and_writes_the_subtitles(
        media_stall, library):
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
            await backend_answers_while(client, media_stall, translate)
            media_stall.release()
            await asyncio.wait_for(translate, REQUEST_DEADLINE_SECONDS)
            await asyncio.wait_for(queued, REQUEST_DEADLINE_SECONDS)

    run_scenario(scenario())

    assert [s.text for s in pysubs2.load(str(target))] == ["zh-TW: Hello", "zh-TW: Goodbye"]
    episode = library.show / "S01E01.zh-TW.srt"
    assert [s.text for s in pysubs2.load(str(episode))] == ["zh-TW: Hello", "zh-TW: Goodbye"]
    assert main.pop_from_queue() is None, "the queued path is still in the legacy queue"
