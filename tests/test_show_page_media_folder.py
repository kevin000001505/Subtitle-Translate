"""The show page (/library/{server}/shows/{id}) behaves like the library page
while the media folder stalls.

It computes its episodes' badges in one batch on the media folder's own thread
pool, waits only the configured media-folder timeout, then answers 503 naming
that folder and logs a warning. The other endpoints keep answering while it
waits, and it answers normally once the folder does.

See the stalled media folder harness in conftest: the file checks under the test
library root wait on an event, and every request shares one event loop, as
uvicorn serves them in production.
"""
import asyncio
import logging
import time

from conftest import (SAMPLE_SRT, REQUEST_DEADLINE_SECONDS, asgi_client, media_folder_is_stuck,
                      request_within, run_scenario)

# What the healthy show page answers, the same list test_library.py asserts:
# the badges when one episode is already translated, and when none is.
SEVERANCE_SEASONS = [(1, [(1, "missing"), (2, "done")]), (2, [(1, "unavailable")])]
SEVERANCE_SEASONS_WITHOUT_TRANSLATIONS = [(1, [(1, "missing"), (2, "missing")]),
                                          (2, [(1, "unavailable")])]


def seasons_of(body) -> list:
    return [(s["season"], [(e["number"], e["status"]) for e in s["episodes"]])
            for s in body["seasons"]]


def test_a_show_page_that_waits_too_long_answers_503_with_the_media_folder(media_stall, caplog,
                                                                          short_media_wait):
    """Story 10 and story 13: a show page gives up after the media-folder
    timeout, answers 503 naming that folder, and logs the warning an operator
    can match to a NAS problem. The worker waiting on the mount stays waiting."""
    media_stall.stall()
    caplog.clear()

    async def scenario():
        async with asgi_client() as client:
            request = asyncio.create_task(client.get("/library/jellyfin/shows/s1"))
            assert await media_folder_is_stuck(media_stall), "the show page never reached the media folder"

            response = await request_within(client, "GET", "/library/jellyfin/shows/s1",
                                            timeout=short_media_wait + REQUEST_DEADLINE_SECONDS)
            assert response.status_code == 503
            assert response.json()["detail"] == (
                f"The media folder is not answering ({media_stall.root})")

            await asyncio.wait_for(request, short_media_wait + REQUEST_DEADLINE_SECONDS)
            # Each request that gives up logs its own warning.
            warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert len(warnings) == 2, [r.getMessage() for r in warnings]
            assert all(str(media_stall.root) in r.getMessage() for r in warnings), caplog.text

    run_scenario(scenario())


def test_the_api_answers_while_a_show_page_waits_for_the_media_folder(media_stall, short_media_wait):
    """Story 5-8 and story 17 while a *show* page waits: the page header and
    languages (/config), the job list, the model list, a poster and a new job
    (a media-server lookup with no file check) all answer inside the deadline."""
    media_stall.stall()

    async def scenario():
        async with asgi_client() as client:
            show_page = asyncio.create_task(client.get("/library/jellyfin/shows/s1"))
            assert await media_folder_is_stuck(media_stall), "the show page never reached the media folder"

            for method, path, kwargs in (("GET", "/config", {}),
                                         ("GET", "/jobs", {}),
                                         ("GET", "/models", {}),
                                         ("GET", "/poster/jellyfin/m1", {}),
                                         ("POST", "/jobs", {"json": {"server": "jellyfin", "op": "shift",
                                                                     "items": [{"kind": "movie",
                                                                                "id": "m1"}]}})):
                response = await request_within(client, method, path, **kwargs)
                assert response.status_code == 200, (method, path, response.text)

            media_stall.release()
            response = await asyncio.wait_for(show_page, REQUEST_DEADLINE_SECONDS)
            assert response.status_code == 200
            # Its own badge batch was held by the stall, not these later checks:
            # no subtitle is translated in this test.
            assert seasons_of(response.json()) == SEVERANCE_SEASONS_WITHOUT_TRANSLATIONS

    run_scenario(scenario())


def test_a_show_page_answers_normally_after_the_media_folder_starts_again(media_stall, library,
                                                                         short_media_wait):
    """Story 12: the request that gave up does not poison the backend. Once the
    media folder answers, the next show page is a normal 200 with its seasons
    and episode badges, with no restart."""
    media_stall.stall()

    async def scenario():
        async with asgi_client() as client:
            stalled = await request_within(client, "GET", "/library/jellyfin/shows/s1",
                                           timeout=short_media_wait + REQUEST_DEADLINE_SECONDS)
            assert stalled.status_code == 503

            media_stall.release()
            (library.show / "S01E02.zh-TW.srt").write_text(SAMPLE_SRT)

            response = await request_within(client, "GET", "/library/jellyfin/shows/s1")
            assert response.status_code == 200
            assert seasons_of(response.json()) == SEVERANCE_SEASONS

    run_scenario(scenario())


def test_a_show_page_gives_up_once_not_once_per_episode(media_stall, short_media_wait):
    """Story 11: the show page waits once for the media folder, not once per
    episode. A page that waited per episode would take 20 s per episode and
    reach nginx's 60 s timeout before it could give up."""
    media_stall.stall()

    async def scenario():
        async with asgi_client() as client:
            started = time.monotonic()
            response = await request_within(client, "GET", "/library/jellyfin/shows/s1",
                                            timeout=6 * short_media_wait)
            elapsed = time.monotonic() - started
            assert response.status_code == 503
            # One batch waits the timeout once. Two episodes have files here,
            # so a per-episode wait would take at least twice as long.
            assert elapsed < 2 * short_media_wait, f"the show page took {elapsed:.1f} s to give up"

    run_scenario(scenario())


def test_a_show_page_that_gives_up_does_not_hold_the_library_page(media_stall, short_media_wait):
    """One show page that gave up on the folder does not make the library page,
    or another show page, wait for the same stuck worker: each request submits
    its own batch."""
    media_stall.stall()

    async def scenario():
        async with asgi_client() as client:
            for path in ("/library/jellyfin/shows/s1", "/library/jellyfin"):
                response = await request_within(client, "GET", path,
                                                timeout=short_media_wait + REQUEST_DEADLINE_SECONDS)
                assert response.status_code == 503, path

            media_stall.release()
            response = await request_within(client, "GET", "/library/jellyfin")
            assert response.status_code == 200
            assert {t["name"]: t["status"] for t in response.json()["titles"]} == {
                "Arrival": "missing", "Elsewhere": "unmapped", "Severance": "missing"}

    run_scenario(scenario())
