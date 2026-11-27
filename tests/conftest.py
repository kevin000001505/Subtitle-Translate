import asyncio
import os
import pathlib
import shutil
import subprocess
import re
import threading

import httpx
import pytest


# The app reads these at import time. Clear them so the suite runs against
# the code defaults, never against a developer's shell or .env (and never
# calls the real Gemini API).
for _name in ("GEMINI_API_KEY", "GEMINI_MODEL", "TRANSLATOR_BACKEND",
              "TRANSLATE_GEMMA_URL", "TRANSLATE_GEMMA_MODEL", "TRANSLATE_GEMMA_API_KEY"):
    os.environ.pop(_name, None)

import main  # noqa: E402
import translate_gemma  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from media_servers import Jellyfin, PathMap  # noqa: E402

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
        self.prefix = ""   # e.g. a <think> block a reasoning model emits first

    def post(self, url, headers=None, json=None, timeout=None):
        self.requests.append({"url": url, "json": json})
        if self.status_code != 200:
            return FakeResponse(self.status_code, "server error")
        content = json["messages"][0]["content"]
        if isinstance(content, list):          # TranslateGemma's structured request
            body = content[0]["text"].split("INPUT:", 1)[1]
            target = content[0]["target_lang_code"]
        else:                                  # plain chat prompt: system names the language
            body = json["messages"][1]["content"].split("INPUT:", 1)[1]
            target = re.search(r"\((\S+)\)\.", content).group(1)
        reply = "\n".join(f"[{i}] {target}: {t}" for i, t in _LINE_RE.findall(body))
        return FakeResponse(200, self.prefix + reply)


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


@pytest.fixture
def make_mkv(tmp_path):
    """Writes a tiny MKV with an English subtitle track (needs ffmpeg)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")

    def make(path):
        srt = tmp_path / "source.srt"
        srt.write_text(SAMPLE_SRT)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=16x16:d=5",
             "-i", str(srt), "-map", "0", "-map", "1", "-c:v", "ffv1", "-c:s", "srt",
             "-metadata:s:s:0", "language=eng", "-y", str(path)],
            check=True,
        )
        return path
    return make



class FakeHttp:
    """Stands in for requests.Session against a media server: routes map
    (METHOD, path) to a JSON body, bytes, or an HTTP status code."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def request(self, method, url, params=None, headers=None, timeout=None):
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        self.calls.append({"method": method, "path": path, "params": params, "headers": headers, "timeout": timeout})
        body = self.routes.get((method, path), 404)

        class _Resp:
            pass
        resp = _Resp()
        resp.status_code = body if isinstance(body, int) else 200
        resp.headers = {"Content-Type": "image/png"}
        resp.content = body if isinstance(body, bytes) else b""
        resp.json = lambda: body
        return resp


# Jellyfin's libraries ("virtual folders") and the items inside them.
# Jellyfin reports a library type per folder; Discover has none (it is the
# JellyBridge placeholder library) and Collections is a boxsets library.
JELLYFIN_LIBRARIES = [
    {"Name": "Movies", "CollectionType": "movies", "ItemId": "lib-movies",
     "Locations": ["/data/media/Movies"]},
    {"Name": "Shows", "CollectionType": "tvshows", "ItemId": "lib-shows",
     "Locations": ["/data/media/TV"]},
    {"Name": "Discover", "ItemId": "lib-discover", "Locations": ["/data/JellyBridge/Discover"]},
    {"Name": "Collections", "CollectionType": "boxsets", "ItemId": "lib-boxsets",
     "Locations": ["/data/media/Movies/Collections"]},
]

JELLYFIN_ITEMS = {"Items": [
    {"Id": "m1", "Name": "Arrival", "Type": "Movie", "ProductionYear": 2016,
     "Path": "/data/media/Movies/Arrival (2016)/Arrival.mkv", "ParentId": "lib-movies"},
    {"Id": "m2", "Name": "Elsewhere", "Type": "Movie", "ProductionYear": 2020,
     "Path": "/data/JellyBridge/Elsewhere.mkv", "ParentId": "lib-movies"},
    {"Id": "s1", "Name": "Severance", "Type": "Series", "ProductionYear": 2022,
     "ParentId": "lib-shows"},
    # Jellyfin answers a library with its box sets too: with "group movies into
    # collections" on, the Movies library itself holds BoxSet items, even though
    # the request asks for IncludeItemTypes=Movie. A box set's path is a folder
    # in Jellyfin's own config, so no path-map rule covers it.
    {"Id": "bx1", "Name": "Along with the Gods Collection", "Type": "BoxSet",
     "ProductionYear": None, "Path": "/data/collections/Along with the Gods Collection",
     "ParentId": "lib-movies"},
    {"Id": "bx2", "Name": "Severance Collection", "Type": "BoxSet",
     "ProductionYear": None, "Path": "/data/collections/Severance Collection",
     "ParentId": "lib-shows"},
    {"Id": "d1", "Name": "Dune", "Type": "Movie", "ProductionYear": 2021,
     "Path": "/data/JellyBridge/Discover/Dune (2021)/Dune.mkv", "ParentId": "lib-discover"},
    {"Id": "c1", "Name": "Blade Runner Collection", "Type": "BoxSet",
     "ProductionYear": None, "Path": "/data/media/Movies/Blade Runner Collection",
     "ParentId": "lib-boxsets"},
], "TotalRecordCount": 5}

SEVERANCE_EPISODES = [  # in the Shows library
    {"Id": "e2", "Name": "Half Loop", "SeriesId": "s1", "ParentIndexNumber": 1, "IndexNumber": 2,
     "Path": "/data/media/TV/Severance/S01E02.mkv", "LocationType": "FileSystem", "ParentId": "lib-shows"},
    {"Id": "e1", "Name": "Good News", "SeriesId": "s1", "ParentIndexNumber": 1, "IndexNumber": 1,
     "Path": "/data/media/TV/Severance/S01E01.mkv", "LocationType": "FileSystem", "ParentId": "lib-shows"},
    {"Id": "e3", "Name": "Unaired", "SeriesId": "s1", "ParentIndexNumber": 2, "IndexNumber": 1,
     "LocationType": "Virtual", "ParentId": "lib-shows"},
    # A behind-the-scenes special, reported inside the Shows library and by the
    # show's own episode list. It has a real file, but it is not an Episode.
    {"Id": "sp1", "Name": "Severance Behind the Scenes", "Type": "Movie", "SeriesId": "s1",
     "ParentIndexNumber": 1, "IndexNumber": 99,
     "Path": "/data/media/TV/Severance/Behind the Scenes.mkv", "LocationType": "FileSystem",
     "ParentId": "lib-shows"},
]

DISCOVER_EPISODES = [  # placeholder file in the Discover library
    {"Id": "e4", "Name": "Dolor", "SeriesId": "s2", "ParentIndexNumber": 1, "IndexNumber": 1,
     "Path": "/data/JellyBridge/Discover/The Expanse (2015)/S01E01.mkv",
     "LocationType": "FileSystem", "ParentId": "lib-discover"},
]

JELLYFIN_EPISODES = {"Items": SEVERANCE_EPISODES + DISCOVER_EPISODES}


def jellyfin_routes(libraries=None):
    return {
        ("GET", "/Items"): None,  # filled per call type below
        ("GET", "/Library/VirtualFolders"): JELLYFIN_LIBRARIES if libraries is None else libraries,
        ("GET", "/Shows/s1/Episodes"): {"Items": SEVERANCE_EPISODES, "TotalRecordCount": 3},
        ("GET", "/Items/m1/Images/Primary"): b"poster-bytes",
        ("POST", "/Items/m1/Refresh"): b"",
        ("POST", "/Items/e1/Refresh"): b"",
    }


class FakeJellyfinHttp(FakeHttp):
    """/Library/VirtualFolders answers with the configured libraries.
    /Items answers with titles or with all episodes depending on IncludeItemTypes,
    and only with the items of the library (ParentId) the request asked about.
    Set title_items or episode_items to change those listings."""

    title_items = None
    episode_items = None

    def request(self, method, url, params=None, headers=None, timeout=None):
        if url.endswith("/Items") and params:
            episodes = self.episode_items or JELLYFIN_EPISODES
            titles = self.title_items or JELLYFIN_ITEMS
            body = episodes if params.get("IncludeItemTypes") == "Episode" else titles
            parent = params.get("ParentId")
            items = [item for item in body["Items"] if not parent or item.get("ParentId") == parent]
            self.routes[("GET", "/Items")] = {"Items": items, "TotalRecordCount": len(items)}
        return super().request(method, url, params, headers, timeout)


@pytest.fixture
def library(tmp_path, monkeypatch, llama):
    """The backend with a Jellyfin server and a real media folder on disk.

    The media folder is the temporary root's "lib" folder: the file checks the
    badges make run against real files, and StalledMediaFolder can hold them."""
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
    monkeypatch.setattr(main, "_gemini_blocked_until", 0.0)

    class Library:
        pass
    lib = Library()
    lib.root, lib.movie, lib.show, lib.http, lib.client = root, movie, show, http, TestClient(main.app)
    return lib


# ---------- stalled media folder harness ----------
# Production mounts the media folder (/media) as a hard NFS mount: a file check
# on it can wait forever, it never fails by itself, and a thread that waits
# there cannot be cancelled. These helpers hold the file checks a request makes
# under a test library root until the test releases them, and drive the backend
# through one event loop, as uvicorn does.

REQUEST_DEADLINE_SECONDS = 3.0      # a test waits this long for one request
SCENARIO_DEADLINE_SECONDS = 12.0    # ... and this long for a whole scenario
MEDIA_WAIT_SECONDS = 1.0            # ... and this long for the media folder


@pytest.fixture
def short_media_wait(monkeypatch):
    """Make requests give up on the media folder after MEDIA_WAIT_SECONDS.

    Production waits longer (MEDIA_FOLDER_TIMEOUT_SECONDS, default 20 s); a test
    only needs the wait to be bounded. Returns the value, so a test can add it
    to its own deadline for the request that is waiting.
    """
    monkeypatch.setattr(main, "MEDIA_FOLDER_TIMEOUT_SECONDS", MEDIA_WAIT_SECONDS)
    return MEDIA_WAIT_SECONDS


class StalledMediaFolder:
    """File checks and file opens under root wait on an event until the test
    releases the stall.

    install() holds the patched module's own `os` / `os.path` checks, and every
    file open (through `pathlib.Path.open`, which is how pysubs2 reads and writes
    a subtitle), for paths under root only, so nothing else in the test is
    affected. A stalled mount blocks reads and writes the same way it blocks an
    existence check, so a test that looks for "no media-folder file access on the
    event loop" has to see both. stall(checks=False) or stall(opens=False) holds
    one kind only, which lets a flow reach the phase a test means to look at.

    A test must not touch a file under root while the stall is active: its own
    file check or write would wait too.
    """

    def __init__(self, root):
        self.root = os.path.abspath(str(root))
        # A stalled mount holds a file check and a file open apart from each
        # other, so a test can stall one kind only and let a flow reach the phase
        # it means to test (a subtitle read, past the folder listing).
        self._stalled = {"check": threading.Event(), "open": threading.Event()}
        self._stuck = threading.Event()

    def stall(self, checks=True, opens=True):
        """The media folder stops answering.

        stall() holds every media-folder file access. stall(checks=False) holds
        only the opens, so a flow gets past its folder listing and is left sitting
        on the subtitle it reads or writes; stall(opens=False) holds only the
        checks.
        """
        if checks:
            self._stalled["check"].set()
        if opens:
            self._stalled["open"].set()

    def release(self):
        """The media folder answers again."""
        for event in self._stalled.values():
            event.clear()
        self._stuck.clear()

    def wait_until_stuck(self, timeout) -> bool:
        """Whether a file access is waiting on the stalled media folder."""
        return self._stuck.wait(timeout)

    def _under_root(self, path) -> bool:
        target = os.path.abspath(str(path))
        return target == self.root or target.startswith(self.root + os.sep)

    def hold(self, path, kind="check"):
        """Wait here while the media folder is stalled for that kind of access
        and path is under root."""
        if not self._under_root(path):
            return
        stalled = self._stalled[kind]
        if not stalled.is_set():
            return
        self._stuck.set()
        while stalled.is_set():
            stalled.wait(0.05)

    def install(self, monkeypatch, module=main):
        """Give module an `os` that holds the file checks under root, and hold
        file opens under root.

        Patching the real os module instead would hold pytest's own file checks
        too: building a failure report resolves paths, and each of those calls
        would then wait on the stall.
        """
        monkeypatch.setattr(module, "os", _StalledOS(self))
        monkeypatch.setattr(pathlib.Path, "open", _held_path_open(self), raising=False)


# The file checks the backend makes on the media folder.
OS_CHECKS = ("listdir", "stat")
PATH_CHECKS = ("exists", "isdir", "isfile")
_REAL_PATH_OPEN = pathlib.Path.open


def _held_path_open(stall: StalledMediaFolder):
    """A `pathlib.Path.open` that waits on the stalled media folder.

    A file open is how a subtitle is read or written, so it has to wait on the
    folder like an existence check does. A plain function is a descriptor, so
    installed as Path.open it receives the Path instance like the real one does.
    """

    def held(path, *args, **kwargs):
        stall.hold(path, "open")
        return _REAL_PATH_OPEN(path, *args, **kwargs)

    return held


class _StalledView:
    """A stand-in for os or os.path.

    The checks in held_names wait for a path under the stall's root; every other
    attribute (join, getenv, chmod, ...) is the real one.
    """

    def __init__(self, owner, held_names, stall):
        self._owner = owner
        self._held = frozenset(held_names)
        self._stall = stall

    def __getattr__(self, name):
        call = getattr(self._owner, name)
        if name not in self._held:
            return call

        def held(path, *args, **kwargs):
            self._stall.hold(path)
            return call(path, *args, **kwargs)

        return held


class _StalledOS(_StalledView):
    """The os module, and its path submodule, as a stalled media folder sees them."""

    def __init__(self, stall: StalledMediaFolder):
        super().__init__(os, OS_CHECKS, stall)
        self.path = _StalledView(os.path, PATH_CHECKS, stall)


@pytest.fixture
def media_stall(monkeypatch, library):
    """The library fixture, with the backend's media folder able to stall.

    The stall holds main.py's file checks for paths under the library root. It
    is released at teardown, so a scenario that gave up cannot hang the suite."""
    stalled = StalledMediaFolder(library.root)
    stalled.install(monkeypatch)
    yield stalled
    stalled.release()


def asgi_client(app=main.app) -> httpx.AsyncClient:
    """An async HTTP client over the ASGI app, so every request shares one event
    loop as in production.

    Not TestClient: used without `with` it gives each request its own event loop,
    so a stall would look harmless. The ASGI transport does not run the lifespan,
    so the legacy queue monitor never starts (it spins while asyncio.sleep is a
    no-op, as the autouse fixture replaces it).
    """
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://backend")


async def request_within(client, method, path, timeout=REQUEST_DEADLINE_SECONDS, **kwargs):
    """One request with a deadline, so a stalled backend fails the test instead
    of hanging it."""
    return await asyncio.wait_for(client.request(method, path, **kwargs), timeout)


async def media_folder_is_stuck(stalled: StalledMediaFolder,
                                timeout=REQUEST_DEADLINE_SECONDS) -> bool:
    """Wait in a worker thread for a file check to reach the stalled media folder.

    Waiting here would hold the event loop the test is driving, which is exactly
    the failure these tests look for.
    """
    return await asyncio.wait_for(asyncio.to_thread(stalled.wait_until_stuck, timeout), timeout)


# The endpoints that answer without opening a media-folder file: they read
# configuration, the job queue, the model list, or a media-server poster.
ENDPOINTS_WITHOUT_A_MEDIA_FOLDER_FILE = ("/config", "/jobs", "/models", "/poster/jellyfin/m1")


async def backend_answers_while(client, stalled: StalledMediaFolder, waiting,
                                paths=ENDPOINTS_WITHOUT_A_MEDIA_FOLDER_FILE):
    """Probe the endpoints that need no media-folder file, while `waiting` is
    stuck on the stalled media folder.

    `waiting` must still be pending: a request that answered while the folder was
    stalled answered without the file it came for."""
    assert await media_folder_is_stuck(stalled), "no media-folder file access happened"
    assert not waiting.done(), "the waiting request answered while the media folder was stalled"
    for path in paths:
        response = await request_within(client, "GET", path)
        assert response.status_code == 200, path


def run_scenario(coro, deadline=SCENARIO_DEADLINE_SECONDS):
    """Run an async scenario on its own event loop, in a worker thread.

    A request that blocks the event loop never returns, so the deadline is kept
    outside that loop: the test fails when the scenario is still running past it.
    """
    outcome = {}

    def worker():
        try:
            outcome["value"] = asyncio.run(coro)
        except BaseException as exc:      # reported to the test, not swallowed
            outcome["error"] = exc

    thread = threading.Thread(target=worker, name="scenario", daemon=True)
    thread.start()
    thread.join(deadline)
    if thread.is_alive():
        raise AssertionError(
            f"the backend did not answer within {deadline} s: a request blocked the event loop")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")
