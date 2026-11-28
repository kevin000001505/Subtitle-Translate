"""The Jellyfin library behind the backend's library interface.

Jellyfin reports file paths as its own container sees them
(e.g. /data/media/Movies/X.mkv); PathMap turns them into paths this backend
can open (/media/Movies/X.mkv). A path no rule covers maps to None, and the
item is shown as unavailable instead of being translated in the wrong place.
"""
import os
from dataclasses import dataclass

import requests

TIMEOUT = 15
POSTER_TIMEOUT = 3   # a hung server must not hold browser connections for long


# The container's media folder (GLOSSARY: media folder): the folder every path
# map rule maps into. docker-compose.yml mounts ${MEDIA}/media here.
MEDIA_FOLDER = "/media"


class PathMap:
    """Comma-separated server_path:backend_path pairs, longest prefix wins."""

    def __init__(self, spec: str = ""):
        self.rules = []
        for pair in (spec or "").split(","):
            pair = pair.strip()
            if not pair:
                continue
            server, sep, local = pair.partition(":")
            if not sep or not server.strip() or not local.strip():
                raise ValueError(f"Path map entry {pair!r} must be server_path:backend_path")
            self.rules.append((server.strip().rstrip("/"), local.strip().rstrip("/")))
        self.rules.sort(key=lambda rule: len(rule[0]), reverse=True)

    def local_roots(self) -> list[str]:
        """The media-folder roots these rules map into, in order.

        They name the folders this backend opens this server's files from."""
        return sorted({local for _server, local in self.rules})

    def to_local(self, server_path: str | None) -> str | None:
        if not server_path:
            return None
        normalised = server_path.replace("\\", "/")
        for server, local in self.rules:
            if normalised == server or normalised.startswith(server + "/"):
                return local + normalised[len(server):]
        return None


@dataclass
class Episode:
    id: str
    season: int
    number: int
    name: str
    path: str | None          # backend path, None when missing or not mapped
    server_path: str | None = None

    @property
    def unmapped(self) -> bool:
        """The server has a file, but no path-map rule reaches it."""
        return bool(self.server_path) and not self.path


@dataclass
class Title:
    id: str
    kind: str                 # "movie" or "show"
    name: str
    year: int | None
    path: str | None = None   # movies only: backend path of the video
    server_path: str | None = None

    @property
    def unmapped(self) -> bool:
        return bool(self.server_path) and not self.path


class MediaServerError(Exception):
    pass


def _by_episode(episodes):
    return sorted(episodes, key=lambda e: (e.season, e.number))


# Jellyfin library types whose items can be translated, mapped to the title
# kind each one holds. _libraries_by_kind says why every other type is skipped.
JELLYFIN_LIBRARY_TYPES = {"movies": "movie", "tvshows": "show"}

# The item type each title kind holds. Jellyfin answers a library with items of
# other types as well: with "group movies into collections" on, the Movies
# library also returns BoxSet items, even though the request asks for
# IncludeItemTypes=Movie. The item's own Type decides what becomes a title.
JELLYFIN_TITLE_TYPES = {"movie": "Movie", "show": "Series"}

# Episode lookups follow the same rule: only an Episode is an episode.
JELLYFIN_EPISODE_TYPE = "Episode"


def _type_matches(item, expected: str) -> bool:
    """Whether Jellyfin says this item is the type being read.

    An item whose Type is absent is kept: /Items always reports Type in real
    Jellyfin, and a lookup that asks for no type leaves the field out.
    """
    actual = item.get("Type")
    return not actual or actual == expected


class Jellyfin:
    """The Jellyfin library: its titles, episodes, posters and item refreshes.

    Jellyfin authenticates with an Authorization header. The legacy
    X-Emby-Token header is deprecated and the current Jellyfin API answers 401
    for it, so the API key from JELLYFIN_API_KEY goes in Authorization.
    """
    name = "jellyfin"
    label = "Jellyfin"

    def __init__(self, url: str, api_key: str, path_map: PathMap, http=None):
        self.url = url.rstrip("/")
        self.headers = {"Authorization": f'MediaBrowser Token="{api_key}"',
                        "Accept": "application/json"}
        self.path_map = path_map
        self.http = http or requests.Session()

    @property
    def media_folder(self) -> str:
        """The media folder this server's files live in, as this backend names it.

        A server with no path-map rule names the container's mount, so an error
        about the media folder always names a folder."""
        return ", ".join(self.path_map.local_roots()) or MEDIA_FOLDER

    def _send(self, method, path, params=None, timeout=TIMEOUT):
        """One call to Jellyfin, answered with Jellyfin's own response.

        A server that cannot be reached, or that answers 4xx/5xx, becomes
        MediaServerError naming Jellyfin, so every caller fails the same way."""
        try:
            response = self.http.request(method, f"{self.url}{path}", params=params,
                                         headers=self.headers, timeout=timeout)
        except requests.RequestException as e:
            raise MediaServerError(f"{self.label} is unreachable at {self.url} ({type(e).__name__})") from e
        if response.status_code >= 400:
            raise MediaServerError(f"{self.label} answered HTTP {response.status_code} for {path}")
        return response

    def _json(self, method, path, params=None):
        """A call answered as JSON: the form every lookup in this class reads."""
        return self._send(method, path, params).json()

    def _episode(self, item) -> Episode:
        server_path = item.get("Path") if item.get("LocationType") != "Virtual" else None
        return Episode(
            id=item["Id"],
            season=item.get("ParentIndexNumber") or 0,
            number=item.get("IndexNumber") or 0,
            name=item.get("Name", ""),
            path=self.path_map.to_local(server_path),
            server_path=server_path,
        )

    def _libraries_by_kind(self) -> dict[str, list[str]]:
        """The ids of the libraries worth reading, grouped by title kind.

        Jellyfin reports one collection type per library. Only "movies" and
        "tvshows" are used; everything else (a boxsets library, the JellyBridge
        Discover placeholder library, a library with no type) is skipped.
        Fetched once per call, never per item.

        /Library/VirtualFolders needs elevation, and Jellyfin gives every API
        key the Administrator role, so the key in JELLYFIN_API_KEY can read it.
        """
        by_kind: dict[str, list[str]] = {}
        for folder in self._json("GET", "/Library/VirtualFolders"):
            kind = JELLYFIN_LIBRARY_TYPES.get((folder.get("CollectionType") or "").lower())
            if kind and folder.get("ItemId"):
                by_kind.setdefault(kind, []).append(folder["ItemId"])
        return by_kind

    def titles(self) -> list[Title]:
        """The movies and shows this backend can translate.

        An item becomes a title only when Jellyfin reports it as the type that
        library kind holds (JELLYFIN_TITLE_TYPES), because Jellyfin also answers
        a library with the box sets its grouping settings put there.
        """
        titles: dict[str, Title] = {}   # by item id: two libraries can hold the same file
        for kind, library_ids in self._libraries_by_kind().items():
            for library_id in library_ids:
                data = self._json("GET", "/Items", params={
                    "ParentId": library_id,
                    "Recursive": "true",
                    "IncludeItemTypes": "Movie" if kind == "movie" else "Series",
                    "Fields": "Path,ProductionYear",
                    "SortBy": "SortName",
                    "SortOrder": "Ascending",
                })
                for item in data.get("Items", []):
                    if not _type_matches(item, JELLYFIN_TITLE_TYPES[kind]):
                        continue
                    server_path = item.get("Path") if kind == "movie" else None
                    titles.setdefault(item["Id"], Title(
                        id=item["Id"], kind=kind, name=item.get("Name", ""),
                        year=item.get("ProductionYear"),
                        path=self.path_map.to_local(server_path), server_path=server_path,
                    ))
        return sorted(titles.values(), key=lambda t: t.name.lower())

    def episodes_by_show(self) -> dict[str, list[Episode]]:
        """Every episode of every show in the tvshows libraries."""
        by_show: dict[str, list[Episode]] = {}
        seen: set[str] = set()   # two libraries can hold the same file: count it once
        for library_id in self._libraries_by_kind().get("show", []):
            data = self._json("GET", "/Items", params={
                "ParentId": library_id,
                "Recursive": "true",
                "IncludeItemTypes": "Episode",
                "Fields": "Path",
            })
            for item in data.get("Items", []):
                if not _type_matches(item, JELLYFIN_EPISODE_TYPE):
                    continue
                if item["Id"] in seen:
                    continue
                seen.add(item["Id"])
                by_show.setdefault(item.get("SeriesId", ""), []).append(self._episode(item))
        return {show: _by_episode(episodes) for show, episodes in by_show.items()}

    def episodes(self, show_id: str) -> list[Episode]:
        """The episodes of one show, as its own page lists them."""
        data = self._json("GET", f"/Shows/{show_id}/Episodes", params={"Fields": "Path"})
        episodes = [self._episode(item) for item in data.get("Items", [])
                    if _type_matches(item, JELLYFIN_EPISODE_TYPE)]
        return _by_episode(episodes)

    def poster(self, item_id: str):
        """The poster bytes and their content type, with a short timeout: a hung
        server must not hold a browser connection for long."""
        response = self._send("GET", f"/Items/{item_id}/Images/Primary",
                              params={"fillHeight": 450, "quality": 90}, timeout=POSTER_TIMEOUT)
        return response.content, response.headers.get("Content-Type", "image/jpeg")

    def refresh(self, item_id: str):
        """Ask Jellyfin to rescan this item, so a subtitle written next to its
        video shows up in the library."""
        self._send("POST", f"/Items/{item_id}/Refresh", params={
            "metadataRefreshMode": "Default", "imageRefreshMode": "Default",
            "replaceAllMetadata": "false", "replaceAllImages": "false",
        })


def configured_servers(env=os.environ) -> dict:
    """The media servers configured by their URL, keyed by name, in switcher order.

    Jellyfin is the only media server this backend knows, so an environment
    that sets no JELLYFIN_URL configures no server, whatever else it sets."""
    servers = {}
    if env.get("JELLYFIN_URL"):
        servers["jellyfin"] = Jellyfin(env["JELLYFIN_URL"], env.get("JELLYFIN_API_KEY", ""),
                                       PathMap(env.get("JELLYFIN_PATH_MAP", "")))
    return servers
