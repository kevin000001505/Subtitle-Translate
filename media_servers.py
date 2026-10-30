"""Jellyfin and Plex behind one library interface.

A media server reports file paths as its own container sees them
(e.g. /data/media/Movies/X.mkv); PathMap turns them into paths this backend
can open (/media/Movies/X.mkv). A path no rule covers maps to None, and the
item is shown as unavailable instead of being translated in the wrong place.
"""
import os
from dataclasses import dataclass, field

import requests

TIMEOUT = 15


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


@dataclass
class Title:
    id: str
    kind: str                 # "movie" or "show"
    name: str
    year: int | None
    path: str | None = None   # movies only: backend path of the video
    server_path: str | None = None
    episodes: list[Episode] = field(default_factory=list)


class MediaServerError(Exception):
    pass


class Jellyfin:
    name = "jellyfin"
    label = "Jellyfin"

    def __init__(self, url: str, api_key: str, path_map: PathMap, http=None):
        self.url = url.rstrip("/")
        self.headers = {"X-Emby-Token": api_key, "Accept": "application/json"}
        self.path_map = path_map
        self.http = http or requests.Session()

    def _call(self, method, path, params=None, raw=False):
        try:
            response = self.http.request(method, f"{self.url}{path}", params=params,
                                         headers=self.headers, timeout=TIMEOUT)
        except requests.RequestException as e:
            raise MediaServerError(f"Jellyfin is unreachable: {e}") from e
        if response.status_code >= 400:
            raise MediaServerError(f"Jellyfin answered {response.status_code} for {path}")
        return response if raw else response.json()

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

    def titles(self) -> list[Title]:
        data = self._call("GET", "/Items", params={
            "Recursive": "true",
            "IncludeItemTypes": "Movie,Series",
            "Fields": "Path,ProductionYear",
            "SortBy": "SortName",
            "SortOrder": "Ascending",
        })
        titles = []
        for item in data.get("Items", []):
            kind = "movie" if item.get("Type") == "Movie" else "show"
            server_path = item.get("Path") if kind == "movie" else None
            titles.append(Title(
                id=item["Id"], kind=kind, name=item.get("Name", ""),
                year=item.get("ProductionYear"),
                path=self.path_map.to_local(server_path), server_path=server_path,
            ))
        return titles

    def episodes_by_show(self) -> dict[str, list[Episode]]:
        data = self._call("GET", "/Items", params={
            "Recursive": "true",
            "IncludeItemTypes": "Episode",
            "Fields": "Path",
        })
        by_show: dict[str, list[Episode]] = {}
        for item in data.get("Items", []):
            by_show.setdefault(item.get("SeriesId", ""), []).append(self._episode(item))
        for episodes in by_show.values():
            episodes.sort(key=lambda e: (e.season, e.number))
        return by_show

    def episodes(self, show_id: str) -> list[Episode]:
        data = self._call("GET", f"/Shows/{show_id}/Episodes", params={"Fields": "Path"})
        episodes = [self._episode(item) for item in data.get("Items", [])]
        return sorted(episodes, key=lambda e: (e.season, e.number))

    def poster(self, item_id: str):
        response = self._call("GET", f"/Items/{item_id}/Images/Primary",
                              params={"fillHeight": 450, "quality": 90}, raw=True)
        return response.content, response.headers.get("Content-Type", "image/jpeg")

    def refresh(self, item_id: str):
        self._call("POST", f"/Items/{item_id}/Refresh", params={
            "metadataRefreshMode": "Default", "imageRefreshMode": "Default",
            "replaceAllMetadata": "false", "replaceAllImages": "false",
        }, raw=True)


class Plex:
    name = "plex"
    label = "Plex"

    def __init__(self, url: str, token: str, path_map: PathMap, http=None):
        self.url = url.rstrip("/")
        self.headers = {"X-Plex-Token": token, "Accept": "application/json"}
        self.path_map = path_map
        self.http = http or requests.Session()

    def _call(self, method, path, params=None, raw=False):
        try:
            response = self.http.request(method, f"{self.url}{path}", params=params,
                                         headers=self.headers, timeout=TIMEOUT)
        except requests.RequestException as e:
            raise MediaServerError(f"Plex is unreachable: {e}") from e
        if response.status_code >= 400:
            raise MediaServerError(f"Plex answered {response.status_code} for {path}")
        return response if raw else response.json().get("MediaContainer", {})

    @staticmethod
    def _file(item) -> str | None:
        for media in item.get("Media", []):
            for part in media.get("Part", []):
                if part.get("file"):
                    return part["file"]
        return None

    def _episode(self, item) -> Episode:
        server_path = self._file(item)
        return Episode(
            id=str(item["ratingKey"]),
            season=item.get("parentIndex") or 0,
            number=item.get("index") or 0,
            name=item.get("title", ""),
            path=self.path_map.to_local(server_path),
            server_path=server_path,
        )

    def _sections(self, kind):
        return [d["key"] for d in self._call("GET", "/library/sections").get("Directory", [])
                if d.get("type") == kind]

    def titles(self) -> list[Title]:
        titles = []
        for plex_type, kind in (("movie", "movie"), ("show", "show")):
            for key in self._sections(plex_type):
                for item in self._call("GET", f"/library/sections/{key}/all").get("Metadata", []):
                    server_path = self._file(item) if kind == "movie" else None
                    titles.append(Title(
                        id=str(item["ratingKey"]), kind=kind, name=item.get("title", ""),
                        year=item.get("year"),
                        path=self.path_map.to_local(server_path), server_path=server_path,
                    ))
        return sorted(titles, key=lambda t: t.name.lower())

    def episodes_by_show(self) -> dict[str, list[Episode]]:
        by_show: dict[str, list[Episode]] = {}
        for key in self._sections("show"):
            container = self._call("GET", f"/library/sections/{key}/all", params={"type": 4})
            for item in container.get("Metadata", []):
                by_show.setdefault(str(item.get("grandparentRatingKey", "")), []).append(self._episode(item))
        for episodes in by_show.values():
            episodes.sort(key=lambda e: (e.season, e.number))
        return by_show

    def episodes(self, show_id: str) -> list[Episode]:
        container = self._call("GET", f"/library/metadata/{show_id}/allLeaves")
        episodes = [self._episode(item) for item in container.get("Metadata", [])]
        return sorted(episodes, key=lambda e: (e.season, e.number))

    def poster(self, item_id: str):
        response = self._call("GET", f"/library/metadata/{item_id}/thumb", raw=True)
        return response.content, response.headers.get("Content-Type", "image/jpeg")

    def refresh(self, item_id: str):
        self._call("PUT", f"/library/metadata/{item_id}/refresh", raw=True)


def configured_servers(env=os.environ) -> dict:
    """The media servers whose URL is set, keyed by name, in switcher order."""
    servers = {}
    if env.get("JELLYFIN_URL"):
        servers["jellyfin"] = Jellyfin(env["JELLYFIN_URL"], env.get("JELLYFIN_API_KEY", ""),
                                       PathMap(env.get("JELLYFIN_PATH_MAP", "")))
    if env.get("PLEX_URL"):
        servers["plex"] = Plex(env["PLEX_URL"], env.get("PLEX_TOKEN", ""),
                               PathMap(env.get("PLEX_PATH_MAP", "")))
    return servers
