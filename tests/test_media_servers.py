import pytest

from conftest import FakeHttp, FakeJellyfinHttp, jellyfin_routes
from media_servers import Jellyfin, MediaServerError, PathMap, Plex, configured_servers


def test_a_server_path_maps_to_the_backend_media_folder():
    assert PathMap("/data/media:/media").to_local("/data/media/Movies/X.mkv") == "/media/Movies/X.mkv"


def test_the_longest_matching_rule_wins():
    paths = PathMap("/data:/other,/data/media:/media")
    assert paths.to_local("/data/media/TV/a.mkv") == "/media/TV/a.mkv"


def test_a_path_no_rule_covers_maps_to_nothing():
    paths = PathMap("/data/media:/media")
    assert paths.to_local("/data/JellyBridge/a.mkv") is None
    assert paths.to_local("/data/mediaextra/a.mkv") is None


def test_a_malformed_path_map_is_refused():
    with pytest.raises(ValueError):
        PathMap("/data/media")


def jellyfin(http=None):
    return Jellyfin("http://jf:8096", "key", PathMap("/data/media:/media"), http or FakeJellyfinHttp(jellyfin_routes()))


def test_jellyfin_lists_movies_and_shows_with_backend_paths():
    titles = jellyfin().titles()

    assert [(t.id, t.kind, t.name, t.year, t.path) for t in titles] == [
        ("m1", "movie", "Arrival", 2016, "/media/Movies/Arrival (2016)/Arrival.mkv"),
        ("m2", "movie", "Elsewhere", 2020, None),
        ("s1", "show", "Severance", 2022, None),
    ]


def test_jellyfin_sends_its_api_key():
    http = FakeJellyfinHttp(jellyfin_routes())
    jellyfin(http).titles()
    assert http.calls[0]["headers"]["X-Emby-Token"] == "key"


def test_jellyfin_episodes_are_ordered_and_missing_files_have_no_path():
    episodes = jellyfin().episodes("s1")

    assert [(e.id, e.season, e.number, e.path) for e in episodes] == [
        ("e1", 1, 1, "/media/TV/Severance/S01E01.mkv"),
        ("e2", 1, 2, "/media/TV/Severance/S01E02.mkv"),
        ("e3", 2, 1, None),
    ]


def test_jellyfin_groups_all_episodes_by_show():
    assert [e.id for e in jellyfin().episodes_by_show()["s1"]] == ["e1", "e2", "e3"]


def test_jellyfin_poster_and_refresh():
    http = FakeJellyfinHttp(jellyfin_routes())
    server = jellyfin(http)

    assert server.poster("m1") == (b"poster-bytes", "image/png")
    server.refresh("m1")
    assert http.calls[-1]["method"] == "POST" and http.calls[-1]["path"] == "/Items/m1/Refresh"


def test_a_jellyfin_error_is_reported():
    with pytest.raises(MediaServerError):
        Jellyfin("http://jf", "bad", PathMap(""), FakeHttp({})).titles()


PLEX_ROUTES = {
    ("GET", "/library/sections"): {"MediaContainer": {"Directory": [
        {"key": "1", "type": "movie", "title": "Movies"},
        {"key": "2", "type": "show", "title": "TV"},
    ]}},
    ("GET", "/library/sections/1/all"): {"MediaContainer": {"Metadata": [
        {"ratingKey": 10, "title": "Arrival", "year": 2016, "type": "movie",
         "Media": [{"Part": [{"file": "/data/media/Movies/Arrival.mkv"}]}]},
    ]}},
    ("GET", "/library/sections/2/all"): {"MediaContainer": {"Metadata": [
        {"ratingKey": 20, "title": "Severance", "year": 2022, "type": "show"},
    ]}},
    ("GET", "/library/metadata/20/allLeaves"): {"MediaContainer": {"Metadata": [
        {"ratingKey": 22, "title": "Half Loop", "index": 2, "parentIndex": 1,
         "Media": [{"Part": [{"file": "/data/media/TV/S01E02.mkv"}]}]},
        {"ratingKey": 21, "title": "Good News", "index": 1, "parentIndex": 1,
         "Media": [{"Part": [{"file": "/data/media/TV/S01E01.mkv"}]}]},
    ]}},
    ("GET", "/library/metadata/10/thumb"): b"plex-poster",
    ("PUT", "/library/metadata/10/refresh"): b"",
}


def plex(http=None):
    return Plex("http://plex:32400", "tok", PathMap("/data/media:/media"), http or FakeHttp(dict(PLEX_ROUTES)))


def test_plex_lists_movies_and_shows_with_backend_paths():
    assert [(t.id, t.kind, t.name, t.year, t.path) for t in plex().titles()] == [
        ("10", "movie", "Arrival", 2016, "/media/Movies/Arrival.mkv"),
        ("20", "show", "Severance", 2022, None),
    ]


def test_plex_sends_its_token():
    http = FakeHttp(dict(PLEX_ROUTES))
    plex(http).titles()
    assert http.calls[0]["headers"]["X-Plex-Token"] == "tok"


def test_plex_episodes_are_ordered_with_backend_paths():
    assert [(e.id, e.season, e.number, e.path) for e in plex().episodes("20")] == [
        ("21", 1, 1, "/media/TV/S01E01.mkv"),
        ("22", 1, 2, "/media/TV/S01E02.mkv"),
    ]


def test_plex_poster_and_refresh():
    http = FakeHttp(dict(PLEX_ROUTES))
    server = plex(http)
    assert server.poster("10") == (b"plex-poster", "image/png")
    server.refresh("10")
    assert http.calls[-1]["method"] == "PUT"


def test_only_servers_with_a_url_are_configured():
    assert list(configured_servers({})) == []
    assert list(configured_servers({"PLEX_URL": "http://p"})) == ["plex"]
    assert list(configured_servers({"JELLYFIN_URL": "http://j", "PLEX_URL": "http://p"})) == ["jellyfin", "plex"]


def test_plex_groups_all_episodes_by_show():
    class EpisodeAwareHttp(FakeHttp):
        def request(self, method, url, params=None, headers=None, timeout=None):
            if url.endswith("/library/sections/2/all") and params == {"type": 4}:
                self.routes[("GET", "/library/sections/2/all")] = {"MediaContainer": {"Metadata": [
                    {"ratingKey": 21, "grandparentRatingKey": 20, "index": 1, "parentIndex": 1,
                     "title": "Good News", "Media": [{"Part": [{"file": "/data/media/TV/S01E01.mkv"}]}]},
                ]}}
            return super().request(method, url, params, headers, timeout)

    by_show = plex(EpisodeAwareHttp(dict(PLEX_ROUTES))).episodes_by_show()

    assert [(e.id, e.path) for e in by_show["20"]] == [("21", "/media/TV/S01E01.mkv")]
