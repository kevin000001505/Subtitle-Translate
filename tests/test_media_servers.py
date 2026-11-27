import pytest

from conftest import (FakeHttp, FakeJellyfinHttp, JELLYFIN_EPISODES, JELLYFIN_ITEMS, JELLYFIN_LIBRARIES,
                      SEVERANCE_EPISODES, jellyfin_routes)
from media_servers import MEDIA_FOLDER, Jellyfin, MediaServerError, PathMap, Plex, configured_servers


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


def test_a_servers_media_folder_names_the_folders_its_rules_map_into():
    """A page that gives up on a stalled media folder names it (GLOSSARY: media
    folder), so the error points at the mount, not at the media server."""
    server = Jellyfin("http://jf:8096", "key", PathMap("/data:/other,/data/media:/media"))
    assert server.media_folder == "/media, /other"


def test_a_server_with_no_path_map_names_the_containers_media_folder():
    """A server with no rule holds no file this backend can open, but an error
    about its media folder still names a folder."""
    server = Jellyfin("http://jf:8096", "key", PathMap(""))
    assert server.media_folder == MEDIA_FOLDER == "/media"


def jellyfin(http=None):
    return Jellyfin("http://jf:8096", "key", PathMap("/data/media:/media"), http or FakeJellyfinHttp(jellyfin_routes()))


def test_jellyfin_lists_movies_and_shows_with_backend_paths():
    titles = jellyfin().titles()

    assert [(t.id, t.kind, t.name, t.year, t.path) for t in titles] == [
        ("m1", "movie", "Arrival", 2016, "/media/Movies/Arrival (2016)/Arrival.mkv"),
        ("m2", "movie", "Elsewhere", 2020, None),
        ("s1", "show", "Severance", 2022, None),
    ]


def test_jellyfin_lists_only_titles_from_movies_and_tvshows_libraries():
    """Discover (the JellyBridge placeholder library) and Collections are not
    Movies or Shows libraries, so their items are not in the library."""
    titles = jellyfin().titles()

    assert [(t.id, t.kind, t.name) for t in titles] == [
        ("m1", "movie", "Arrival"),
        ("m2", "movie", "Elsewhere"),
        ("s1", "show", "Severance"),
    ]


def test_jellyfin_leaves_out_library_items_that_are_not_movies_or_shows():
    """The shared fixture puts a movie box set in the Movies library and a show
    box set in the Shows library, the way Jellyfin does when its grouping
    settings are on. An item becomes a title only when its own Type matches the
    kind being read."""
    box_sets_in_a_title_library = [(item["Id"], item["ParentId"])
                                   for item in JELLYFIN_ITEMS["Items"]
                                   if item["Type"] == "BoxSet" and item["ParentId"] in ("lib-movies", "lib-shows")]

    assert box_sets_in_a_title_library == [("bx1", "lib-movies"), ("bx2", "lib-shows")]
    assert [t.id for t in jellyfin().titles()] == ["m1", "m2", "s1"]


def test_jellyfin_sends_its_api_key_in_the_authorization_header():
    http = FakeJellyfinHttp(jellyfin_routes())
    jellyfin(http).titles()
    headers = http.calls[0]["headers"]
    assert headers["Authorization"] == 'MediaBrowser Token="key"'
    assert "X-Emby-Token" not in headers


def test_jellyfin_episodes_are_ordered_and_missing_files_have_no_path():
    episodes = jellyfin().episodes("s1")

    assert [(e.id, e.season, e.number, e.path) for e in episodes] == [
        ("e1", 1, 1, "/media/TV/Severance/S01E01.mkv"),
        ("e2", 1, 2, "/media/TV/Severance/S01E02.mkv"),
        ("e3", 2, 1, None),
    ]


def test_jellyfin_groups_all_episodes_by_show():
    assert [e.id for e in jellyfin().episodes_by_show()["s1"]] == ["e1", "e2", "e3"]


def test_jellyfin_episode_lookup_uses_only_tvshows_libraries():
    by_show = jellyfin().episodes_by_show()

    assert {show: [e.id for e in episodes] for show, episodes in by_show.items()} == {
        "s1": ["e1", "e2", "e3"],
    }


def test_jellyfin_episode_lookups_keep_only_episodes():
    """Only Episode items become episodes.

    A show's library can hold other items with a real file - the fixture reports
    a behind-the-scenes special as a Movie - and a phantom episode would add a
    row to the show page and a job to a show queue.
    """
    assert [item["Id"] for item in SEVERANCE_EPISODES if item.get("Type")] == ["sp1"]

    assert [e.id for e in jellyfin().episodes("s1")] == ["e1", "e2", "e3"]
    assert [e.id for e in jellyfin().episodes_by_show()["s1"]] == ["e1", "e2", "e3"]


def test_jellyfin_lists_an_episode_that_reports_no_type():
    """Real Jellyfin reports Type for every /Items result. A lookup that asks for
    no type leaves the field out, and an absent Type never drops an item: the
    episode fixtures report no Type and are still listed."""
    episodes_without_a_type = [item["Id"] for item in SEVERANCE_EPISODES if "Type" not in item]

    assert episodes_without_a_type == ["e2", "e1", "e3"]
    assert [e.id for e in jellyfin().episodes("s1")] == ["e1", "e2", "e3"]


def test_a_jellyfin_server_without_movies_or_shows_libraries_lists_an_empty_library():
    other_libraries = [folder for folder in JELLYFIN_LIBRARIES
                       if folder.get("CollectionType") not in ("movies", "tvshows")]
    server = jellyfin(FakeJellyfinHttp(jellyfin_routes(libraries=other_libraries)))

    assert server.titles() == []
    assert server.episodes_by_show() == {}


def test_jellyfin_orders_titles_by_name_whatever_order_the_libraries_come_in():
    shows_first = list(reversed(JELLYFIN_LIBRARIES))
    titles = jellyfin(FakeJellyfinHttp(jellyfin_routes(libraries=shows_first))).titles()

    assert [t.name for t in titles] == ["Arrival", "Elsewhere", "Severance"]


def test_jellyfin_fetches_the_library_list_once_per_lookup():
    """Once per titles() and once per episodes_by_show(), never once per
    library or per item."""
    http = FakeJellyfinHttp(jellyfin_routes())
    server = jellyfin(http)

    def library_list_calls():
        return [c for c in http.calls if c["path"] == "/Library/VirtualFolders"]

    server.titles()
    assert len(library_list_calls()) == 1
    server.episodes_by_show()
    assert len(library_list_calls()) == 2


def _library_sharing_a_folder(kind: str) -> dict:
    """A second library of this kind whose folder sits inside the first one's.
    Jellyfin then answers both libraries' queries with the same items."""
    library_type, label, folder = (("movies", "Kids Movies", "/data/media/Movies/Kids") if kind == "movie"
                                   else ("tvshows", "Kids Shows", "/data/media/TV/Kids"))
    return {"Name": label, "CollectionType": library_type,
            "ItemId": f"lib-kids-{kind}", "Locations": [folder]}


def test_jellyfin_lists_a_title_two_movies_libraries_share_only_once():
    """Listing the shared file twice would double its badge and the count in
    the path-map warning."""
    kids = _library_sharing_a_folder("movie")
    shared = [dict(item, ParentId=kids["ItemId"]) for item in JELLYFIN_ITEMS["Items"] if item["Id"] == "m1"]
    http = FakeJellyfinHttp(jellyfin_routes(libraries=JELLYFIN_LIBRARIES + [kids]))
    http.title_items = {"Items": JELLYFIN_ITEMS["Items"] + shared}

    titles = jellyfin(http).titles()

    assert [t.id for t in titles] == ["m1", "m2", "s1"]


def test_jellyfin_counts_an_episode_two_shows_libraries_share_only_once():
    kids = _library_sharing_a_folder("show")
    shared = [dict(e, ParentId=kids["ItemId"]) for e in SEVERANCE_EPISODES]
    http = FakeJellyfinHttp(jellyfin_routes(libraries=JELLYFIN_LIBRARIES + [kids]))
    http.episode_items = {"Items": JELLYFIN_EPISODES["Items"] + shared}

    by_show = jellyfin(http).episodes_by_show()

    assert {show: [e.id for e in episodes] for show, episodes in by_show.items()} == {
        "s1": ["e1", "e2", "e3"],
    }


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


@pytest.mark.parametrize("make", [jellyfin, plex])
def test_posters_give_up_quickly_on_a_hung_server(make):
    http = FakeJellyfinHttp(jellyfin_routes()) if make is jellyfin else FakeHttp(dict(PLEX_ROUTES))
    make(http).poster("m1" if make is jellyfin else "10")
    assert http.calls[-1]["timeout"] <= 5
