# Subtitle Translate

A self-hosted service that fetches subtitles from media servers and translates them with AI models. A web UI (frontend) talks to an API (backend) that manages a queue of translation jobs.

## Language

**Frontend service**:
The nginx container that serves the web UI and proxies `/api/` requests to the backend.
_Avoid_: UI server, web server

**Backend service**:
The FastAPI container that runs translation jobs and serves the `/api/` endpoints.
_Avoid_: API server, worker

**Web root**:
The directory nginx serves the UI from. It is a bind mount of the repo's `frontend/` folder, so the files live on the host.
_Avoid_: static folder, document root

**Stale bind mount**:
A failure mode where a running container stays bound to a host directory that a redeploy replaced with a fresh copy, so the container serves the old (now empty or obsolete) directory while the host looks correct.
_Avoid_: broken mount, wrong path

**Job queue**:
The persisted set of translation jobs the backend processes, stored in `queue_data/` on the host.
_Avoid_: task list, backlog

**Media folder**:
The folder the backend opens media files from: `/media` in the container, the local side of the path maps. It is the file system, not the media server's index; see **Media library**.
_Avoid_: media drive, media share

**Media pool**:
The threads the backend uses to open media-folder files, separate from the lookup, model and poster pools. Only these threads can be held by a stalled **media folder**: one check holds one of the pool's `MEDIA_POOL_WORKERS` threads until the folder answers, and the lookup, model and poster pools and the event loop stay free. A library page or a show page waits at most `MEDIA_FOLDER_TIMEOUT_SECONDS` for them and then answers 503 naming the media folder; a job or a legacy request (`/translate`, `/shift`, `/sync`) may wait longer, in a thread no page depends on.
_Avoid_: media thread, worker pool

**Media library**:
A library that the media server indexes: a Jellyfin virtual folder or a Plex library section. Only a library of type movies or tvshows feeds the library page and the episode lookup. Jellyfin skips every other library: a boxsets library, a library with no type, and a placeholder library such as JellyBridge Discover. Jellyfin also skips the items its grouping settings put inside a movies or tvshows library: a BoxSet item (a box set) is not a title, so it is never listed and never counted as unmapped. Plex sections list no such items.
_Avoid_: collection, shelf
