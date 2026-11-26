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
