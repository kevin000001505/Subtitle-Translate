# Subtitle Translate

A self-hosted service that fetches subtitles from media servers (Jellyfin, Plex) and translates them with AI models. The web UI (frontend service) talks to the API (backend service), which manages the job queue of translation work.

## Setup (fresh machine)

Prerequisite: Docker with the Compose plugin.

1. Clone the repo:

   ```sh
   git clone https://github.com/kevin000001505/Subtitle-Translate
   cd Subtitle-Translate
   ```

2. Create `.env` from the example and fill in the values. The comments in `.env.example` explain every entry; in particular:

   - `MEDIA` — the host folder that holds your media folder; `${MEDIA}/media` is mounted at `/media` in the backend. Use the same value as the `MEDIA` your media server stack uses.
   - `MEDIA_FOLDER_TIMEOUT_SECONDS` — how long a library page or show page waits for the media folder before it answers 503 naming that folder. Keep it below the frontend proxy's own timeout (nginx waits 60 s). Translation, shift and sync jobs, and the legacy endpoints, are not bound by it.
   - `GEMINI_API_KEY` — leave it empty to turn Gemini off.
   - The Jellyfin / Plex entries are called from inside the backend container, so use an address it can reach (an address on the local network, or its service name if it is on the same Docker network).

   ```sh
   cp .env.example .env
   # then edit .env
   ```

3. Build and start the stack:

   ```sh
   docker compose up -d --build
   ```

   This builds the backend image, then starts both services. The backend creates its job queue (`queue_data/`) on first start.

4. Open the UI at `http://<server>:8081`. Requests to `http://<server>:8081/api/` are proxied to the backend service.

## Redeploy (after updating the code)

Update the existing checkout in place, then start whatever changed:

```sh
cd Subtitle-Translate   # the existing checkout
git pull
docker compose up -d
```

Add `--build` when the Dockerfile or the Python dependencies changed: `up -d` alone reuses the existing backend image.

**Why in-place matters.** The frontend service bind-mounts the `frontend/` folder from the host as its web root, and the backend bind-mounts `queue_data/` (the job queue). If a redeploy re-copies the project tree (a fresh-copy deploy), those directories are **replaced**. The running containers keep their mounts to the old directories — a **stale bind mount**. The host tree looks correct, but the frontend service serves an empty web root: without `index.html`, `/` is a 403 (directory index forbidden) and the fallback rule returns 500 for missing files. The container stays "Up", so `docker ps` shows nothing. An in-place `git pull` updates file contents in place and never replaces a directory, so this cannot happen.

## Web root check and restart policy

- **Web root check.** On every start of the frontend service, the service runs `check-webroot.sh` against its web root before nginx starts. The check validates only `index.html`, so it passes even while other assets are mid-copy during an update. If `index.html` is missing, the service **exits** with a non-zero status, and its log carries a message that names the missing file and the directory it expects. A stale web root therefore fails loudly instead of serving 403s silently. Because the restart policy restarts it, `docker ps` shows the service as `Restarting`, not `Exited`; `docker logs frontend` carries the reason.
- **Known limitation.** The check runs at start. A directory that is replaced while the container is running (a fresh-copy deploy) is caught at the next (re)start, not immediately. Until then the container keeps serving the old, empty web root.
- **Restart policy.** Both services run with `restart: unless-stopped`: they start automatically when the host reboots, and a crashed service restarts on its own, but a service you stopped stays stopped.

## Recovery after a fresh-copy deploy

If the frontend service is serving an empty web root because a deploy replaced `frontend/`:

1. Recreate the frontend container, so its mount re-resolves to the current `frontend/` directory:

   ```sh
   docker compose up -d --force-recreate frontend
   ```

2. The check re-runs at start. The web root now has `index.html`, so nginx starts and the UI serves again.

## Notes

- `docker-compose.override.yml` (applied automatically by `docker compose`) live-mounts the backend modules for development.
- The frontend service bind-mounts `frontend/` as its web root, and `frontend/nginx.conf` as `/etc/nginx/nginx.conf`, both read-only. You can edit the web UI on the host without rebuilding an image.
- The same web root mount exposes every file under `frontend/` as a static file, so `http://<server>:8081/nginx.conf` and `/check-webroot.sh` are also downloadable. Nothing secret lives in either file.
