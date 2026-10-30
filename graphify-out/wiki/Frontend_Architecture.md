# Frontend Architecture

## Overview
The frontend consists of nginx (serves static files) and a FastAPI backend. Nginx proxies `/api/*` requests to the backend container.

## Network Topology
- **Backend**: FastAPI on port 8000, no port mapping (internal docker network only)
- **Frontend**: Nginx on port 80, mapped to host port 8081
- **NPM**: Proxy host forwards all requests to frontend:8081

## Nginx Proxy Config (`frontend/nginx.conf`)
```
location /api/ {
    proxy_pass http://backend:8000/;  # trailing slash strips /api/ prefix
}
```

## API Calls
Frontend `app.js` uses `API_BASE = '/api'`:
- `/api/list-directory` → `GET /list-directory` on backend
- `/api/translate` → `POST /translate` on backend
- `/api/shift` → `POST /shift` on backend
- `/api/sync` → `POST /sync` on backend
- `/api/sync/progress/{job_id}` → `GET /sync/progress/{job_id}` (SSE) on backend
- `/api/history` → `GET /history` on backend

## Security
Backend is not directly accessible — all API calls go through frontend nginx.
