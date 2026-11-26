#!/bin/sh
# Validate the web root before nginx starts. Pure check: it reads one
# directory and starts nothing. Usage: check-webroot.sh [webroot]
webroot="${1:-/usr/share/nginx/html}"
if [ ! -f "$webroot/index.html" ]; then
    echo "web root check failed: $webroot/index.html is missing. The web root '$webroot' has no index.html, so nginx would serve 403 for / until it is restored." >&2
    exit 1
fi
exit 0
