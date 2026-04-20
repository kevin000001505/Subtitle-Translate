"""
OpenSubtitles API Python Client
================================
A comprehensive Python class wrapping all endpoints of the OpenSubtitles REST API v1.
API Documentation: https://opensubtitles.stoplight.io/docs/opensubtitles-api

Base URL:     https://api.opensubtitles.com/api/v1
VIP URL:      https://vip-api.opensubtitles.com/api/v1

Authentication:
  - Api-Key header  : required for all requests (consumer/application key)
  - Bearer token    : required for user-specific actions (login first)

Usage example:
    client = OpenSubtitlesClient(api_key="YOUR_API_KEY", app_name="MyApp", app_version="1.0")
    client.login("username", "password")
    results = client.search_subtitles(query="Inception", languages="en")
    link    = client.download(file_id=results["data"][0]["attributes"]["files"][0]["file_id"])
"""

import struct
import hashlib
import os
from typing import Optional, Union

try:
    import requests
except ImportError:
    raise ImportError("The 'requests' library is required. Install it with: pip install requests")


class OpenSubtitlesError(Exception):
    """Raised when the API returns an error response."""
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class OpenSubtitlesClient:
    """
    Full-featured Python client for the OpenSubtitles REST API v1.

    Covers every endpoint group:
      - Authentication  : login, logout
      - Subtitles       : search
      - Download        : request download URL
      - Features        : search features (movies / TV shows)
      - Discover        : popular features, latest subtitles, most downloaded subtitles
      - Infos           : subtitle formats, languages, AI translation info, AI transcription info
      - User            : user info, credits info, buy credits
      - AI Transcribe   : transcribe media, check status
      - AI Translate    : translate subtitles, check status
      - Utilities       : guessit, detect language from text, detect language from audio, status
    """

    DEFAULT_BASE_URL = "https://api.opensubtitles.com/api/v1"
    VIP_BASE_URL     = "https://vip-api.opensubtitles.com/api/v1"

    def __init__(
        self,
        api_key: str,
        app_name: str = "OpenSubtitlesPythonClient",
        app_version: str = "1.0",
        use_vip: bool = False,
        timeout: int = 30,
    ):
        """
        Initialise the client.

        Args:
            api_key:      Your application API key (obtained from your opensubtitles.com profile).
            app_name:     Name of your application (used in the User-Agent header).
            app_version:  Version of your application.
            use_vip:      Use the VIP API endpoint (faster, requires VIP subscription).
            timeout:      HTTP request timeout in seconds.
        """
        self.api_key      = api_key
        self.app_name     = app_name
        self.app_version  = app_version
        self.timeout      = timeout
        self.token: Optional[str] = None
        self.base_url     = self.VIP_BASE_URL if use_vip else self.DEFAULT_BASE_URL

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _user_agent(self) -> str:
        return f"{self.app_name} v{self.app_version}"

    def _build_headers(self, content_type: Optional[str] = None, require_token: bool = False) -> dict:
        headers = {
            "Accept": "application/json",
            "Api-Key": self.api_key,
            "User-Agent": self._user_agent,
        }
        if content_type:
            headers["Content-Type"] = content_type
        if require_token:
            if not self.token:
                raise OpenSubtitlesError("A valid bearer token is required. Call login() first.")
            headers["Authorization"] = f"Bearer {self.token}"
        elif self.token:
            # Attach token when available even if not strictly required
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _raise_for_status(self, response: requests.Response) -> dict:
        try:
            data = response.json()
        except Exception:
            data = {}
        if not response.ok:
            msg = data.get("message") or data.get("error") or response.reason or "Unknown error"
            raise OpenSubtitlesError(msg, status_code=response.status_code, response=data)
        return data

    def _get(self, path: str, params: Optional[dict] = None, require_token: bool = False) -> dict:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        resp = requests.get(
            self._url(path),
            headers=self._build_headers(require_token=require_token),
            params=params,
            timeout=self.timeout,
        )
        return self._raise_for_status(resp)

    def _post(self, path: str, json: Optional[dict] = None, data=None,
              files=None, require_token: bool = False, content_type: Optional[str] = None) -> dict:
        headers = self._build_headers(
            content_type=content_type if (json is not None and files is None) else None,
            require_token=require_token,
        )
        resp = requests.post(
            self._url(path),
            headers=headers,
            json=json,
            data=data,
            files=files,
            timeout=self.timeout,
        )
        return self._raise_for_status(resp)

    def _delete(self, path: str, require_token: bool = False) -> dict:
        resp = requests.delete(
            self._url(path),
            headers=self._build_headers(require_token=require_token),
            timeout=self.timeout,
        )
        return self._raise_for_status(resp)

    # ------------------------------------------------------------------
    # Authentication
    # POST /login
    # DELETE /logout
    # ------------------------------------------------------------------

    def login(self, username: str, password: str) -> dict:
        """
        Create a JWT token to authenticate a user.

        Automatically stores the returned token for subsequent requests.
        Also updates base_url if the API returns a different base_url
        (e.g. vip-api.opensubtitles.com for VIP users).

        Rate limit: 1 req/s, 10 req/min, 30 req/hour.

        Args:
            username: OpenSubtitles account username.
            password: OpenSubtitles account password.

        Returns:
            dict with keys: user, base_url, token, status.
        """
        result = self._post("/login", json={"username": username, "password": password},
                            content_type="application/json")
        self.token = result.get("token")
        # Honour the base_url returned by the server
        returned_base = result.get("base_url")
        if returned_base:
            self.base_url = f"https://{returned_base}/api/v1"
        return result

    def logout(self) -> dict:
        """
        Destroy the current user session/token.

        Clears the locally stored token after a successful call.

        Returns:
            dict with keys: message, status.
        """
        result = self._delete("/logout", require_token=True)
        self.token = None
        return result

    # ------------------------------------------------------------------
    # Subtitles
    # GET /subtitles
    # ------------------------------------------------------------------

    def search_subtitles(
        self,
        ai_translated: Optional[str] = None,
        episode_number: Optional[int] = None,
        foreign_parts_only: Optional[str] = None,
        hearing_impaired: Optional[str] = None,
        id: Optional[int] = None,
        imdb_id: Optional[int] = None,
        languages: Optional[str] = None,
        machine_translated: Optional[str] = None,
        moviehash: Optional[str] = None,
        moviehash_match: Optional[str] = None,
        order_by: Optional[str] = None,
        order_direction: Optional[str] = None,
        page: Optional[int] = None,
        parent_feature_id: Optional[int] = None,
        parent_imdb_id: Optional[int] = None,
        parent_tmdb_id: Optional[int] = None,
        query: Optional[str] = None,
        season_number: Optional[int] = None,
        tmdb_id: Optional[int] = None,
        trusted_sources: Optional[str] = None,
        type: Optional[str] = None,
        uploader_id: Optional[int] = None,
        year: Optional[int] = None,
    ) -> dict:
        """
        Search for subtitles (GET /subtitles).

        Parameters can be combined. At least one search criterion is recommended.

        Args:
            ai_translated:      "exclude" | "include" (default: include)
            episode_number:     Episode number for TV shows.
            foreign_parts_only: "exclude" | "include" | "only" (default: include)
            hearing_impaired:   "include" | "exclude" | "only" (default: include)
            id:                 OpenSubtitles feature ID.
            imdb_id:            IMDB ID (no leading zeros).
            languages:          Comma-separated language codes sorted alphabetically, e.g. "en,fr".
            machine_translated: "exclude" | "include" (default: exclude)
            moviehash:          16-character hex movie hash.
            moviehash_match:    "include" | "only" (default: include)
            order_by:           Field name to sort by.
            order_direction:    "asc" | "desc"
            page:               Results page (1-based).
            parent_feature_id:  For TV shows – parent feature ID.
            parent_imdb_id:     For TV shows – parent IMDB ID.
            parent_tmdb_id:     For TV shows – parent TMDB ID.
            query:              Free-text / filename search.
            season_number:      Season number for TV shows.
            tmdb_id:            TMDB ID.
            trusted_sources:    "include" | "only" (default: include)
            type:               "movie" | "episode" | "all" (default: all)
            uploader_id:        Filter by uploader (use alone).
            year:               Filter by release year.

        Returns:
            dict with keys: total_pages, total_count, per_page, page, data[].
        """
        params = dict(
            ai_translated=ai_translated,
            episode_number=episode_number,
            foreign_parts_only=foreign_parts_only,
            hearing_impaired=hearing_impaired,
            id=id,
            imdb_id=imdb_id,
            languages=languages,
            machine_translated=machine_translated,
            moviehash=moviehash,
            moviehash_match=moviehash_match,
            order_by=order_by,
            order_direction=order_direction,
            page=page,
            parent_feature_id=parent_feature_id,
            parent_imdb_id=parent_imdb_id,
            parent_tmdb_id=parent_tmdb_id,
            query=query,
            season_number=season_number,
            tmdb_id=tmdb_id,
            trusted_sources=trusted_sources,
            type=type,
            uploader_id=uploader_id,
            year=year,
        )
        return self._get("/subtitles", params=params)

    # ------------------------------------------------------------------
    # Download
    # POST /download
    # ------------------------------------------------------------------

    def download(
        self,
        file_id: int,
        sub_format: Optional[str] = None,
        file_name: Optional[str] = None,
        in_fps: Optional[float] = None,
        out_fps: Optional[float] = None,
        timeshift: Optional[float] = None,
        force_download: Optional[bool] = None,
    ) -> dict:
        """
        Request a temporary download URL for a subtitle file (POST /download).

        Requires a valid bearer token (login first).
        Download quota resets at midnight UTC.

        Args:
            file_id:        file_id from subtitle search results (required).
            sub_format:     Target subtitle format (see get_subtitle_formats()).
            file_name:      Desired output file name.
            in_fps:         Source FPS – required when converting FPS.
            out_fps:        Target FPS – required when converting FPS.
            timeshift:      Seconds to shift subtitle timing (positive or negative).
            force_download: If True, sets Content-Disposition to force-download.

        Returns:
            dict with keys: link, file_name, requests, remaining, message,
                            reset_time, reset_time_utc.
        """
        body: dict = {"file_id": file_id}
        if sub_format     is not None: body["sub_format"]      = sub_format
        if file_name      is not None: body["file_name"]       = file_name
        if in_fps         is not None: body["in_fps"]          = in_fps
        if out_fps        is not None: body["out_fps"]         = out_fps
        if timeshift      is not None: body["timeshift"]       = timeshift
        if force_download is not None: body["force_download"]  = force_download
        return self._post("/download", json=body, require_token=True,
                          content_type="application/json")

    # ------------------------------------------------------------------
    # Features
    # GET /features
    # ------------------------------------------------------------------

    def search_features(
        self,
        feature_id: Optional[int] = None,
        full_search: Optional[bool] = None,
        imdb_id: Optional[str] = None,
        query: Optional[str] = None,
        query_match: Optional[str] = None,
        tmdb_id: Optional[str] = None,
        type: Optional[str] = None,
        year: Optional[int] = None,
    ) -> dict:
        """
        Search for movies / TV shows / episodes (GET /features).

        Useful for autocomplete or to retrieve subtitle counts for a title.

        Args:
            feature_id:  OpenSubtitles feature_id.
            full_search: If True, also search in translated titles/aka.
            imdb_id:     IMDB ID (no leading zeros).
            query:       Text search (min 3 characters). Accepts release/file names.
            query_match: "start" (default) | "word" | "exact"
            tmdb_id:     TheMovieDB ID.
            type:        "" (all) | "movie" | "tvshow" | "episode"
            year:        Filter by year (combine with query).

        Returns:
            dict with movie / episode / tv objects, each having id, type, attributes.
        """
        params = dict(
            feature_id=feature_id,
            full_search=full_search,
            imdb_id=imdb_id,
            query=query,
            query_match=query_match,
            tmdb_id=tmdb_id,
            type=type,
            year=year,
        )
        return self._get("/features", params=params)

    # ------------------------------------------------------------------
    # Discover
    # GET /discover/popular
    # GET /discover/latest
    # GET /discover/most_downloaded
    # ------------------------------------------------------------------

    def discover_popular(
        self,
        language: Optional[str] = None,
        type: Optional[str] = None,
    ) -> dict:
        """
        Discover popular features based on last 30 days downloads (GET /discover/popular).

        Args:
            language: Language code (e.g. "en") or "all".
            type:     "movie" | "tvshow"

        Returns:
            dict containing subtitle objects.
        """
        return self._get("/discover/popular", params=dict(language=language, type=type))

    def discover_latest(
        self,
        language: Optional[str] = None,
        type: Optional[str] = None,
    ) -> dict:
        """
        List 60 most recently uploaded subtitles (GET /discover/latest).

        Args:
            language: Language code (e.g. "en") or "all".
            type:     "movie" | "tvshow"

        Returns:
            dict with keys: total_pages, total_count, page, data[].
        """
        return self._get("/discover/latest", params=dict(language=language, type=type))

    def discover_most_downloaded(
        self,
        language: Optional[str] = None,
        type: Optional[str] = None,
    ) -> dict:
        """
        Discover most downloaded subtitles in the last 30 days (GET /discover/most_downloaded).

        Args:
            language: Language code (e.g. "en") or "all".
            type:     "movie" | "tvshow"

        Returns:
            dict with keys: total_pages, total_count, page, data[].
        """
        return self._get("/discover/most_downloaded", params=dict(language=language, type=type))

    # ------------------------------------------------------------------
    # Infos
    # GET /infos/formats
    # GET /infos/languages
    # GET /ai/info/translation
    # GET /ai/info/transcription
    # ------------------------------------------------------------------

    def get_subtitle_formats(self) -> dict:
        """
        Get all supported subtitle file formats (GET /infos/formats).

        Returns:
            dict with a list of supported format strings.
        """
        return self._get("/infos/formats")

    def get_languages(self) -> dict:
        """
        Get all supported languages with their codes (GET /infos/languages).

        Returns:
            dict with data[] containing language_code and language_name.
        """
        return self._get("/infos/languages")

    def get_ai_translation_info(self) -> dict:
        """
        Get available AI translation APIs and their supported languages
        (GET /ai/info/translation).

        No authentication required.

        Returns:
            dict with data[] containing name, display_name, description,
            pricing, reliability, price, languages_supported[].
        """
        return self._get("/ai/info/translation")

    def get_ai_transcription_info(self) -> dict:
        """
        Get available AI transcription APIs and their supported languages
        (GET /ai/info/transcription).

        No authentication required.

        Returns:
            dict with data[] containing name, display_name, description,
            pricing, reliability, price, languages_supported[].
        """
        return self._get("/ai/info/transcription")

    # ------------------------------------------------------------------
    # User
    # GET /infos/user
    # GET /ai/credits
    # GET /ai/credits/buy
    # ------------------------------------------------------------------

    def get_user_info(self) -> dict:
        """
        Get information about the currently authenticated user (GET /infos/user).

        Requires bearer token (login first).

        Returns:
            dict with data containing: allowed_downloads, level, user_id, vip,
            downloads_count, remaining_downloads.
        """
        return self._get("/infos/user", require_token=True)

    def get_user_credits(self) -> dict:
        """
        Get the AI credits balance for the authenticated user (GET /ai/credits).

        Requires Api-Key and Authorization header.

        Returns:
            dict with data.credits (integer).
        """
        return self._get("/ai/credits", require_token=True)

    def get_buy_credits_options(self) -> dict:
        """
        Get available credit packages with checkout URLs (GET /ai/credits/buy).

        Returns:
            dict with data[] containing name, value, discount_percent, checkout_url.
        """
        return self._get("/ai/credits/buy", require_token=True)

    # ------------------------------------------------------------------
    # AI Transcribe
    # POST /ai/transcribe
    # GET  /ai/transcribe/{correlation_id}
    # ------------------------------------------------------------------

    def ai_transcribe(
        self,
        api: str,
        file: Union[str, bytes],
        language: str,
    ) -> dict:
        """
        Transcribe a media (audio/video) file into subtitles using AI
        (POST /ai/transcribe).

        Credits on the user account are needed.
        Max file size: 100 MB.

        Args:
            api:      Transcription API name (see get_ai_transcription_info()).
            file:     Path to the media file (str) or raw bytes.
            language: Language code of the media audio (e.g. "en").

        Returns:
            dict with keys: status ("CREATED"), correlation_id.
        """
        if isinstance(file, str):
            with open(file, "rb") as fh:
                file_bytes = fh.read()
            filename = os.path.basename(file)
        else:
            file_bytes = file
            filename = "media"

        files_payload = {
            "file": (filename, file_bytes),
        }
        params = {"api": api, "language": language}
        headers = self._build_headers(require_token=True)
        # Remove Content-Type so requests sets it with the multipart boundary
        headers.pop("Content-Type", None)
        resp = requests.post(
            self._url("/ai/transcribe"),
            headers=headers,
            params=params,
            files=files_payload,
            timeout=self.timeout,
        )
        return self._raise_for_status(resp)

    def get_ai_transcribe_status(self, correlation_id: str) -> dict:
        """
        Get the status of a transcription job (GET /ai/transcribe/{correlation_id}).

        Status values: CREATED | PENDING | COMPLETED | ERROR | TIMEOUT

        Args:
            correlation_id: The correlation_id returned by ai_transcribe().

        Returns:
            dict with job status details.
        """
        return self._get(f"/ai/transcribe/{correlation_id}", require_token=True)

    # ------------------------------------------------------------------
    # AI Translate
    # POST /ai/translate
    # GET  /ai/translate/{correlation_id}
    # ------------------------------------------------------------------

    def ai_translate(
        self,
        api: str,
        file: Union[str, bytes],
        translate_to: str,
        file_id: Optional[int] = None,
        translate_from: Optional[str] = "auto",
    ) -> dict:
        """
        Translate a subtitle file using AI (POST /ai/translate).

        Credits on the user account are needed.

        Args:
            api:            Translation API name (see get_ai_translation_info()).
            file:           Path to the subtitle file (str) or raw bytes.
            translate_to:   Target language ISO 639-1 code (e.g. "fr").
            file_id:        file_id from /subtitles search results (optional).
            translate_from: Source language ISO 639-1 code (default "auto").

        Returns:
            dict with keys: status ("CREATED"), correlation_id.
        """
        if isinstance(file, str):
            with open(file, "rb") as fh:
                file_bytes = fh.read()
            filename = os.path.basename(file)
        else:
            file_bytes = file
            filename = "subtitle.srt"

        files_payload = {"file": (filename, file_bytes)}
        params: dict = {"api": api, "translate_to": translate_to}
        if file_id        is not None: params["file_id"]        = file_id
        if translate_from is not None: params["translate_from"] = translate_from

        headers = self._build_headers(require_token=True)
        headers.pop("Content-Type", None)
        resp = requests.post(
            self._url("/ai/translate"),
            headers=headers,
            params=params,
            files=files_payload,
            timeout=self.timeout,
        )
        return self._raise_for_status(resp)

    def get_ai_translate_status(self, correlation_id: str) -> dict:
        """
        Get the status of a translation job (GET /ai/translate/{correlation_id}).

        Status values: CREATED | PENDING | COMPLETED | ERROR | TIMEOUT

        Args:
            correlation_id: The correlation_id returned by ai_translate().

        Returns:
            dict with job status details.
        """
        return self._get(f"/ai/translate/{correlation_id}", require_token=True)

    # ------------------------------------------------------------------
    # Utilities
    # GET  /utilities/guessit
    # POST /ai/detect_language_text
    # POST /ai/detect_language_audio
    # GET  /ai/detect_language_audio/{correlation_id}
    # ------------------------------------------------------------------

    def guessit(self, filename: str) -> dict:
        """
        Extract metadata from a video filename using the GuessIt library
        (GET /utilities/guessit).

        Works for both movies and TV show episodes.

        Args:
            filename: The video filename to analyse (e.g. "movie.2023.1080p.mkv").

        Returns:
            dict with keys: title, year, language, subtitle_language, screen_size,
            streaming_service, source, other, audio_codec, audio_channels,
            video_codec, release_group, type.
        """
        return self._get("/utilities/guessit", params={"filename": filename})

    def detect_language_text(self, file: Union[str, bytes]) -> dict:
        """
        Detect the language of a subtitle file (POST /ai/detect_language_text).

        At least 1 AI credit is required.

        Args:
            file: Path to the subtitle file (str) or raw bytes.

        Returns:
            dict with data.format, data.type, data.language
            (W3C, name, native, ISO_639_1, ISO_639_2b).
        """
        if isinstance(file, str):
            with open(file, "rb") as fh:
                file_bytes = fh.read()
            filename = os.path.basename(file)
        else:
            file_bytes = file
            filename = "subtitle.srt"

        headers = self._build_headers(require_token=True)
        headers.pop("Content-Type", None)
        resp = requests.post(
            self._url("/ai/detect_language_text"),
            headers=headers,
            files={"file": (filename, file_bytes)},
            timeout=self.timeout,
        )
        return self._raise_for_status(resp)

    def detect_language_audio(
        self,
        api: str,
        file: Union[str, bytes],
        language: str,
    ) -> dict:
        """
        Detect the spoken language of a media audio file asynchronously
        (POST /ai/detect_language_audio).

        At least 1 AI credit is required. Max file size: 100 MB.
        Use get_detect_language_audio_status() to poll for the result.

        Args:
            api:      Transcription/detection API name.
            file:     Path to the media file (str) or raw bytes.
            language: Language hint for the media audio (e.g. "en" or "auto").

        Returns:
            dict with keys: status ("CREATED"), correlation_id.
        """
        if isinstance(file, str):
            with open(file, "rb") as fh:
                file_bytes = fh.read()
            filename = os.path.basename(file)
        else:
            file_bytes = file
            filename = "media"

        headers = self._build_headers(require_token=True)
        headers.pop("Content-Type", None)
        resp = requests.post(
            self._url("/ai/detect_language_audio"),
            headers=headers,
            params={"api": api, "language": language},
            files={"file": (filename, file_bytes)},
            timeout=self.timeout,
        )
        return self._raise_for_status(resp)

    def get_detect_language_audio_status(self, correlation_id: str) -> dict:
        """
        Get the status of a detect-language-audio job
        (GET /ai/detect_language_audio/{correlation_id}).

        Status values: CREATED | PENDING | COMPLETED | ERROR | TIMEOUT

        Args:
            correlation_id: The correlation_id returned by detect_language_audio().

        Returns:
            dict with job status details.
        """
        return self._get(f"/ai/detect_language_audio/{correlation_id}", require_token=True)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_movie_hash(file_path: str) -> str:
        """
        Compute the OpenSubtitles movie hash for a local video file.

        The hash is a 16-character hex string derived from the file size
        and the first/last 64 KB of the file content.

        Args:
            file_path: Absolute or relative path to the video file.

        Returns:
            16-character lowercase hex string.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError:        If the file is too small to hash.
        """
        file_size = os.path.getsize(file_path)
        if file_size < 131072:
            raise ValueError("File is too small to compute a reliable movie hash (< 128 KB).")

        chunk_size = 65536  # 64 KB
        movie_hash = file_size

        with open(file_path, "rb") as fh:
            # Read first 64 KB
            for _ in range(chunk_size // 8):
                (value,) = struct.unpack("Q", fh.read(8))
                movie_hash += value
                movie_hash &= 0xFFFFFFFFFFFFFFFF  # keep 64-bit

            # Read last 64 KB
            fh.seek(-chunk_size, os.SEEK_END)
            for _ in range(chunk_size // 8):
                (value,) = struct.unpack("Q", fh.read(8))
                movie_hash += value
                movie_hash &= 0xFFFFFFFFFFFFFFFF

        return f"{movie_hash:016x}"

    def search_by_file(
        self,
        file_path: str,
        languages: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """
        Convenience method: compute the movie hash from a local video file and
        search for matching subtitles in one call.

        Args:
            file_path: Path to the local video file.
            languages: Comma-separated language codes (e.g. "en,fr").
            **kwargs:  Any additional parameters accepted by search_subtitles().

        Returns:
            Same structure as search_subtitles().
        """
        movie_hash = self.compute_movie_hash(file_path)
        filename   = os.path.basename(file_path)
        return self.search_subtitles(
            moviehash=movie_hash,
            query=filename,
            languages=languages,
            **kwargs,
        )

    def download_to_file(
        self,
        file_id: int,
        dest_path: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Convenience method: request a download URL and immediately save the
        subtitle file to disk.

        Args:
            file_id:   file_id from subtitle search results.
            dest_path: Directory or full file path for the saved subtitle.
                       If None, saves in the current working directory using
                       the server-supplied filename.
            **kwargs:  Additional parameters for download() (sub_format, etc.).

        Returns:
            Path where the subtitle file was saved.
        """
        info = self.download(file_id=file_id, **kwargs)
        url  = info["link"]
        suggested_name = info.get("file_name", "subtitle.srt")

        if dest_path is None:
            save_path = suggested_name
        elif os.path.isdir(dest_path):
            save_path = os.path.join(dest_path, suggested_name)
        else:
            save_path = dest_path

        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        with open(save_path, "wb") as fh:
            fh.write(resp.content)
        return save_path
