from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import AfterValidator, BaseModel, Field
from typing import Annotated
import uvicorn
import os
import pysubs2
import logging
import sqlite3
import time
import asyncio
import re
import json
import uuid
import subprocess
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal
import requests
import jobs
from media_servers import MediaServerError, configured_servers
from translate_gemma import TRANSLATE_GEMMA_MODEL, TranslateGemmaClient, list_models
from subtitle_files import (DEFAULT_TARGET_LANG, LANGUAGES, SOURCE_LANG, canonical_lang, is_source_subtitle,
                            output_subtitle_path, sidecar_subtitle_path, strip_source_suffix,
                            subtitle_suffix)

try:
    from google import genai
except ImportError:
    genai = None

# ================= CONFIGURATION =================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

client = genai.Client(api_key=GEMINI_API_KEY) if (GEMINI_API_KEY and genai) else None

# TranslateGemma (local llama.cpp server)
translate_gemma_client = TranslateGemmaClient()

# Which translator to use by default: "translatemgemma" or "gemini"
TRANSLATOR_BACKEND = os.getenv("TRANSLATOR_BACKEND", "translatemgemma")


@dataclass(frozen=True)
class Translator:
    """Which model translates a subtitle.

    backend is "llama-swap" (a local model behind the OpenAI-compatible API)
    or "gemini"; model is that backend's model id. gemini_fallback retries a
    failed local chunk on Gemini."""
    backend: str
    model: str
    gemini_fallback: bool = False


def default_translator() -> Translator:
    """The translator the app used before it could be chosen per request."""
    if TRANSLATOR_BACKEND == "gemini" and client is not None:
        return Translator(backend="gemini", model=GEMINI_MODEL)
    return Translator(backend="llama-swap", model=TRANSLATE_GEMMA_MODEL, gemini_fallback=True)

# --- PLAYBACK SYNC CONTROL ---
# Set to -1000 to show Chinese subtitles 1 second EARLIER.
# If they are still late, try -1500. If too early, try -500.
TIME_OFFSET_MS = -500 

# LIMITS
DAILY_LIMIT = 9500  
GAP_THRESHOLD_MS = 2000
MAX_CHUNK_DURATION_MS = 120000 
DB_FILE = "/app/data/queue.db"

# SPEED SETTINGS
CONCURRENCY_LIMIT = 20 

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("SubtitleAPI")
sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

# A BCP-47 style code such as "ja" or "zh-TW". It becomes part of the
# subtitle file name, so anything else (spaces, slashes, dots) is refused,
# and it is capped at the 35 characters RFC 5646 asks tags to fit in. Casing
# is normalised so "zh-tw" finds an existing "zh-TW" subtitle.
LangCode = Annotated[str, Field(pattern=r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$", max_length=35),
                     AfterValidator(canonical_lang)]

class SubtitleRequest(BaseModel):
    path: str
    target_lang: LangCode = DEFAULT_TARGET_LANG

class ShiftRequest(BaseModel):
    path: str  # can be a file or folder
    offset_ms: int = 0
    target_lang: LangCode = DEFAULT_TARGET_LANG

class SyncRequest(BaseModel):
    path: str  # folder path
    priority: str = "auto"  # "srt", "audio", or "auto"
    target_lang: LangCode = DEFAULT_TARGET_LANG




# ================= SYNC JOB TRACKING =================

# job_id -> { status, total, current, files: [{name, status, method?, error?}], message?, synced?, failed?, finished_at? }
sync_jobs: dict = {}
SYNC_JOB_TTL_SECONDS = 3600  # drop finished jobs after 1h

def _prune_sync_jobs():
    now = time.time()
    stale = [
        jid for jid, j in sync_jobs.items()
        if j.get("finished_at") and (now - j["finished_at"]) > SYNC_JOB_TTL_SECONDS
    ]
    for jid in stale:
        sync_jobs.pop(jid, None)


def ffsubsync_command(video_path: str, target_srt: str, priority: str) -> tuple[list[str], str]:
    """The ffsubsync call that aligns target_srt in place, and its method:
    against the English subtitle ("srt") when allowed and present, otherwise
    against the video's audio. ffsubsync tells a subtitle reference by its
    extension; its old --srt flag is ambiguous in current releases."""
    en_srt = sidecar_subtitle_path(video_path, SOURCE_LANG)
    if priority in ("srt", "auto") and os.path.exists(en_srt):
        return ["ffsubsync", en_srt, "-i", target_srt, "-o", target_srt], "srt"
    return ["ffsubsync", video_path, "-i", target_srt, "-o", target_srt], "audio"


async def run_ffsubsync_with_progress(job_id: str, folder_path: str, priority: str,
                                     target_lang: str = DEFAULT_TARGET_LANG):
    """
    Run ffsubsync for all matching files in folder_path, updating sync_jobs[job_id]
    with per-file progress as each file completes.
    """
    job = sync_jobs[job_id]

    if not os.path.isdir(folder_path):
        job["status"]      = "error"
        job["message"]     = f"Folder not found: {folder_path}"
        job["finished_at"] = time.time()
        return

    video_files = sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith((".mkv", ".mp4"))
    )
    if not video_files:
        job["status"]      = "error"
        job["message"]     = "No video files (.mkv/.mp4) found in folder"
        job["finished_at"] = time.time()
        return

    # Build work list — only files that have a matching <target_lang>.srt
    files = []
    for video_file in video_files:
        target_srt = sidecar_subtitle_path(os.path.join(folder_path, video_file), target_lang)
        if os.path.exists(target_srt):
            files.append({"name": video_file, "status": "pending"})
        else:
            logger.warning(f"SYNC: No {target_lang}.srt for {video_file}, skipping")

    if not files:
        job["status"]      = "error"
        job["message"]     = f"No {target_lang}.srt files found to sync"
        job["finished_at"] = time.time()
        return

    job["total"]   = len(files)
    job["current"] = 0
    job["files"]   = files
    job["status"]  = "running"

    synced = []
    failed = []

    for file_entry in files:
        video_file = file_entry["name"]
        file_entry["status"] = "processing"

        video_path = os.path.join(folder_path, video_file)
        target_srt = sidecar_subtitle_path(video_path, target_lang)
        en_srt     = sidecar_subtitle_path(video_path, SOURCE_LANG)

        cmd, file_entry["method"] = ffsubsync_command(video_path, target_srt, priority)
        logger.info(f"SYNC: {file_entry['method']}-ref mode for {video_file}")

        try:
            result = await asyncio.to_thread(
                subprocess.run, cmd, capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0:
                file_entry["status"] = "done"
                synced.append(video_file)
                logger.info(f"SYNC: Synced {video_file}")
            else:
                error_msg = (result.stderr or "unknown error")[:300]
                file_entry["status"] = "failed"
                file_entry["error"]  = error_msg
                failed.append({"file": video_file, "error": error_msg})
                logger.error(f"SYNC: Failed {video_file}: {error_msg}")
        except subprocess.TimeoutExpired:
            file_entry["status"] = "failed"
            file_entry["error"]  = "timed out"
            failed.append({"file": video_file, "error": "timed out"})
            logger.error(f"SYNC: Timeout for {video_file}")
        except FileNotFoundError:
            file_entry["status"] = "failed"
            file_entry["error"]  = "ffsubsync not installed"
            failed.append({"file": video_file, "error": "ffsubsync not installed"})
            job["status"]      = "error"
            job["message"]     = "ffsubsync is not installed"
            job["finished_at"] = time.time()
            return
        except Exception as e:
            file_entry["status"] = "failed"
            file_entry["error"]  = str(e)
            failed.append({"file": video_file, "error": str(e)})
            logger.error(f"SYNC: Error for {video_file}: {e}")

        job["current"] += 1

    job["status"]      = "done"
    job["synced"]      = synced
    job["failed"]      = failed
    job["message"]     = f"Synced {len(synced)} file(s), {len(failed)} failed"
    job["finished_at"] = time.time()
    logger.info(f"SYNC: {job['message']}")
    if synced:
        log_history(folder_path, "sync")


# ================= DATABASE & QUEUE LOGIC =================

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS usage_log (timestamp REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS job_queue
                 (id INTEGER PRIMARY KEY, path TEXT, status TEXT, added_at REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY, path TEXT, operation TEXT, timestamp REAL)''')
    queue_columns = [row[1] for row in c.execute("PRAGMA table_info(job_queue)")]
    if "target_lang" not in queue_columns:
        c.execute(f"ALTER TABLE job_queue ADD COLUMN target_lang TEXT NOT NULL DEFAULT '{DEFAULT_TARGET_LANG}'")
    conn.commit()
    conn.close()
    jobs.init(DB_FILE)

def log_api_call():
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        c = conn.cursor()
        c.execute("INSERT INTO usage_log VALUES (?)", (time.time(),))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB Log Error: {e}")

def get_calls_last_24h():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    cutoff = time.time() - (24 * 3600)
    c.execute("DELETE FROM usage_log WHERE timestamp < ?", (cutoff,))
    conn.commit()
    c.execute("SELECT COUNT(*) FROM usage_log")
    count = c.fetchone()[0]
    conn.close()
    return count

def add_to_queue(path, target_lang=DEFAULT_TARGET_LANG):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM job_queue WHERE path = ? AND target_lang = ? AND status = 'PENDING'",
              (path, target_lang))
    if c.fetchone() is None:
        c.execute("INSERT INTO job_queue (path, status, added_at, target_lang) VALUES (?, 'PENDING', ?, ?)",
                  (path, time.time(), target_lang))
        conn.commit()
        logger.info(f"QUEUE: Added {path} to queue.")
    conn.close()

def pop_from_queue():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, path, target_lang FROM job_queue WHERE status = 'PENDING' ORDER BY added_at ASC LIMIT 1")
    row = c.fetchone()
    if row:
        c.execute("UPDATE job_queue SET status = 'PROCESSING' WHERE id = ?", (row[0],))
        conn.commit()
        conn.close()
        return row[1], row[2]
    conn.close()
    return None

def mark_done_in_queue(path, target_lang=DEFAULT_TARGET_LANG):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM job_queue WHERE path = ? AND target_lang = ?", (path, target_lang))
    conn.commit()
    conn.close()

def log_history(path: str, operation: str):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        c = conn.cursor()
        c.execute("INSERT INTO history (path, operation, timestamp) VALUES (?, ?, ?)",
                  (path, operation, time.time()))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"History log error: {e}")

def get_history(limit: int = 200):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT path, operation, timestamp FROM history ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return [{"path": r[0], "operation": r[1], "timestamp": r[2]} for r in rows]

def get_smart_chunks(subs):
    chunks = []
    current_chunk = []
    for i, sub in enumerate(subs):
        if not current_chunk:
            current_chunk.append((i, sub))
            continue
        prev_sub = current_chunk[-1][1]
        time_gap = sub.start - prev_sub.end
        chunk_duration = sub.end - current_chunk[0][1].start
        if time_gap > GAP_THRESHOLD_MS or chunk_duration > MAX_CHUNK_DURATION_MS:
            chunks.append(current_chunk)
            current_chunk = []
        current_chunk.append((i, sub))
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# ================= MKV SUBTITLE EXTRACTION =================

async def extract_english_sub_from_mkv(mkv_path: str) -> str | None:
    """
    Extract English subtitle from a specific MKV file.
    Returns the path to the extracted .en.srt file, or None if failed.
    """
    if not os.path.exists(mkv_path):
        logger.error(f"MKV file not found: {mkv_path}")
        return None

    logger.info(f"EXTRACT: Probing MKV: {mkv_path}")

    # Probe subtitle streams
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "s", mkv_path
    ]
    try:
        probe_result = await asyncio.to_thread(
            subprocess.run, probe_cmd, capture_output=True, text=True
        )
        streams = json.loads(probe_result.stdout).get("streams", [])

        # Find English subtitle stream by language tag
        eng_stream = None
        for stream in streams:
            lang = stream.get("tags", {}).get("language", "").lower()
            if lang in ("eng", "en"):
                eng_stream = stream
                break
        # Fall back to first subtitle stream if no language tag found
        if eng_stream is None and streams:
            eng_stream = streams[0]
            logger.warning(f"EXTRACT: No English tag found in {os.path.basename(mkv_path)}, using first subtitle stream")

        if eng_stream is None:
            logger.warning(f"EXTRACT: No subtitle streams found in {mkv_path}")
            return None

        # Extract to .en.srt
        stream_index = eng_stream["index"]
        out_srt = sidecar_subtitle_path(mkv_path, SOURCE_LANG)

        extract_cmd = [
            "ffmpeg", "-v", "quiet", "-i", mkv_path,
            "-map", f"0:{stream_index}", "-y", out_srt
        ]
        result = await asyncio.to_thread(
            subprocess.run, extract_cmd, capture_output=True, text=True
        )

        if result.returncode == 0 and os.path.exists(out_srt):
            logger.info(f"EXTRACT: Extracted -> {os.path.basename(out_srt)}")
            return out_srt
        else:
            logger.error(f"EXTRACT: ffmpeg failed: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"EXTRACT: Error probing/extracting {mkv_path}: {e}")
        return None

async def find_or_extract_english_sub(folder: str) -> str | None:
    """
    Look for an existing .en.srt in the folder.
    If not found, use the first MKV file and extract its English subtitle.
    Returns the path to the .en.srt file, or None if not possible.
    """
    # 1. Check for existing .en.srt
    for f in os.listdir(folder):
        if is_source_subtitle(f):
            logger.info(f"EXTRACT: Found existing English sub: {f}")
            return os.path.join(folder, f)

    # 2. Look for first MKV/MP4 file and extract
    mkv_files = [f for f in os.listdir(folder) if f.lower().endswith((".mkv", ".mp4"))]
    if not mkv_files:
        logger.info(f"EXTRACT: No .en.srt and no MKV/MP4 found in {folder}")
        return None

    mkv_path = os.path.join(folder, mkv_files[0])
    return await extract_english_sub_from_mkv(mkv_path)
def get_progress_bar(current, total, length=20):
    percent = current / total
    filled_length = int(length * percent)
    bar = "█" * filled_length + "-" * (length - filled_length)
    return f"[{bar}] {int(percent * 100)}%"

_ID_BLOCK_RE = re.compile(
    r'\[(\d+)\]\s*(.*?)(?=(?:\r?\n)\s*\[\d+\]|\Z)',
    re.DOTALL,
)

def parse_gemini_response(response_text):
    """
    Parse lines of the form `[ID] translated text`. Supports multi-line
    translations; stops each entry at the next `[ID]` on its own line,
    so trailing prose after the final entry is bounded by newline-breaks.
    """
    updates = {}
    for match in _ID_BLOCK_RE.finditer(response_text):
        try:
            sub_id = int(match.group(1))
        except ValueError:
            continue
        text = match.group(2).strip()
        # Drop any trailing non-translation tail on the last entry:
        # if the text contains a blank line, keep only the first block.
        if "\n\n" in text:
            text = text.split("\n\n", 1)[0].strip()
        updates[sub_id] = text
    return updates

async def generate_with_gemini(prompt, model, index):
    for attempt in range(3):
        try:
            response = await client.aio.models.generate_content(model=model, contents=prompt)
            log_api_call()
            return response.text
        except Exception as e:
            if attempt == 2:
                logger.error(f"Failed chunk {index} (gemini): {e}")
                return None
            await asyncio.sleep((attempt + 1) * 2)

async def generate_with_llama_swap(batch_text, context_text, model, target_lang):
    for attempt in range(3):
        result_text = await translate_gemma_client.translate(batch_text, context_text, model, target_lang)
        if result_text is not None:
            return result_text
        if attempt < 2:
            await asyncio.sleep((attempt + 1) * 2)
    return None

async def process_single_chunk(index, batch, context_text, total_chunks, progress_counter, translator, target_lang):
    async with sem: 
        batch_text = "\n".join([f"[{idx}] {sub.text}" for idx, sub in batch])
        
        prompt = f"""
        CONTEXT: The movie is about: {context_text[:2000]}...
        TASK: Translate to {target_lang}.
        
        CRITICAL RULES:
        1. Format MUST be: [ID] Translated Text
        2. If the original has multiple lines (e.g., two speakers), your output MUST have multiple lines too.
        3. Do not merge separate dialogue lines into one.
        
        INPUT:
        {batch_text}
        """

        if translator.backend == "gemini":
            result_text = await generate_with_gemini(prompt, translator.model, index)
        else:
            result_text = await generate_with_llama_swap(batch_text, context_text, translator.model, target_lang)
            if result_text is None:
                if not (translator.gemini_fallback and client is not None):
                    logger.error(f"Failed chunk {index}: {translator.model} returned nothing")
                    return None
                logger.warning(f"{translator.model} failed for chunk {index}, falling back to Gemini")
                result_text = await generate_with_gemini(prompt, GEMINI_MODEL, index)

        if result_text:
            progress_counter['completed'] += 1
            current = progress_counter['completed']
            if progress_counter.get('report'):
                progress_counter['report'](current / total_chunks)
            if current % 10 == 0 or current == total_chunks:
                bar = get_progress_bar(current, total_chunks)
                logger.info(f"PROGRESS: {bar} ({current}/{total_chunks})")

        return result_text

async def translate_single_subtitle(en_sub_path: str, translator: Translator | None = None,
                                    target_lang: str = DEFAULT_TARGET_LANG, progress=None,
                                    queue_on_limit: bool = True) -> bool:
    """
    Translate a single subtitle file to target_lang.
    Returns True if successful, False otherwise. progress(fraction) is called
    as chunks finish. At the daily limit the file goes to the legacy queue,
    unless queue_on_limit is False (a library job waits in the job queue).
    """
    translator = translator or default_translator()
    if translator.backend == "gemini" and client is None:
        logger.error(f"Cannot translate {en_sub_path} with {translator.model}: GEMINI_API_KEY is not set")
        return False
    logger.info(f"START: Processing {en_sub_path}")

    try:
        subs = await asyncio.to_thread(pysubs2.load, en_sub_path)
    except Exception as e:
        logger.error(f"Failed to load subtitle {en_sub_path}: {e}")
        return False

    chunks = get_smart_chunks(subs)
    total_chunks = len(chunks)
    
    # The daily limit counts Gemini calls, so it only binds when Gemini may be used.
    may_use_gemini = translator.backend == "gemini" or translator.gemini_fallback
    if may_use_gemini and (get_calls_last_24h() + total_chunks) > DAILY_LIMIT:
        logger.warning(f"LIMIT HIT: {en_sub_path}")
        if queue_on_limit:
            add_to_queue(en_sub_path, target_lang)
        return False

    logger.info(f"Translating {total_chunks} chunks (Concurrency: {CONCURRENCY_LIMIT})...")
    full_text_preview = " ".join([s.text.replace("\n", " ") for s in subs[:500]])

    tasks = []
    progress_counter = {'completed': 0, 'report': progress}
    
    for i, batch in enumerate(chunks):
        task = process_single_chunk(i, batch, full_text_preview, total_chunks, progress_counter, translator, target_lang)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    # Apply translations
    success_count = 0
    for raw_text in results:
        if raw_text:
            updates = parse_gemini_response(raw_text)
            for sub_id, new_text in updates.items():
                if 0 <= sub_id < len(subs):
                    subs[sub_id].text = new_text.strip().replace("\r", "")
                    success_count += 1

    logger.info(f"Applied {success_count} translations.")
    if success_count == 0:
        # Saving would write the English text under the target name, and the
        # skip check would then block every retry.
        logger.error(f"FAILED: no line of {en_sub_path} was translated; nothing saved")
        return False

    # Apply timing offset to ALL subtitles so every line is shifted consistently,
    # including any lines the translation skipped or left untouched.
    if TIME_OFFSET_MS != 0:
        for sub in subs:
            sub.start = max(0, sub.start + TIME_OFFSET_MS)
            sub.end   = max(0, sub.end   + TIME_OFFSET_MS)
        logger.info(f"Applied {TIME_OFFSET_MS}ms offset to all {len(subs)} subtitles.")

    # Save
    new_path = output_subtitle_path(en_sub_path, target_lang)
    await asyncio.to_thread(subs.save, new_path, encoding="utf-8")

    try:
        os.chmod(new_path, 0o644)
        source_stat = os.stat(en_sub_path)
        os.chown(new_path, source_stat.st_uid, source_stat.st_gid)
    except:
        pass

    logger.info(f"DONE: Saved {new_path}")
    return True

async def process_movie(input_path: str, translator: Translator | None = None, target_lang: str = DEFAULT_TARGET_LANG):
    """
    Main entry point. Handles both single files and series folders.
    - If input_path is a file: translate its subtitle
    - If input_path is a folder: find all .en.srt, MKV, and MP4 files and translate their subtitles
    """
    if not os.path.exists(input_path):
        logger.error(f"Path not found: {input_path}")
        mark_done_in_queue(input_path, target_lang)
        return

    # Check if input is a folder (series) or file
    if os.path.isdir(input_path):
        logger.info(f"SERIES: Processing folder {input_path}")
        movie_dir = input_path

        # Find all .en.srt, MKV, and MP4 files in the folder
        srt_files   = [f for f in os.listdir(movie_dir) if is_source_subtitle(f)]
        video_files = [f for f in os.listdir(movie_dir) if f.lower().endswith((".mkv", ".mp4"))]
        if not srt_files and not video_files:
            logger.warning(f"SKIP: No .en.srt or MKV/MP4 files found in {movie_dir}")
            mark_done_in_queue(input_path, target_lang)
            return

        logger.info(f"SERIES: Found {len(srt_files)} .en.srt and {len(video_files)} video file(s).")

        # Process existing .en.srt files — skip individual episodes already translated
        processed_bases = set()
        for srt_file in sorted(srt_files):
            base     = strip_source_suffix(srt_file)
            srt_path = os.path.join(movie_dir, srt_file)
            out_path = output_subtitle_path(srt_path, target_lang)
            if os.path.exists(out_path):
                logger.info(f"EPISODE: Skipping {srt_file}, {os.path.basename(out_path)} already exists.")
                processed_bases.add(base.lower())
                continue
            processed_bases.add(base.lower())
            logger.info(f"EPISODE: Translating {srt_file}")
            success = await translate_single_subtitle(srt_path, translator, target_lang)
            if not success:
                logger.warning(f"EPISODE: Failed to translate {srt_path}")

        # Process video files whose translation doesn't exist yet
        for video_file in sorted(video_files):
            base    = os.path.splitext(video_file)[0]
            out_path = sidecar_subtitle_path(os.path.join(movie_dir, video_file), target_lang)
            if os.path.exists(out_path):
                logger.info(f"EPISODE: Skipping {video_file}, {os.path.basename(out_path)} already exists.")
                continue
            if base.lower() in processed_bases:
                logger.info(f"EPISODE: Skipping {video_file}, .en.srt already processed.")
                continue
            video_path = os.path.join(movie_dir, video_file)
            logger.info(f"EPISODE: Extracting subtitle from {video_file}")
            en_sub_path = await extract_english_sub_from_mkv(video_path)
            if en_sub_path is None:
                logger.warning(f"EPISODE: Could not extract subtitle from {video_file}")
                continue
            success = await translate_single_subtitle(en_sub_path, translator, target_lang)
            if not success:
                logger.warning(f"EPISODE: Failed to translate {en_sub_path}")

        logger.info(f"SERIES: Completed {movie_dir}")
        log_history(input_path, "translate")
        mark_done_in_queue(input_path, target_lang)
    else:
        # Single file mode — resolve to an .en.srt and translate it
        movie_dir = os.path.dirname(input_path)

        if is_source_subtitle(input_path):
            en_sub_path = input_path
        else:
            en_sub_path = await find_or_extract_english_sub(movie_dir)

        if en_sub_path is None:
            logger.warning(f"SKIP: Could not find or extract an English subtitle in {movie_dir}")
            mark_done_in_queue(input_path, target_lang)
            return

        # Skip if the translation already exists for this specific file.
        out_path = output_subtitle_path(en_sub_path, target_lang)
        if os.path.exists(out_path):
            logger.info(f"SKIP: {out_path} already exists.")
            mark_done_in_queue(input_path, target_lang)
            return

        success = await translate_single_subtitle(en_sub_path, translator, target_lang)
        if success:
            log_history(input_path, "translate")
        mark_done_in_queue(input_path, target_lang)

# ================= SCHEDULER & APP =================

async def run_next_queued_job():
    used = get_calls_last_24h()
    if (DAILY_LIMIT - used) > 100:
        next_job = pop_from_queue()
        if next_job:
            next_file, target_lang = next_job
            logger.info(f"MONITOR: Starting queued job: {next_file} ({target_lang})")
            await process_movie(next_file, target_lang=target_lang)

async def queue_monitor():
    while True:
        await run_next_queued_job()
        await asyncio.sleep(3600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    asyncio.create_task(queue_monitor())
    asyncio.create_task(job_worker())
    yield
    # Shutdown (if needed)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/translate")
async def trigger_translation(req: SubtitleRequest, background_tasks: BackgroundTasks):
    used = get_calls_last_24h()
    if used > DAILY_LIMIT:
        add_to_queue(req.path, req.target_lang)
        return {"status": "queued_limit_reached", "message": "Daily limit reached — queued"}

    background_tasks.add_task(process_movie, req.path, target_lang=req.target_lang)
    return {"status": "processing_async", "message": "Translation started in background"}

@app.post("/shift")
async def shift_subtitle(req: ShiftRequest):
    if not os.path.exists(req.path):
        return {"status": "error", "message": "Path not found"}

    # If it's a file, shift just that file
    if os.path.isfile(req.path):
        targets = [req.path]
    # If it's a folder, shift all <target_lang>.srt inside
    else:
        suffix = subtitle_suffix(req.target_lang).lower()
        targets = [
            os.path.join(req.path, f)
            for f in os.listdir(req.path)
            if f.lower().endswith(suffix)
        ]

    if not targets:
        return {"status": "error", "message": f"No {req.target_lang}.srt files found"}

    shifted = []
    failed = []
    for srt_path in targets:
        try:
            subs = pysubs2.load(srt_path)
            subs.shift(ms=req.offset_ms)
            subs.save(srt_path, encoding="utf-8")
            shifted.append(srt_path)
        except Exception as e:
            failed.append({"path": srt_path, "error": str(e)})

    if shifted:
        log_history(req.path, "shift")
    return {
        "status": "done",
        "offset_ms": req.offset_ms,
        "shifted": shifted,
        "failed": failed,
        "total": len(shifted)
    }

@app.post("/sync")
async def sync_folder(req: SyncRequest, background_tasks: BackgroundTasks):
    """
    Start an async sync job. Returns a job_id immediately; poll
    GET /sync/progress/{job_id} (SSE) for live per-file progress.
    """
    if not os.path.exists(req.path):
        return {"status": "error", "message": "Path not found"}

    _prune_sync_jobs()
    job_id = str(uuid.uuid4())
    sync_jobs[job_id] = {
        "status":  "starting",
        "total":   0,
        "current": 0,
        "files":   [],
    }
    logger.info(f"SYNC: Queued job {job_id} for {req.path} (priority={req.priority})")
    background_tasks.add_task(run_ffsubsync_with_progress, job_id, req.path, req.priority, req.target_lang)
    return {"job_id": job_id, "status": "started"}


@app.get("/sync/progress/{job_id}")
async def sync_progress(job_id: str):
    """SSE endpoint — streams job state once per second until done/error."""
    async def event_gen():
        while True:
            job = sync_jobs.get(job_id)
            if job is None:
                yield f"data: {json.dumps({'status': 'error', 'message': 'Job not found'})}\n\n"
                break
            yield f"data: {json.dumps(job)}\n\n"
            if job["status"] in ("done", "error"):
                break
            await asyncio.sleep(1)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/history")
def history_endpoint(limit: int = 200):
    return {"history": get_history(limit)}


@app.get("/list-directory")
def list_directory_endpoint(path: str):
    """
    List contents of a directory for the browser UI.
    
    Returns JSON with items array containing {name, type, path} objects.
    type is either "folder" or "file".
    """
    if not os.path.exists(path) or not os.path.isdir(path):
        return {"items": [], "error": "Path not found or not a directory"}
    
    items = []
    for entry in sorted(os.listdir(path)):
        full_path = os.path.join(path, entry)
        item_type = "folder" if os.path.isdir(full_path) else "file"
        items.append({
            "name": entry,
            "type": item_type,
            "path": full_path
        })
    
    return {"items": items}


# ================= LIBRARY JOBS =================

# Media servers configured in .env (JELLYFIN_URL / PLEX_URL), in switcher order.
SERVERS = configured_servers()
LIBRARY_TTL_SECONDS = 120
_library_cache: dict = {}   # server name -> (fetched_at, titles, episodes_by_show)
job_wakeup = asyncio.Event()


async def run_translate_job(job, report) -> tuple[str, str]:
    video, lang = job["video_path"], job["target_lang"]
    if not os.path.exists(video):
        return "failed", "Video file not found"
    out_path = sidecar_subtitle_path(video, lang)
    if os.path.exists(out_path) and not job["overwrite"]:
        return "skipped", f"{os.path.basename(out_path)} already exists"
    source = sidecar_subtitle_path(video, SOURCE_LANG)
    if not os.path.exists(source):
        source = await extract_english_sub_from_mkv(video)
        if source is None:
            return "failed", "No English subtitle (no .en.srt and none in the video)"
    translator = Translator(backend=job["backend"], model=job["model"])
    if not await translate_single_subtitle(source, translator, lang, progress=report, queue_on_limit=False):
        return "failed", f"Translation with {job['model']} failed (see the backend log)"
    return "done", f"Wrote {os.path.basename(out_path)}"


async def run_shift_job(job, report) -> tuple[str, str]:
    target = sidecar_subtitle_path(job["video_path"], job["target_lang"])
    if not os.path.exists(target):
        return "skipped", f"No {os.path.basename(target)} to shift"
    offset = int(job["params"].get("offset_ms", 0))
    subs = await asyncio.to_thread(pysubs2.load, target)
    subs.shift(ms=offset)
    await asyncio.to_thread(subs.save, target, encoding="utf-8")
    return "done", f"Shifted {offset:+d} ms"


async def run_sync_job(job, report) -> tuple[str, str]:
    video = job["video_path"]
    target = sidecar_subtitle_path(video, job["target_lang"])
    if not os.path.exists(target):
        return "skipped", f"No {os.path.basename(target)} to sync"
    cmd, method = ffsubsync_command(video, target, job["params"].get("priority", "auto"))
    result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        return "failed", f"ffsubsync ({method}) failed: {(result.stderr or 'unknown error')[-300:]}"
    return "done", f"Synced against the {'English subtitle' if method == 'srt' else 'audio'}"


JOB_RUNNERS = {"translate": run_translate_job, "shift": run_shift_job, "sync": run_sync_job}


async def run_next_job() -> bool:
    """Run the oldest queued job to completion. Returns False when nothing
    was runnable. Gemini jobs wait while the daily limit is (nearly) used up."""
    skip = ("gemini",) if (DAILY_LIMIT - get_calls_last_24h()) <= 100 else ()
    job = jobs.claim_next(DB_FILE, skip)
    if job is None:
        return False
    logger.info(f"JOB {job['id']}: {job['op']} {job['video_path']} ({job['target_lang']})")

    def report(fraction):
        jobs.update(DB_FILE, job["id"], progress=round(min(max(fraction, 0), 1) * 100, 1))

    try:
        status, message = await JOB_RUNNERS[job["op"]](job, report)
    except Exception as e:
        logger.exception(f"JOB {job['id']} crashed")
        status, message = "failed", f"Error: {e}"

    server = SERVERS.get(job["server"] or "")
    if status == "done" and server and job["item_id"]:
        try:
            await asyncio.to_thread(server.refresh, job["item_id"])
        except Exception as e:
            message += f" (warning: {server.label} refresh failed: {e})"
    if status == "done":
        log_history(job["video_path"], job["op"])
    jobs.update(DB_FILE, job["id"], status=status, message=message,
                progress=100 if status == "done" else job["progress"])
    return True


async def job_worker():
    while True:
        try:
            if await run_next_job():
                continue
        except Exception:
            logger.exception("JOB worker error")
        job_wakeup.clear()
        try:
            await asyncio.wait_for(job_wakeup.wait(), timeout=30)
        except asyncio.TimeoutError:
            pass


def _server(name: str):
    server = SERVERS.get(name)
    if server is None:
        raise HTTPException(404, f"Media server {name!r} is not configured")
    return server


async def _library(name: str, refresh: bool = False):
    server = _server(name)
    cached = _library_cache.get(name)
    if cached and not refresh and time.time() - cached[0] < LIBRARY_TTL_SECONDS:
        return cached[1], cached[2]
    try:
        titles = await asyncio.to_thread(server.titles)
        episodes = await asyncio.to_thread(server.episodes_by_show)
    except MediaServerError as e:
        raise HTTPException(502, str(e))
    _library_cache[name] = (time.time(), titles, episodes)
    return titles, episodes


def _subtitle_status(path, lang, active) -> str:
    if not path:
        return "unavailable"
    if (path, lang) in active:
        return "running"
    return "done" if os.path.exists(sidecar_subtitle_path(path, lang)) else "missing"


class LibraryQuery(BaseModel):
    lang: LangCode = DEFAULT_TARGET_LANG


@app.get("/config")
def config_endpoint():
    return {
        "servers": [{"name": s.name, "label": s.label} for s in SERVERS.values()],
        "languages": [{"code": code, "name": name} for code, name in LANGUAGES.items()],
        "default_lang": DEFAULT_TARGET_LANG,
        "default_model": TRANSLATE_GEMMA_MODEL,
    }


@app.get("/models")
async def models_endpoint():
    gemini = {"backend": "gemini", "id": GEMINI_MODEL, "name": f"Gemini ({GEMINI_MODEL})"} if client else None
    try:
        local = await asyncio.to_thread(list_models)
        error = None
    except (requests.RequestException, ValueError) as e:
        local, error = [], f"llama-swap is unreachable: {e}"
    return {"local": [{"backend": "llama-swap", **m} for m in local], "local_error": error, "gemini": gemini}


@app.get("/library/{server_name}")
async def library_endpoint(server_name: str, lang: str = DEFAULT_TARGET_LANG, refresh: bool = False):
    lang = LibraryQuery(lang=lang).lang
    titles, episodes_by_show = await _library(server_name, refresh)
    active = jobs.active_targets(DB_FILE)
    result = []
    for title in titles:
        entry = {"id": title.id, "kind": title.kind, "name": title.name, "year": title.year,
                 "poster": f"/poster/{server_name}/{title.id}"}
        if title.kind == "movie":
            entry["status"] = _subtitle_status(title.path, lang, active)
        else:
            statuses = [_subtitle_status(e.path, lang, active) for e in episodes_by_show.get(title.id, [])]
            available = [s for s in statuses if s != "unavailable"]
            done = statuses.count("done")
            entry.update(done=done, total=len(available))
            if "running" in statuses:
                entry["status"] = "running"
            elif not available:
                entry["status"] = "unavailable"
            elif done == len(available):
                entry["status"] = "done"
            else:
                entry["status"] = "partial" if done else "missing"
        result.append(entry)
    return {"server": server_name, "lang": lang, "titles": result}


@app.get("/library/{server_name}/shows/{show_id}")
async def show_endpoint(server_name: str, show_id: str, lang: str = DEFAULT_TARGET_LANG):
    lang = LibraryQuery(lang=lang).lang
    server = _server(server_name)
    try:
        episodes = await asyncio.to_thread(server.episodes, show_id)
    except MediaServerError as e:
        raise HTTPException(502, str(e))
    active = jobs.active_targets(DB_FILE)
    seasons: dict[int, list] = {}
    for e in episodes:
        seasons.setdefault(e.season, []).append({
            "id": e.id, "number": e.number, "name": e.name,
            "status": _subtitle_status(e.path, lang, active),
        })
    return {"seasons": [{"season": n, "episodes": eps} for n, eps in sorted(seasons.items())]}


@app.get("/poster/{server_name}/{item_id}")
async def poster_endpoint(server_name: str, item_id: str):
    server = _server(server_name)
    try:
        content, content_type = await asyncio.to_thread(server.poster, item_id)
    except MediaServerError:
        raise HTTPException(404, "No poster")
    return Response(content, media_type=content_type, headers={"Cache-Control": "max-age=86400"})


class JobItem(BaseModel):
    kind: Literal["movie", "show", "episode"]
    id: str
    show_id: str | None = None


class JobRequest(BaseModel):
    server: str
    op: Literal["translate", "shift", "sync"] = "translate"
    lang: LangCode = DEFAULT_TARGET_LANG
    backend: Literal["llama-swap", "gemini"] = "llama-swap"
    model: str = TRANSLATE_GEMMA_MODEL
    overwrite: bool = False
    offset_ms: int = 0
    priority: Literal["auto", "srt", "audio"] = "auto"
    items: list[JobItem] = Field(min_length=1)


@app.post("/jobs")
async def create_jobs(req: JobRequest):
    """Queue one job per selected movie or episode (a show expands to every
    episode that has a file)."""
    if req.op == "translate" and req.backend == "gemini" and client is None:
        raise HTTPException(400, "Gemini is not configured (GEMINI_API_KEY is empty)")
    server = _server(req.server)
    titles, episodes_by_show = await _library(req.server)
    by_id = {t.id: t for t in titles}
    targets, unavailable = [], []   # (item_id, title, path)
    for item in req.items:
        if item.kind == "movie":
            title = by_id.get(item.id)
            name = title.name if title else item.id
            targets.append((item.id, name, title.path if title else None))
        else:
            show_id = item.id if item.kind == "show" else item.show_id
            show = by_id.get(show_id)
            episodes = episodes_by_show.get(show_id) or await asyncio.to_thread(server.episodes, show_id)
            for e in episodes:
                if item.kind == "episode" and e.id != item.id:
                    continue
                label = f"{show.name if show else show_id} S{e.season:02d}E{e.number:02d}"
                targets.append((e.id, label, e.path))
    created = existing = 0
    for item_id, name, path in targets:
        if not path:
            unavailable.append(name)
            continue
        _, is_new = jobs.add(DB_FILE, op=req.op, video_path=path, target_lang=req.lang, title=name,
                             server=req.server, item_id=item_id, backend=req.backend, model=req.model,
                             overwrite=req.overwrite,
                             params={"offset_ms": req.offset_ms, "priority": req.priority})
        created += is_new
        existing += not is_new
    job_wakeup.set()
    return {"created": created, "already_queued": existing, "unavailable": unavailable}


@app.get("/jobs")
def list_jobs(limit: int = 100):
    return {"jobs": jobs.recent(DB_FILE, limit)}


@app.delete("/jobs/finished")
def clear_finished_jobs():
    jobs.clear_finished(DB_FILE)
    return {"status": "cleared"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
