from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
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
from translate_gemma import TranslateGemmaClient

# ================= CONFIGURATION =================
TARGET_LANG = "zh-TW"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# TranslateGemma (local llama.cpp server)
translate_gemma_client = TranslateGemmaClient()

# Which translator to use: "translatemgemma" or "gemini"
TRANSLATOR_BACKEND = os.getenv("TRANSLATOR_BACKEND", "translatemgemma")

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

class SubtitleRequest(BaseModel):
    path: str

class ShiftRequest(BaseModel):
    path: str  # can be a file or folder
    offset_ms: int = 0

class SyncRequest(BaseModel):
    path: str  # folder path
    priority: str = "auto"  # "srt", "audio", or "auto"




# ================= SYNC JOB TRACKING =================

# job_id -> { status, total, current, files: [{name, status, method?, error?}], message?, synced?, failed? }
sync_jobs: dict = {}


async def run_ffsubsync_with_progress(job_id: str, folder_path: str, priority: str):
    """
    Run ffsubsync for all matching files in folder_path, updating sync_jobs[job_id]
    with per-file progress as each file completes.
    """
    job = sync_jobs[job_id]

    if not os.path.isdir(folder_path):
        job["status"] = "error"
        job["message"] = f"Folder not found: {folder_path}"
        return

    video_files = sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith((".mkv", ".mp4"))
    )
    if not video_files:
        job["status"] = "error"
        job["message"] = "No video files (.mkv/.mp4) found in folder"
        return

    # Build work list — only files that have a matching zh-TW.srt
    files = []
    for video_file in video_files:
        base   = os.path.splitext(video_file)[0]
        zh_srt = os.path.join(folder_path, f"{base}.zh-TW.srt")
        if os.path.exists(zh_srt):
            files.append({"name": video_file, "status": "pending"})
        else:
            logger.warning(f"SYNC: No zh-TW.srt for {video_file}, skipping")

    if not files:
        job["status"] = "error"
        job["message"] = "No zh-TW.srt files found to sync"
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

        base       = os.path.splitext(video_file)[0]
        video_path = os.path.join(folder_path, video_file)
        zh_srt     = os.path.join(folder_path, f"{base}.zh-TW.srt")
        en_srt     = os.path.join(folder_path, f"{base}.en.srt")

        use_srt_ref = (priority in ("srt", "auto")) and os.path.exists(en_srt)
        if use_srt_ref:
            cmd = ["ffsubsync", en_srt, "-i", zh_srt, "-o", zh_srt, "--srt"]
            file_entry["method"] = "srt"
            logger.info(f"SYNC: SRT-ref mode for {video_file}")
        else:
            cmd = ["ffsubsync", video_path, "-i", zh_srt, "-o", zh_srt]
            file_entry["method"] = "audio"
            logger.info(f"SYNC: Audio-ref mode for {video_file}")

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
            job["status"]  = "error"
            job["message"] = "ffsubsync is not installed"
            return
        except Exception as e:
            file_entry["status"] = "failed"
            file_entry["error"]  = str(e)
            failed.append({"file": video_file, "error": str(e)})
            logger.error(f"SYNC: Error for {video_file}: {e}")

        job["current"] += 1

    job["status"]  = "done"
    job["synced"]  = synced
    job["failed"]  = failed
    job["message"] = f"Synced {len(synced)} file(s), {len(failed)} failed"
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
    conn.commit()
    conn.close()

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

def add_to_queue(path):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM job_queue WHERE path = ? AND status = 'PENDING'", (path,))
    if c.fetchone() is None:
        c.execute("INSERT INTO job_queue (path, status, added_at) VALUES (?, 'PENDING', ?)", (path, time.time()))
        conn.commit()
        logger.info(f"QUEUE: Added {path} to queue.")
    conn.close()

def pop_from_queue():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, path FROM job_queue WHERE status = 'PENDING' ORDER BY added_at ASC LIMIT 1")
    row = c.fetchone()
    if row:
        c.execute("UPDATE job_queue SET status = 'PROCESSING' WHERE id = ?", (row[0],))
        conn.commit()
        conn.close()
        return row[1]
    conn.close()
    return None

def mark_done_in_queue(path):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM job_queue WHERE path = ?", (path,))
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
        base_name = os.path.splitext(mkv_path)[0]
        out_srt = f"{base_name}.en.srt"

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
        if f.lower().endswith(".en.srt"):
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

def parse_gemini_response(response_text):
    """
    NEW PARSER: Uses Regex Split to handle multi-line subtitles correctly.
    """
    updates = {}
    parts = re.split(r'\[(\d+)\]', response_text)
    for i in range(1, len(parts), 2):
        try:
            sub_id = int(parts[i])
            text = parts[i+1].strip()
            updates[sub_id] = text
        except (ValueError, IndexError):
            continue
    return updates

async def process_single_chunk(index, batch, context_text, total_chunks, progress_counter):
    async with sem: 
        batch_text = "\n".join([f"[{idx}] {sub.text}" for idx, sub in batch])
        
        prompt = f"""
        CONTEXT: The movie is about: {context_text[:2000]}...
        TASK: Translate to {TARGET_LANG}.
        
        CRITICAL RULES:
        1. Format MUST be: [ID] Translated Text
        2. If the original has multiple lines (e.g., two speakers), your output MUST have multiple lines too.
        3. Do not merge separate dialogue lines into one.
        
        INPUT:
        {batch_text}
        """

        for attempt in range(3):
            try:
                if TRANSLATOR_BACKEND == "gemini" and client is not None:
                    response = await client.aio.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=prompt
                    )
                    log_api_call()
                    result_text = response.text
                else:
                    result_text = await translate_gemma_client.translate(batch_text, context_text)
                    if result_text is None:
                        raise Exception("TranslateGemma returned None")

                progress_counter['completed'] += 1
                current = progress_counter['completed']
                if current % 10 == 0 or current == total_chunks:
                    bar = get_progress_bar(current, total_chunks)
                    logger.info(f"PROGRESS: {bar} ({current}/{total_chunks})")

                return result_text
            except Exception as e:
                wait_time = (attempt + 1) * 2
                if attempt == 2:
                    logger.error(f"Failed chunk {index}: {e}")
                    return None
                await asyncio.sleep(wait_time)

async def translate_single_subtitle(en_sub_path: str) -> bool:
    """
    Translate a single subtitle file to Chinese.
    Returns True if successful, False otherwise.
    """
    logger.info(f"START: Processing {en_sub_path}")

    try:
        subs = await asyncio.to_thread(pysubs2.load, en_sub_path)
    except Exception as e:
        logger.error(f"Failed to load subtitle {en_sub_path}: {e}")
        return False

    chunks = get_smart_chunks(subs)
    total_chunks = len(chunks)
    
    used_today = get_calls_last_24h()
    if (used_today + total_chunks) > DAILY_LIMIT:
        logger.warning(f"LIMIT HIT: Queueing {en_sub_path}")
        add_to_queue(en_sub_path)
        return False

    logger.info(f"Translating {total_chunks} chunks (Concurrency: {CONCURRENCY_LIMIT})...")
    full_text_preview = " ".join([s.text.replace("\n", " ") for s in subs[:500]])

    tasks = []
    progress_counter = {'completed': 0}
    
    for i, batch in enumerate(chunks):
        task = process_single_chunk(i, batch, full_text_preview, total_chunks, progress_counter)
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

    # Apply timing offset to ALL subtitles so every line is shifted consistently,
    # including any lines the translation skipped or left untouched.
    if TIME_OFFSET_MS != 0:
        for sub in subs:
            sub.start = max(0, sub.start + TIME_OFFSET_MS)
            sub.end   = max(0, sub.end   + TIME_OFFSET_MS)
        logger.info(f"Applied {TIME_OFFSET_MS}ms offset to all {len(subs)} subtitles.")

    # Save
    if en_sub_path.endswith(".en.srt"):
        new_path = en_sub_path.replace(".en.srt", ".zh-TW.srt")
    elif en_sub_path.endswith(".srt"):
        base_name = en_sub_path[:-4]
        new_path = f"{base_name}(AI).zh-TW.srt"
    else:
        new_path = en_sub_path + ".zh-TW.srt"
        
    await asyncio.to_thread(subs.save, new_path, encoding="utf-8")

    try:
        os.chmod(new_path, 0o644)
        source_stat = os.stat(en_sub_path)
        os.chown(new_path, source_stat.st_uid, source_stat.st_gid)
    except:
        pass

    logger.info(f"DONE: Saved {new_path}")
    return True

async def process_movie(input_path: str):
    """
    Main entry point. Handles both single files and series folders.
    - If input_path is a file: translate its subtitle
    - If input_path is a folder: find all .en.srt, MKV, and MP4 files and translate their subtitles
    """
    if not os.path.exists(input_path):
        logger.error(f"Path not found: {input_path}")
        mark_done_in_queue(input_path)
        return

    # Check if input is a folder (series) or file
    if os.path.isdir(input_path):
        logger.info(f"SERIES: Processing folder {input_path}")
        movie_dir = input_path

        # Find all .en.srt, MKV, and MP4 files in the folder
        srt_files   = [f for f in os.listdir(movie_dir) if f.lower().endswith(".en.srt")]
        video_files = [f for f in os.listdir(movie_dir) if f.lower().endswith((".mkv", ".mp4"))]
        if not srt_files and not video_files:
            logger.warning(f"SKIP: No .en.srt or MKV/MP4 files found in {movie_dir}")
            mark_done_in_queue(input_path)
            return

        logger.info(f"SERIES: Found {len(srt_files)} .en.srt and {len(video_files)} video file(s).")

        # Process existing .en.srt files — skip individual episodes already translated
        processed_bases = set()
        for srt_file in sorted(srt_files):
            base     = srt_file[:-len(".en.srt")]
            zh_path  = os.path.join(movie_dir, f"{base}.zh-TW.srt")
            if os.path.exists(zh_path):
                logger.info(f"EPISODE: Skipping {srt_file}, zh-TW.srt already exists.")
                processed_bases.add(base.lower())
                continue
            srt_path = os.path.join(movie_dir, srt_file)
            processed_bases.add(base.lower())
            logger.info(f"EPISODE: Translating {srt_file}")
            success = await translate_single_subtitle(srt_path)
            if not success:
                logger.warning(f"EPISODE: Failed to translate {srt_path}")

        # Process video files whose translation doesn't exist yet
        for video_file in sorted(video_files):
            base    = os.path.splitext(video_file)[0]
            zh_path = os.path.join(movie_dir, f"{base}.zh-TW.srt")
            if os.path.exists(zh_path):
                logger.info(f"EPISODE: Skipping {video_file}, zh-TW.srt already exists.")
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
            success = await translate_single_subtitle(en_sub_path)
            if not success:
                logger.warning(f"EPISODE: Failed to translate {en_sub_path}")

        logger.info(f"SERIES: Completed {movie_dir}")
        log_history(input_path, "translate")
        mark_done_in_queue(input_path)
    else:
        # Single file mode — resolve to an .en.srt and translate it
        movie_dir = os.path.dirname(input_path)

        if input_path.lower().endswith(".en.srt"):
            en_sub_path = input_path
        else:
            en_sub_path = await find_or_extract_english_sub(movie_dir)

        if en_sub_path is None:
            logger.warning(f"SKIP: Could not find or extract an English subtitle in {movie_dir}")
            mark_done_in_queue(input_path)
            return

        # Skip if zh-TW already exists for this specific file
        base    = en_sub_path[:-len(".en.srt")] if en_sub_path.endswith(".en.srt") else en_sub_path[:-4]
        zh_path = f"{base}.zh-TW.srt"
        if os.path.exists(zh_path):
            logger.info(f"SKIP: {zh_path} already exists.")
            mark_done_in_queue(input_path)
            return

        success = await translate_single_subtitle(en_sub_path)
        if success:
            log_history(input_path, "translate")
        mark_done_in_queue(input_path)

# ================= SCHEDULER & APP =================

async def queue_monitor():
    while True:
        used = get_calls_last_24h()
        if (DAILY_LIMIT - used) > 100:
            next_file = pop_from_queue()
            if next_file:
                logger.info(f"MONITOR: Starting queued job: {next_file}")
                await process_movie(next_file)
        await asyncio.sleep(3600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    asyncio.create_task(queue_monitor())
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
        add_to_queue(req.path)
        return {"status": "queued_limit_reached"}
    
    background_tasks.add_task(process_movie, req.path)
    return {"status": "processing_async"}

@app.post("/shift")
async def shift_subtitle(req: ShiftRequest):
    if not os.path.exists(req.path):
        return {"status": "error", "message": "Path not found"}

    # If it's a file, shift just that file
    if os.path.isfile(req.path):
        targets = [req.path]
    # If it's a folder, shift all zh-TW.srt inside
    else:
        targets = [
            os.path.join(req.path, f)
            for f in os.listdir(req.path)
            if f.lower().endswith(".zh-tw.srt")
        ]

    if not targets:
        return {"status": "error", "message": "No zh-TW.srt files found"}

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

    job_id = str(uuid.uuid4())
    sync_jobs[job_id] = {
        "status":  "starting",
        "total":   0,
        "current": 0,
        "files":   [],
    }
    logger.info(f"SYNC: Queued job {job_id} for {req.path} (priority={req.priority})")
    background_tasks.add_task(run_ffsubsync_with_progress, job_id, req.path, req.priority)
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
