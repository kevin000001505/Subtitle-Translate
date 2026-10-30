# Graph Report - /home/kevin/ai-subtitle-service  (2026-04-23)

## Corpus Check
- Corpus is ~8,959 words - fits in a single context window. You may not need a graph.

## Summary
- 155 nodes · 264 edges · 58 communities detected
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Structure Signals
- Entity graph basis: 148 non-file, non-concept node(s)
- Weakly connected components: 22
- Singleton components: 12
- Isolated nodes: 12
- Largest component: 54 node(s) (36% of the entity graph basis)
- Low-cohesion communities: 1
- Largest low-cohesion community: 36 node(s) (cohesion 0.13)

## Workspace Bridges
1. `CLAUDE.md` - connects `API Endpoints`, `Frontend UI`, `Graphify`, `Running The Service`; home: `Claude`; degree 6; score 536.67
  source files: `/home/kevin/ai-subtitle-service/AGENTS.md`
2. `API Endpoints` - connects `Auto Sync Sync`, `Claude`, `Subtitle Translation Translate`, `Timing Shift Shift`; home: `API Endpoints`; degree 4; score 227
  source files: `/home/kevin/ai-subtitle-service/AGENTS.md`
3. `API Endpoints` - connects `Auto Sync Sync \(16\)`, `Subtitle Translation Translate \(19\)`, `Timing Shift Shift \(20\)`; home: `API Endpoints \(15\)`; degree 3; score 216
  source files: `/home/kevin/ai-subtitle-service/CLAUDE.md`
4. `OpenSubtitlesClient` - connects `Full Featured Python Client For The Open Subtitles Rest API V1 Covers Every Endpoint Group Authentication Login Logout Subtitles Search Download Request Download URL Features Search Features Movies Tv Shows Discover Popular Featur`, `User Agent`; home: `Subtitle Fetch`; degree 35; score 1151.53
  source files: `/home/kevin/ai-subtitle-service/subtitle\_fetch.py`
5. `TranslateGemmaClient` - connects `Init`, `Translate`; home: `Translate Gemma Client`; degree 2; score 145
  source files: `/home/kevin/ai-subtitle-service/translate\_gemma.py`
6. `process\_movie\(\)` - connects `Main Entry Point Handles Both Single Files And Series Folders If Input Path Is A File Translate Its Subtitle If Input Path Is A Folder Find All En Srt Mkv And Mp4 Files And Translate Their Subtitles`; home: `Claude`; degree 7; score 93.17
  source files: `/home/kevin/ai-subtitle-service/main.py`

## God Nodes
1. `OpenSubtitlesClient` - 36 edges
2. `.\_get\(\)` - 20 edges
3. `.\_build\_headers\(\)` - 9 edges
4. `.\_raise\_for\_status\(\)` - 9 edges
5. `loadLevel\(\)` - 9 edges
6. `.\_url\(\)` - 8 edges
7. `CLAUDE.md` - 8 edges
8. `process\_movie\(\)` - 8 edges
9. `translate\_single\_subtitle\(\)` - 8 edges
10. `esc\(\)` - 7 edges

## Surprising Connections
- None detected - all connections are within the same source files.

## Semantic Anomalies
- **[HIGH] Bridge node** - OpenSubtitlesClient bridges Subtitle Fetch and User Agent, Full Featured Python Client For The Open Subtitles Rest API V1 Covers Every Endpoint Group Authentication Login Logout Subtitles Search Download Request Download URL Features Search Features Movies Tv Shows Discover Popular Featur.
  _High betweenness centrality \(1096.531\) across 3 communities makes this node a likely dependency chokepoint._
- **[HIGH] Bridge node** - CLAUDE.md bridges Claude and Agents Markdown, Running The Service, API Endpoints, Frontend UI, Graphify.
  _High betweenness centrality \(490.667\) across 6 communities makes this node a likely dependency chokepoint._
- **[HIGH] Bridge node** - API Endpoints bridges API Endpoints \(15\) and Claude, Subtitle Translation Translate \(19\), Timing Shift Shift \(20\), Auto Sync Sync \(16\).
  _High betweenness centrality \(183.000\) across 5 communities makes this node a likely dependency chokepoint._
- **[HIGH] Low-cohesion community** - Subtitle Fetch is weakly connected for its size.
  _Cohesion score 0.13 across 36 nodes suggests this community may mix unrelated responsibilities._

## Communities

### Community 0 - "Subtitle Fetch"
Cohesion (entity basis within full-graph community): 0.13
Nodes (36): OpenSubtitlesClient, .ai\_transcribe\(\), .ai\_translate\(\), .\_build\_headers\(\), .compute\_movie\_hash\(\), .\_delete\(\), .detect\_language\_audio\(\), .detect\_language\_text\(\) (+28 more)

### Community 1 - "Claude"
Cohesion (entity basis within full-graph community): 0.06
Nodes (29): Architecture, CLAUDE.md, Key Configuration Constants \(top of main.py\), Architecture, Key Configuration Constants \(top of main.py\), add\_to\_queue\(\), extract\_english\_sub\_from\_mkv\(\), find\_or\_extract\_english\_sub\(\) (+21 more)

### Community 2 - "Frontend App"
Cohesion (entity basis within full-graph community): 0.09
Nodes (30): addStatusRow\(\), addSyncRetryBtn\(\), bindEvents\(\), buildExtra\(\), esc\(\), finishSyncCard\(\), folderNameOf\(\), init\(\) (+22 more)

### Community 3 - "Agents Markdown"
Cohesion (entity basis within full-graph community): n/a
Nodes (0): 

### Community 4 - "API Endpoints"
Cohesion (entity basis within full-graph community): 1
Nodes (1): API Endpoints

### Community 5 - "Auto Sync Sync"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Auto-Sync \(/sync\)

### Community 6 - "Frontend UI"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Frontend UI

### Community 7 - "Graphify"
Cohesion (entity basis within full-graph community): 1
Nodes (1): graphify

### Community 8 - "Running The Service"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Running the Service

### Community 9 - "Subtitle Translation Translate"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Subtitle Translation \(/translate\)

### Community 10 - "Timing Shift Shift"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Timing Shift \(/shift\)

### Community 11 - "Compute Badges"
Cohesion (entity basis within full-graph community): 1
Nodes (1): computeBadges\(\)

### Community 12 - "File Icon"
Cohesion (entity basis within full-graph community): 1
Nodes (1): fileIcon\(\)

### Community 13 - "Select Priority"
Cohesion (entity basis within full-graph community): 1
Nodes (1): selectPriority\(\)

### Community 14 - "Update Sync Card"
Cohesion (entity basis within full-graph community): 1
Nodes (1): updateSyncCard\(\)

### Community 15 - "API Endpoints \(15\)"
Cohesion (entity basis within full-graph community): 1
Nodes (1): API Endpoints

### Community 16 - "Auto Sync Sync \(16\)"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Auto-Sync \(/sync\)

### Community 17 - "Frontend UI \(17\)"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Frontend UI

### Community 18 - "Running The Service \(18\)"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Running the Service

### Community 19 - "Subtitle Translation Translate \(19\)"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Subtitle Translation \(/translate\)

### Community 20 - "Timing Shift Shift \(20\)"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Timing Shift \(/shift\)

### Community 21 - "Extract English Subtitle From A Specific Mkv File Returns The Path To The Extracted En Srt File Or None If Failed"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Extract English subtitle from a specific MKV file. Returns the path to the extracted .en.srt file, or None if failed.

### Community 22 - "Look For An Existing En Srt In The Folder If Not Found Use The First Mkv File And Extract Its English Subtitle Returns The Path To The En Srt File Or None If Not Possible"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Look for an existing .en.srt in the folder. If not found, use the first MKV file and extract its English subtitle. Returns the path to the .en.srt file, or None if not possible.

### Community 23 - "List Directory Endpoint"
Cohesion (entity basis within full-graph community): 1
Nodes (1): list\_directory\_endpoint\(\)

### Community 24 - "List Contents Of A Directory For The Browser UI Returns JSON With Items Array Containing Name Type Path Objects Type Is Either Folder Or File"
Cohesion (entity basis within full-graph community): 1
Nodes (1): List contents of a directory for the browser UI. Returns JSON with items array containing {name, type, path} objects. type is either "folder" or "file".

### Community 25 - "Parse Lines Of The Form ID Translated Text Supports Multi Line Translations Stops Each Entry At The Next ID On Its Own Line So Trailing Prose After The Final Entry Is Bounded By Newline Breaks"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Parse lines of the form \`\[ID\] translated text\`. Supports multi-line translations; stops each entry at the next \`\[ID\]\` on its own line, so trailing prose after the final entry is bounded by newline-breaks.

### Community 26 - "Main Entry Point Handles Both Single Files And Series Folders If Input Path Is A File Translate Its Subtitle If Input Path Is A Folder Find All En Srt Mkv And Mp4 Files And Translate Their Subtitles"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Main entry point. Handles both single files and series folders. - If input\_path is a file: translate its subtitle - If input\_path is a folder: find all .en.srt, MKV, and MP4 files and translate their subtitles

### Community 27 - "Run Ffsubsync For All Matching Files In Folder Path Updating Sync Jobs Job ID With Per File Progress As Each File Completes"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Run ffsubsync for all matching files in folder\_path, updating sync\_jobs\[job\_id\] with per-file progress as each file completes.

### Community 28 - "Shift Request"
Cohesion (entity basis within full-graph community): 1
Nodes (1): ShiftRequest

### Community 29 - "Subtitle Request"
Cohesion (entity basis within full-graph community): 1
Nodes (1): SubtitleRequest

### Community 30 - "Start An Async Sync Job Returns A Job ID Immediately Poll Get Sync Progress Job ID Sse For Live Per File Progress"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Start an async sync job. Returns a job\_id immediately; poll GET /sync/progress/{job\_id} \(SSE\) for live per-file progress.

### Community 31 - "Sync Progress"
Cohesion (entity basis within full-graph community): 1
Nodes (1): sync\_progress\(\)

### Community 32 - "Sse Endpoint Streams Job State Once Per Second Until Done Error"
Cohesion (entity basis within full-graph community): 1
Nodes (1): SSE endpoint — streams job state once per second until done/error.

### Community 33 - "Sync Request"
Cohesion (entity basis within full-graph community): 1
Nodes (1): SyncRequest

### Community 34 - "Translate A Single Subtitle File To Chinese Returns True If Successful False Otherwise"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Translate a single subtitle file to Chinese. Returns True if successful, False otherwise.

### Community 35 - "Compute The Open Subtitles Movie Hash For A Local Video File The Hash Is A 16 Character Hex String Derived From The File Size And The First Last 64 Kb Of The File Content Args File Path Absolute Or Relative Path To The Video File Returns 16 Character"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Compute the OpenSubtitles movie hash for a local video file. The hash is a 16-character hex string derived from the file size and the first/last 64 KB of the file content. Args: file\_path: Absolute or relative path to the video file. Returns: 16-character

### Community 36 - "Detect The Language Of A Subtitle File Post Ai Detect Language Text At Least 1 Ai Credit Is Required Args File Path To The Subtitle File Str Or Raw Bytes Returns Dict With Data Format Data Type Data Language W3 C Name Native Iso 639 1 Iso"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Detect the language of a subtitle file \(POST /ai/detect\_language\_text\). At least 1 AI credit is required. Args: file: Path to the subtitle file \(str\) or raw bytes. Returns: dict with data.format, data.type, data.language \(W3C, name, native, ISO\_639\_1, ISO\_

### Community 37 - "Get The Status Of A Transcription Job Get Ai Transcribe Correlation ID Status Values Created Pending Completed Error Timeout Args Correlation ID The Correlation ID Returned By Ai Transcribe Returns Dict With Job Status Details"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Get the status of a transcription job \(GET /ai/transcribe/{correlation\_id}\). Status values: CREATED | PENDING | COMPLETED | ERROR | TIMEOUT Args: correlation\_id: The correlation\_id returned by ai\_transcribe\(\). Returns: dict with job status details.

### Community 38 - "Get Available Ai Transcription Ap Is And Their Supported Languages Get Ai Info Transcription No Authentication Required Returns Dict With Data Containing Name Display Name Description Pricing Reliability Price Languages Supported"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Get available AI transcription APIs and their supported languages \(GET /ai/info/transcription\). No authentication required. Returns: dict with data\[\] containing name, display\_name, description, pricing, reliability, price, languages\_supported\[\].

### Community 39 - "Get The Status Of A Translation Job Get Ai Translate Correlation ID Status Values Created Pending Completed Error Timeout Args Correlation ID The Correlation ID Returned By Ai Translate Returns Dict With Job Status Details"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Get the status of a translation job \(GET /ai/translate/{correlation\_id}\). Status values: CREATED | PENDING | COMPLETED | ERROR | TIMEOUT Args: correlation\_id: The correlation\_id returned by ai\_translate\(\). Returns: dict with job status details.

### Community 40 - "Get Available Ai Translation Ap Is And Their Supported Languages Get Ai Info Translation No Authentication Required Returns Dict With Data Containing Name Display Name Description Pricing Reliability Price Languages Supported"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Get available AI translation APIs and their supported languages \(GET /ai/info/translation\). No authentication required. Returns: dict with data\[\] containing name, display\_name, description, pricing, reliability, price, languages\_supported\[\].

### Community 41 - "Get Available Credit Packages With Checkout Ur Ls Get Ai Credits Buy Returns Dict With Data Containing Name Value Discount Percent Checkout URL"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Get available credit packages with checkout URLs \(GET /ai/credits/buy\). Returns: dict with data\[\] containing name, value, discount\_percent, checkout\_url.

### Community 42 - "Get The Status Of A Detect Language Audio Job Get Ai Detect Language Audio Correlation ID Status Values Created Pending Completed Error Timeout Args Correlation ID The Correlation ID Returned By Detect Language Audio Returns Dict With"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Get the status of a detect-language-audio job \(GET /ai/detect\_language\_audio/{correlation\_id}\). Status values: CREATED | PENDING | COMPLETED | ERROR | TIMEOUT Args: correlation\_id: The correlation\_id returned by detect\_language\_audio\(\). Returns: dict with

### Community 43 - "Get All Supported Languages With Their Codes Get Infos Languages Returns Dict With Data Containing Language Code And Language Name"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Get all supported languages with their codes \(GET /infos/languages\). Returns: dict with data\[\] containing language\_code and language\_name.

### Community 44 - "Get All Supported Subtitle File Formats Get Infos Formats Returns Dict With A List Of Supported Format Strings"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Get all supported subtitle file formats \(GET /infos/formats\). Returns: dict with a list of supported format strings.

### Community 45 - "Get The Ai Credits Balance For The Authenticated User Get Ai Credits Requires API Key And Authorization Header Returns Dict With Data Credits Integer"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Get the AI credits balance for the authenticated user \(GET /ai/credits\). Requires Api-Key and Authorization header. Returns: dict with data.credits \(integer\).

### Community 46 - "Get Information About The Currently Authenticated User Get Infos User Requires Bearer Token Login First Returns Dict With Data Containing Allowed Downloads Level User ID Vip Downloads Count Remaining Downloads"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Get information about the currently authenticated user \(GET /infos/user\). Requires bearer token \(login first\). Returns: dict with data containing: allowed\_downloads, level, user\_id, vip, downloads\_count, remaining\_downloads.

### Community 47 - "Extract Metadata From A Video Filename Using The Guess It Library Get Utilities Guessit Works For Both Movies And Tv Show Episodes Args Filename The Video Filename To Analyse E G Movie 2023 1080p Mkv Returns Dict With Keys Title Year Langua"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Extract metadata from a video filename using the GuessIt library \(GET /utilities/guessit\). Works for both movies and TV show episodes. Args: filename: The video filename to analyse \(e.g. "movie.2023.1080p.mkv"\). Returns: dict with keys: title, year, langua

### Community 48 - "Create A Jwt Token To Authenticate A User Automatically Stores The Returned Token For Subsequent Requests Also Updates Base URL If The API Returns A Different Base URL E G Vip API Opensubtitles Com For Vip Users Rate Limit 1 Req S 10 Req Min 30 Re"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Create a JWT token to authenticate a user. Automatically stores the returned token for subsequent requests. Also updates base\_url if the API returns a different base\_url \(e.g. vip-api.opensubtitles.com for VIP users\). Rate limit: 1 req/s, 10 req/min, 30 re

### Community 49 - "Destroy The Current User Session Token Clears The Locally Stored Token After A Successful Call Returns Dict With Keys Message Status"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Destroy the current user session/token. Clears the locally stored token after a successful call. Returns: dict with keys: message, status.

### Community 50 - "Full Featured Python Client For The Open Subtitles Rest API V1 Covers Every Endpoint Group Authentication Login Logout Subtitles Search Download Request Download URL Features Search Features Movies Tv Shows Discover Popular Featur"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Full-featured Python client for the OpenSubtitles REST API v1. Covers every endpoint group: - Authentication : login, logout - Subtitles : search - Download : request download URL - Features : search features \(movies / TV shows\) - Discover : popular featur

### Community 51 - "User Agent"
Cohesion (entity basis within full-graph community): 1
Nodes (1): .\_user\_agent\(\)

### Community 52 - "Raised When The API Returns An Error Response"
Cohesion (entity basis within full-graph community): 1
Nodes (1): Raised when the API returns an error response.

### Community 53 - "Open Subtitles API Python Client A Comprehensive Python Class Wrapping All Endpoints Of The Open Subtitles Rest API V1 API Documentation HTTPS Opensubtitles Stoplight IO Docs Opensubtitles API Base URL HTTPS API Opensu"
Cohesion (entity basis within full-graph community): 1
Nodes (1): OpenSubtitles API Python Client ================================ A comprehensive Python class wrapping all endpoints of the OpenSubtitles REST API v1. API Documentation: https://opensubtitles.stoplight.io/docs/opensubtitles-api Base URL: https://api.opensu

### Community 54 - "Translate Gemma Py"
Cohesion (entity basis within full-graph community): n/a
Nodes (0): 

### Community 55 - "Translate Gemma Client"
Cohesion (entity basis within full-graph community): 1
Nodes (1): TranslateGemmaClient

### Community 56 - "Init"
Cohesion (entity basis within full-graph community): 1
Nodes (1): .\_\_init\_\_\(\)

### Community 57 - "Translate"
Cohesion (entity basis within full-graph community): 1
Nodes (1): .translate\(\)

## Knowledge Gaps
- **77 weakly connected node(s):** `computeBadges\(\)`, `fileIcon\(\)`, `renderBreadcrumb\(\)`, `navBlockHTML\(\)`, `navigateInto\(\)` (+72 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Agents Markdown`** (1 nodes): `AGENTS.md`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `API Endpoints`** (1 nodes): `API Endpoints`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Auto Sync Sync`** (1 nodes): `Auto-Sync \(/sync\)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Frontend UI`** (1 nodes): `Frontend UI`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Graphify`** (1 nodes): `graphify`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Running The Service`** (1 nodes): `Running the Service`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Subtitle Translation Translate`** (1 nodes): `Subtitle Translation \(/translate\)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Timing Shift Shift`** (1 nodes): `Timing Shift \(/shift\)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Compute Badges`** (1 nodes): `computeBadges\(\)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `File Icon`** (1 nodes): `fileIcon\(\)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Select Priority`** (1 nodes): `selectPriority\(\)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Update Sync Card`** (1 nodes): `updateSyncCard\(\)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `API Endpoints \(15\)`** (1 nodes): `API Endpoints`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Auto Sync Sync \(16\)`** (1 nodes): `Auto-Sync \(/sync\)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Frontend UI \(17\)`** (1 nodes): `Frontend UI`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Running The Service \(18\)`** (1 nodes): `Running the Service`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Subtitle Translation Translate \(19\)`** (1 nodes): `Subtitle Translation \(/translate\)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Timing Shift Shift \(20\)`** (1 nodes): `Timing Shift \(/shift\)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Extract English Subtitle From A Specific Mkv File Returns The Path To The Extracted En Srt File Or None If Failed`** (1 nodes): `Extract English subtitle from a specific MKV file. Returns the path to the extracted .en.srt file, or None if failed.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Look For An Existing En Srt In The Folder If Not Found Use The First Mkv File And Extract Its English Subtitle Returns The Path To The En Srt File Or None If Not Possible`** (1 nodes): `Look for an existing .en.srt in the folder. If not found, use the first MKV file and extract its English subtitle. Returns the path to the .en.srt file, or None if not possible.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `List Directory Endpoint`** (1 nodes): `list\_directory\_endpoint\(\)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `List Contents Of A Directory For The Browser UI Returns JSON With Items Array Containing Name Type Path Objects Type Is Either Folder Or File`** (1 nodes): `List contents of a directory for the browser UI. Returns JSON with items array containing {name, type, path} objects. type is either "folder" or "file".`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Parse Lines Of The Form ID Translated Text Supports Multi Line Translations Stops Each Entry At The Next ID On Its Own Line So Trailing Prose After The Final Entry Is Bounded By Newline Breaks`** (1 nodes): `Parse lines of the form \`\[ID\] translated text\`. Supports multi-line translations; stops each entry at the next \`\[ID\]\` on its own line, so trailing prose after the final entry is bounded by newline-breaks.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Main Entry Point Handles Both Single Files And Series Folders If Input Path Is A File Translate Its Subtitle If Input Path Is A Folder Find All En Srt Mkv And Mp4 Files And Translate Their Subtitles`** (1 nodes): `Main entry point. Handles both single files and series folders. - If input\_path is a file: translate its subtitle - If input\_path is a folder: find all .en.srt, MKV, and MP4 files and translate their subtitles`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Run Ffsubsync For All Matching Files In Folder Path Updating Sync Jobs Job ID With Per File Progress As Each File Completes`** (1 nodes): `Run ffsubsync for all matching files in folder\_path, updating sync\_jobs\[job\_id\] with per-file progress as each file completes.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Shift Request`** (1 nodes): `ShiftRequest`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Subtitle Request`** (1 nodes): `SubtitleRequest`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Start An Async Sync Job Returns A Job ID Immediately Poll Get Sync Progress Job ID Sse For Live Per File Progress`** (1 nodes): `Start an async sync job. Returns a job\_id immediately; poll GET /sync/progress/{job\_id} \(SSE\) for live per-file progress.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Sync Progress`** (1 nodes): `sync\_progress\(\)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Sse Endpoint Streams Job State Once Per Second Until Done Error`** (1 nodes): `SSE endpoint — streams job state once per second until done/error.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Sync Request`** (1 nodes): `SyncRequest`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Translate A Single Subtitle File To Chinese Returns True If Successful False Otherwise`** (1 nodes): `Translate a single subtitle file to Chinese. Returns True if successful, False otherwise.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Compute The Open Subtitles Movie Hash For A Local Video File The Hash Is A 16 Character Hex String Derived From The File Size And The First Last 64 Kb Of The File Content Args File Path Absolute Or Relative Path To The Video File Returns 16 Character`** (1 nodes): `Compute the OpenSubtitles movie hash for a local video file. The hash is a 16-character hex string derived from the file size and the first/last 64 KB of the file content. Args: file\_path: Absolute or relative path to the video file. Returns: 16-character`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Detect The Language Of A Subtitle File Post Ai Detect Language Text At Least 1 Ai Credit Is Required Args File Path To The Subtitle File Str Or Raw Bytes Returns Dict With Data Format Data Type Data Language W3 C Name Native Iso 639 1 Iso`** (1 nodes): `Detect the language of a subtitle file \(POST /ai/detect\_language\_text\). At least 1 AI credit is required. Args: file: Path to the subtitle file \(str\) or raw bytes. Returns: dict with data.format, data.type, data.language \(W3C, name, native, ISO\_639\_1, ISO\_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Get The Status Of A Transcription Job Get Ai Transcribe Correlation ID Status Values Created Pending Completed Error Timeout Args Correlation ID The Correlation ID Returned By Ai Transcribe Returns Dict With Job Status Details`** (1 nodes): `Get the status of a transcription job \(GET /ai/transcribe/{correlation\_id}\). Status values: CREATED | PENDING | COMPLETED | ERROR | TIMEOUT Args: correlation\_id: The correlation\_id returned by ai\_transcribe\(\). Returns: dict with job status details.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Get Available Ai Transcription Ap Is And Their Supported Languages Get Ai Info Transcription No Authentication Required Returns Dict With Data Containing Name Display Name Description Pricing Reliability Price Languages Supported`** (1 nodes): `Get available AI transcription APIs and their supported languages \(GET /ai/info/transcription\). No authentication required. Returns: dict with data\[\] containing name, display\_name, description, pricing, reliability, price, languages\_supported\[\].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Get The Status Of A Translation Job Get Ai Translate Correlation ID Status Values Created Pending Completed Error Timeout Args Correlation ID The Correlation ID Returned By Ai Translate Returns Dict With Job Status Details`** (1 nodes): `Get the status of a translation job \(GET /ai/translate/{correlation\_id}\). Status values: CREATED | PENDING | COMPLETED | ERROR | TIMEOUT Args: correlation\_id: The correlation\_id returned by ai\_translate\(\). Returns: dict with job status details.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Get Available Ai Translation Ap Is And Their Supported Languages Get Ai Info Translation No Authentication Required Returns Dict With Data Containing Name Display Name Description Pricing Reliability Price Languages Supported`** (1 nodes): `Get available AI translation APIs and their supported languages \(GET /ai/info/translation\). No authentication required. Returns: dict with data\[\] containing name, display\_name, description, pricing, reliability, price, languages\_supported\[\].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Get Available Credit Packages With Checkout Ur Ls Get Ai Credits Buy Returns Dict With Data Containing Name Value Discount Percent Checkout URL`** (1 nodes): `Get available credit packages with checkout URLs \(GET /ai/credits/buy\). Returns: dict with data\[\] containing name, value, discount\_percent, checkout\_url.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Get The Status Of A Detect Language Audio Job Get Ai Detect Language Audio Correlation ID Status Values Created Pending Completed Error Timeout Args Correlation ID The Correlation ID Returned By Detect Language Audio Returns Dict With`** (1 nodes): `Get the status of a detect-language-audio job \(GET /ai/detect\_language\_audio/{correlation\_id}\). Status values: CREATED | PENDING | COMPLETED | ERROR | TIMEOUT Args: correlation\_id: The correlation\_id returned by detect\_language\_audio\(\). Returns: dict with`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Get All Supported Languages With Their Codes Get Infos Languages Returns Dict With Data Containing Language Code And Language Name`** (1 nodes): `Get all supported languages with their codes \(GET /infos/languages\). Returns: dict with data\[\] containing language\_code and language\_name.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Get All Supported Subtitle File Formats Get Infos Formats Returns Dict With A List Of Supported Format Strings`** (1 nodes): `Get all supported subtitle file formats \(GET /infos/formats\). Returns: dict with a list of supported format strings.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Get The Ai Credits Balance For The Authenticated User Get Ai Credits Requires API Key And Authorization Header Returns Dict With Data Credits Integer`** (1 nodes): `Get the AI credits balance for the authenticated user \(GET /ai/credits\). Requires Api-Key and Authorization header. Returns: dict with data.credits \(integer\).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Get Information About The Currently Authenticated User Get Infos User Requires Bearer Token Login First Returns Dict With Data Containing Allowed Downloads Level User ID Vip Downloads Count Remaining Downloads`** (1 nodes): `Get information about the currently authenticated user \(GET /infos/user\). Requires bearer token \(login first\). Returns: dict with data containing: allowed\_downloads, level, user\_id, vip, downloads\_count, remaining\_downloads.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Extract Metadata From A Video Filename Using The Guess It Library Get Utilities Guessit Works For Both Movies And Tv Show Episodes Args Filename The Video Filename To Analyse E G Movie 2023 1080p Mkv Returns Dict With Keys Title Year Langua`** (1 nodes): `Extract metadata from a video filename using the GuessIt library \(GET /utilities/guessit\). Works for both movies and TV show episodes. Args: filename: The video filename to analyse \(e.g. "movie.2023.1080p.mkv"\). Returns: dict with keys: title, year, langua`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Create A Jwt Token To Authenticate A User Automatically Stores The Returned Token For Subsequent Requests Also Updates Base URL If The API Returns A Different Base URL E G Vip API Opensubtitles Com For Vip Users Rate Limit 1 Req S 10 Req Min 30 Re`** (1 nodes): `Create a JWT token to authenticate a user. Automatically stores the returned token for subsequent requests. Also updates base\_url if the API returns a different base\_url \(e.g. vip-api.opensubtitles.com for VIP users\). Rate limit: 1 req/s, 10 req/min, 30 re`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Destroy The Current User Session Token Clears The Locally Stored Token After A Successful Call Returns Dict With Keys Message Status`** (1 nodes): `Destroy the current user session/token. Clears the locally stored token after a successful call. Returns: dict with keys: message, status.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Full Featured Python Client For The Open Subtitles Rest API V1 Covers Every Endpoint Group Authentication Login Logout Subtitles Search Download Request Download URL Features Search Features Movies Tv Shows Discover Popular Featur`** (1 nodes): `Full-featured Python client for the OpenSubtitles REST API v1. Covers every endpoint group: - Authentication : login, logout - Subtitles : search - Download : request download URL - Features : search features \(movies / TV shows\) - Discover : popular featur`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `User Agent`** (1 nodes): `.\_user\_agent\(\)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Raised When The API Returns An Error Response`** (1 nodes): `Raised when the API returns an error response.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Open Subtitles API Python Client A Comprehensive Python Class Wrapping All Endpoints Of The Open Subtitles Rest API V1 API Documentation HTTPS Opensubtitles Stoplight IO Docs Opensubtitles API Base URL HTTPS API Opensu`** (1 nodes): `OpenSubtitles API Python Client ================================ A comprehensive Python class wrapping all endpoints of the OpenSubtitles REST API v1. API Documentation: https://opensubtitles.stoplight.io/docs/opensubtitles-api Base URL: https://api.opensu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Translate Gemma Py`** (1 nodes): `translate\_gemma.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Translate Gemma Client`** (1 nodes): `TranslateGemmaClient`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Init`** (1 nodes): `.\_\_init\_\_\(\)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Translate`** (1 nodes): `.translate\(\)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does \`OpenSubtitlesClient\` connect \`Subtitle Fetch\` to \`User Agent\`, \`Full Featured Python Client For The Open Subtitles Rest API V1 Covers Every Endpoint Group Authentication Login Logout Subtitles Search Download Request Download URL Features Search Features Movies Tv Shows Discover Popular Featur\`?**
  _High betweenness centrality \(1096.531\) - this node is a cross-community bridge._
- **Why does \`CLAUDE.md\` connect \`Claude\` to \`Agents Markdown\`, \`Running The Service\`, \`API Endpoints\`, \`Frontend UI\`, \`Graphify\`?**
  _High betweenness centrality \(490.667\) - this node is a cross-community bridge._
- **What connects \`computeBadges\(\)\`, \`fileIcon\(\)\`, \`renderBreadcrumb\(\)\` to the rest of the system?**
  _77 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should \`Subtitle Fetch\` be split into smaller, more focused modules?**
  _Cohesion score 0.13 across 36 entity nodes - this community may mix unrelated responsibilities._
