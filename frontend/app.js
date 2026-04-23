/* ============================================================
   AI Subtitle Service — frontend
   Sections:
     1. CONFIG
     2. STATE
     3. DOM shortcuts
     4. API client
     5. Folder browser
     6. Operation runner (translate / shift / sync)
     7. Sync progress (SSE)
     8. History
     9. Utilities
    10. Init
   ============================================================ */

// 1. CONFIG ────────────────────────────────────────────────────
const API_BASE   = 'http://localhost:8000';
const MEDIA_ROOT = '/media';
const MEDIA_RE   = /\.(srt|mkv|mp4|avi|mov|wav|mp3|m4a|flac|aac)$/i;
const VIDEO_RE   = /\.(mkv|mp4|avi|mov)$/i;
const AUDIO_RE   = /\.(wav|mp3|m4a|flac|aac)$/i;

const OP_META = {
    translate: { label: 'Translate',    verb: 'Translating', emoji: '✨', endpoint: '/translate' },
    shift:     { label: 'Shift Timing', verb: 'Shifting',    emoji: '⏱️', endpoint: '/shift'     },
    sync:      { label: 'Sync & Align', verb: 'Syncing',     emoji: '🔄', endpoint: '/sync'      },
};

// Server response.status values we consider a success
const OK_STATUSES = new Set(['done', 'processing_async', 'success', 'started']);

// 2. STATE ─────────────────────────────────────────────────────
const state = {
    breadcrumb:    [{ name: 'media', path: MEDIA_ROOT }],
    currentBlocks: [], // { name, path, isLeaf, contents, badgeText }
    selectedPaths: new Set(),
    currentOp:     'translate',
};

// 3. DOM shortcuts ─────────────────────────────────────────────
const $  = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

// 4. API client ────────────────────────────────────────────────
const api = {
    async listDir(path) {
        const res = await fetch(`${API_BASE}/list-directory?path=${encodeURIComponent(path)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        return data.items || [];
    },

    async post(endpoint, body) {
        const res = await fetch(`${API_BASE}${endpoint}`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(body),
        });
        // Always try to parse JSON (backend returns JSON even for errors)
        try { return await res.json(); }
        catch { return { status: 'error', message: `HTTP ${res.status}` }; }
    },

    async history() {
        const res = await fetch(`${API_BASE}/history`);
        return (await res.json()).history || [];
    },
};

// 5. FOLDER BROWSER ────────────────────────────────────────────
function computeBadges(items) {
    let videos = 0, subs = 0, audio = 0, folders = 0;
    for (const it of items) {
        if (it.type === 'folder') { folders++; continue; }
        const n = it.name.toLowerCase();
        if      (VIDEO_RE.test(n))   videos++;
        else if (n.endsWith('.srt')) subs++;
        else if (AUDIO_RE.test(n))   audio++;
    }
    const parts = [];
    if (folders) parts.push(`📁 ${folders}`);
    if (videos)  parts.push(`🎬 ${videos}`);
    if (subs)    parts.push(`📝 ${subs}`);
    if (audio)   parts.push(`🔊 ${audio}`);
    return { text: parts.join('  ·  ') || '(empty)', folders };
}

function fileIcon(name) {
    const n = name.toLowerCase();
    if (VIDEO_RE.test(n))       return '🎬';
    if (n.endsWith('.srt'))     return '📝';
    if (AUDIO_RE.test(n))       return '🔊';
    return '📄';
}

async function loadLevel(path) {
    state.selectedPaths.clear();
    updateActionBtn();
    setArea('<div class="blocks-state">Loading…</div>');
    renderBreadcrumb();

    let topItems;
    try {
        topItems = await api.listDir(path);
    } catch (err) {
        setArea(`<div class="blocks-state" style="color:#e74c3c">❌ ${esc(err.message)}</div>`);
        return;
    }

    const folders = topItems
        .filter(i => i.type === 'folder')
        .sort((a, b) => a.name.localeCompare(b.name));

    if (folders.length === 0) {
        setArea('<div class="blocks-state">No subfolders found here</div>');
        return;
    }

    setArea('<div class="blocks-state">Loading folder contents…</div>');

    // Load each subfolder's contents in parallel for badge display
    const settled = await Promise.allSettled(folders.map(f => api.listDir(f.path)));

    state.currentBlocks = folders.map((f, i) => {
        const items    = settled[i].status === 'fulfilled' ? settled[i].value : [];
        const badges   = computeBadges(items);
        const contents = items
            .filter(it => it.type === 'file' && MEDIA_RE.test(it.name))
            .sort((a, b) => a.name.localeCompare(b.name));
        return {
            name:      f.name,
            path:      f.path,
            isLeaf:    badges.folders === 0,
            contents,
            badgeText: badges.text,
        };
    });

    renderBlocks();
}

function renderBreadcrumb() {
    $('breadcrumb').innerHTML = state.breadcrumb.map((b, i) => {
        const sep  = i > 0 ? '<span class="bc-sep"> / </span>' : '';
        const last = i === state.breadcrumb.length - 1;
        if (last) return `${sep}<span class="bc-item current">${esc(b.name)}</span>`;
        return `${sep}<span class="bc-item" data-bc-idx="${i}">${esc(b.name)}</span>`;
    }).join('');
}

function renderBlocks() {
    const html = state.currentBlocks.map((b, i) =>
        b.isLeaf ? leafBlockHTML(b, i) : navBlockHTML(b, i)
    ).join('');
    setArea(`<div class="blocks-grid">${html}</div>`);
}

function navBlockHTML(b, i) {
    return `
        <div class="block-nav" data-nav-idx="${i}" title="${esc(b.path)}">
            <div class="nav-folder-icon">📁</div>
            <div class="nav-info">
                <div class="nav-name">${esc(b.name)}</div>
                <div class="nav-badge">${esc(b.badgeText)}</div>
            </div>
            <div class="nav-arrow">›</div>
        </div>`;
}

function leafBlockHTML(b, i) {
    const sel      = state.selectedPaths.has(b.path);
    const selClass = sel ? ' selected' : '';
    const fileRows = b.contents.map(f =>
        `<div class="file-row">
            <span class="file-icon">${fileIcon(f.name)}</span>
            <span class="file-name">${esc(f.name)}</span>
        </div>`
    ).join('') || '<div style="color:#666;font-size:0.8rem;padding:2px 0">No media files found</div>';

    return `
        <div class="block-leaf${selClass}" data-leaf-idx="${i}" title="${esc(b.path)}">
            <div class="leaf-header">
                <div class="leaf-folder-icon">📁</div>
                <div class="leaf-info">
                    <div class="leaf-name">${esc(b.name)}</div>
                    <div class="leaf-badges">${esc(b.badgeText)}</div>
                </div>
                <div class="leaf-check">✓</div>
            </div>
            <div class="leaf-files">${fileRows}</div>
        </div>`;
}

function setArea(html) { $('blocks-area').innerHTML = html; }

function navigateInto(idx) {
    const b = state.currentBlocks[idx];
    state.breadcrumb.push({ name: b.name, path: b.path });
    loadLevel(b.path);
}

function navToCrumb(idx) {
    state.breadcrumb = state.breadcrumb.slice(0, idx + 1);
    loadLevel(state.breadcrumb[state.breadcrumb.length - 1].path);
}

function toggleSelect(idx) {
    const b = state.currentBlocks[idx];
    if (state.selectedPaths.has(b.path)) state.selectedPaths.delete(b.path);
    else state.selectedPaths.add(b.path);

    // Re-render just this block to avoid a full-grid flicker
    const grid = document.querySelector('.blocks-grid');
    if (grid && grid.children[idx]) {
        grid.children[idx].outerHTML = leafBlockHTML(b, idx);
    }
    updateActionBtn();
}

// 6. OPERATION RUNNER ──────────────────────────────────────────
function selectOp(op) {
    state.currentOp = op;
    $$('.op-card').forEach(c => c.classList.toggle('active', c.dataset.op === op));
    ['translate', 'shift', 'sync'].forEach(id =>
        $(`${id}-settings`).classList.toggle('hidden', id !== op)
    );
    updateActionBtn();
}

function selectPriority(el) {
    $$('.p-option').forEach(o => o.classList.remove('selected'));
    el.classList.add('selected');
    el.querySelector('input[type="radio"]').checked = true;
}

function updateActionBtn() {
    const btn   = $('action-btn');
    const count = state.selectedPaths.size;
    const meta  = OP_META[state.currentOp];
    if (count === 0) {
        btn.disabled    = true;
        btn.textContent = 'Select folders above to continue';
    } else {
        btn.disabled    = false;
        const noun = count === 1 ? 'folder' : 'folders';
        btn.textContent = `${meta.label}: ${count} ${noun} ${meta.emoji}`;
    }
}

function folderNameOf(path) {
    return path.split('/').filter(Boolean).pop() || path;
}

function isOk(resp) {
    return resp && OK_STATUSES.has(resp.status);
}

function respMessage(resp) {
    if (!resp) return 'No response';
    return resp.message || resp.status || 'Unknown';
}

async function runOp() {
    const paths = [...state.selectedPaths];
    if (paths.length === 0) return;

    const btn  = $('action-btn');
    const wrap = $('status-wrap');
    btn.disabled = true;
    wrap.innerHTML = '';

    if (state.currentOp === 'sync') {
        const results = await Promise.all(paths.map(p => runSyncWithProgress(p, wrap)));
        paths.forEach((path, i) => { if (results[i]) state.selectedPaths.delete(path); });
    } else {
        await runTranslateOrShift(paths, wrap);
    }

    renderBlocks();
    updateActionBtn();
    loadHistory();
    btn.disabled = false;
}

function buildExtra() {
    if (state.currentOp === 'shift') {
        return { offset_ms: parseInt($('shift-offset').value, 10) || -500 };
    }
    return {};
}

async function runTranslateOrShift(paths, wrap) {
    const op     = state.currentOp;
    const meta   = OP_META[op];
    const extra  = buildExtra();
    const rows   = paths.map(p => addStatusRow(wrap, folderNameOf(p), 'processing', `${meta.verb}…`));

    const settled = await Promise.allSettled(
        paths.map(path => api.post(meta.endpoint, { path, ...extra }))
    );

    settled.forEach((res, i) => {
        const path = paths[i];
        const name = folderNameOf(path);
        const row  = rows[i];
        if (res.status === 'fulfilled' && isOk(res.value)) {
            state.selectedPaths.delete(path);
            setStatusRow(row, 'success', name, respMessage(res.value));
        } else {
            const msg = res.status === 'fulfilled'
                ? respMessage(res.value)
                : (res.reason?.message || 'Request failed');
            setStatusRow(row, 'error', name, msg, () => retrySingle(path, name, extra, row));
        }
    });
}

async function retrySingle(path, name, extra, row) {
    const meta = OP_META[state.currentOp];
    setStatusRow(row, 'processing', name, `${meta.verb}…`);

    const resp = await api.post(meta.endpoint, { path, ...extra });
    if (isOk(resp)) {
        state.selectedPaths.delete(path);
        setStatusRow(row, 'success', name, respMessage(resp));
    } else {
        setStatusRow(row, 'error', name, respMessage(resp), () => retrySingle(path, name, extra, row));
    }
    renderBlocks();
    updateActionBtn();
    loadHistory();
}

// Status rows are stable DOM nodes — we reuse them across retries
function addStatusRow(wrap, name, type, msg) {
    const row = document.createElement('div');
    wrap.appendChild(row);
    setStatusRow(row, type, name, msg);
    return row;
}

function setStatusRow(row, type, name, msg, onRetry) {
    row.className = `status-item ${type}`;
    const icon = type === 'processing' ? '<div class="spinner"></div>'
               : type === 'success'    ? '✅' : '❌';
    row.innerHTML = `${icon}
        <span class="status-name">${esc(name)}</span>
        <span class="status-msg">${esc(msg)}</span>`;
    if (type === 'error' && onRetry) {
        const btn = document.createElement('button');
        btn.className   = 'retry-btn';
        btn.textContent = '↩ Retry';
        btn.onclick     = onRetry;
        row.appendChild(btn);
    }
}

// 7. SYNC PROGRESS (SSE) ───────────────────────────────────────
// Returns true on job.status === 'done' with zero failures
async function runSyncWithProgress(path, wrap) {
    const priority = document.querySelector('input[name="sync-priority"]:checked')?.value || 'auto';
    const name     = folderNameOf(path);
    const card     = makeSyncCard(wrap, name);

    const start = await api.post('/sync', { path, priority });
    if (!isOk(start) || !start.job_id) {
        finishSyncCard(card, false, start.message || start.status || 'Failed to start');
        addSyncRetryBtn(card, path, wrap);
        return false;
    }

    const finalJob = await streamSyncProgress(start.job_id, card);
    const success  = finalJob?.status === 'done' && !(finalJob?.failed?.length);
    if (!success) addSyncRetryBtn(card, path, wrap);
    return success;
}

function makeSyncCard(wrap, name) {
    const card = document.createElement('div');
    card.className = 'sync-progress-card';
    card.innerHTML = `
        <div class="sync-header">
            <div class="spinner"></div>
            <span class="sync-folder-name">${esc(name)}</span>
            <span class="sync-status-text">Starting…</span>
        </div>
        <div class="sync-overall-bar"><div class="sync-bar-fill" style="width:0%"></div></div>
        <div class="sync-files-list"></div>`;
    wrap.appendChild(card);
    return card;
}

function streamSyncProgress(jobId, card) {
    return new Promise(resolve => {
        const es = new EventSource(`${API_BASE}/sync/progress/${jobId}`);
        let last = null;
        es.onmessage = e => {
            const job = JSON.parse(e.data);
            last = job;
            updateSyncCard(card, job);
            if (job.status === 'done' || job.status === 'error') {
                es.close();
                resolve(job);
            }
        };
        es.onerror = () => { es.close(); resolve(last); };
    });
}

function updateSyncCard(card, job) {
    const statusText = card.querySelector('.sync-status-text');
    const barFill    = card.querySelector('.sync-bar-fill');
    const filesList  = card.querySelector('.sync-files-list');
    const headerIcon = card.querySelector('.sync-header > :first-child');

    const pct = job.total > 0 ? Math.round((job.current / job.total) * 100) : 0;
    barFill.style.width = `${pct}%`;

    if (job.status === 'done') {
        statusText.textContent = job.message || 'Done';
        if (headerIcon.classList.contains('spinner')) headerIcon.outerHTML = '<span>✅</span>';
    } else if (job.status === 'error') {
        statusText.textContent = job.message || 'Error';
        if (headerIcon.classList.contains('spinner')) headerIcon.outerHTML = '<span>❌</span>';
    } else if (job.total > 0) {
        statusText.textContent = `${job.current} / ${job.total} files`;
    }

    if (job.files?.length) {
        filesList.innerHTML = job.files.map(syncFileRowHTML).join('');
    }
}

function syncFileRowHTML(f) {
    const stateClass = f.status === 'done'       ? 'done'
                     : f.status === 'failed'     ? 'failed'
                     : f.status === 'processing' ? 'active' : '';
    const icon = f.status === 'done'       ? '✅'
               : f.status === 'failed'     ? '❌'
               : f.status === 'processing' ? '<span class="spinner-sm"></span>' : '⏳';
    const method = f.method
        ? `<span class="sync-method-badge">${f.method === 'srt' ? '⚡ srt' : '🎤 audio'}</span>`
        : '';
    const err = f.error ? ` — <span class="sync-file-error">${esc(f.error)}</span>` : '';
    return `<div class="sync-file-row ${stateClass}">
        ${icon}
        <span class="sync-file-name">${esc(f.name)}${err}</span>
        ${method}
    </div>`;
}

function finishSyncCard(card, success, msg) {
    card.querySelector('.sync-header > :first-child').outerHTML = success ? '<span>✅</span>' : '<span>❌</span>';
    card.querySelector('.sync-status-text').textContent = msg;
}

function addSyncRetryBtn(card, path, wrap) {
    const btn = document.createElement('button');
    btn.className   = 'retry-btn block';
    btn.textContent = '↩ Retry';
    btn.onclick     = async () => {
        card.remove();
        const ok = await runSyncWithProgress(path, wrap);
        if (ok) state.selectedPaths.delete(path);
        renderBlocks();
        updateActionBtn();
        loadHistory();
    };
    card.appendChild(btn);
}

// 8. HISTORY ───────────────────────────────────────────────────
async function loadHistory() {
    let rows;
    try { rows = await api.history(); }
    catch { return; }

    const card = $('history-card');
    const list = $('history-list');

    if (!rows.length) {
        card.classList.add('hidden');
        return;
    }

    card.classList.remove('hidden');
    const OP_ICON = { translate: '🌐', shift: '⏱️', sync: '🔄' };
    list.innerHTML = rows.map(h => {
        const name = folderNameOf(h.path);
        const date = new Date(h.timestamp * 1000).toLocaleString();
        const icon = OP_ICON[h.operation] || '📋';
        return `<div class="history-row">
            <span>${icon}</span>
            <span class="history-name" title="${esc(h.path)}">${esc(name)}</span>
            <span class="history-op-badge">${esc(h.operation)}</span>
            <span class="history-path" title="${esc(h.path)}">${esc(h.path)}</span>
            <span class="history-time">${esc(date)}</span>
        </div>`;
    }).join('');
}

// 9. UTIL ──────────────────────────────────────────────────────
function esc(str) {
    const d = document.createElement('div');
    d.textContent = String(str ?? '');
    return d.innerHTML;
}

// 10. INIT ─────────────────────────────────────────────────────
function bindEvents() {
    // Op-card selector (event delegation)
    $('op-selector').addEventListener('click', e => {
        const card = e.target.closest('.op-card');
        if (card) selectOp(card.dataset.op);
    });

    // Priority selector
    $('priority-list').addEventListener('click', e => {
        const opt = e.target.closest('.p-option');
        if (opt) selectPriority(opt);
    });

    // Action button
    $('action-btn').addEventListener('click', runOp);

    // Breadcrumb + block grid (event delegation, since they re-render)
    $('breadcrumb').addEventListener('click', e => {
        const item = e.target.closest('.bc-item[data-bc-idx]');
        if (item) navToCrumb(parseInt(item.dataset.bcIdx, 10));
    });

    $('blocks-area').addEventListener('click', e => {
        const nav  = e.target.closest('.block-nav[data-nav-idx]');
        const leaf = e.target.closest('.block-leaf[data-leaf-idx]');
        if (nav)  return navigateInto(parseInt(nav.dataset.navIdx, 10));
        if (leaf) return toggleSelect(parseInt(leaf.dataset.leafIdx, 10));
    });
}

function init() {
    bindEvents();
    selectOp('translate');
    loadLevel(MEDIA_ROOT);
    loadHistory();
}

init();
