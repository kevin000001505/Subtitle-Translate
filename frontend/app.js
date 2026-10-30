/* ============================================================
   Subtitle Library — frontend
   Sections:
     1. CONFIG & STATE
     2. API client
     3. Preferences (localStorage)
     4. Pickers (server, language, model, operation)
     5. Library grid
     6. Selection
     7. Show dialog
     8. Running jobs
     9. Jobs panel
    10. Utilities
    11. Init
   ============================================================ */

// 1. CONFIG & STATE ────────────────────────────────────────────
const API_BASE = '/api';

const OPS = {
    translate: { label: 'Translate', verb: 'translate' },
    shift:     { label: 'Shift',     verb: 'shift' },
    sync:      { label: 'Sync',      verb: 'sync' },
};

const STATUS_LABEL = {
    done: 'Done', partial: 'Partial', missing: 'Missing',
    running: 'Running', unavailable: 'No file',
};

const state = {
    config: null,
    server: null,
    lang: 'zh-TW',
    op: 'translate',
    kind: 'all',
    query: '',
    titles: [],
    selection: new Map(),   // key -> { kind, id, show_id?, label }
    show: null,             // the show open in the dialog
    activeJobs: 0,
};

const $ = id => document.getElementById(id);

// 2. API client ────────────────────────────────────────────────
async function api(path, options = {}) {
    const res = await fetch(`${API_BASE}${path}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
        const detail = Array.isArray(body.detail) ? body.detail.map(d => d.msg).join('; ') : body.detail;
        throw new Error(detail || `HTTP ${res.status}`);
    }
    return body;
}

// 3. Preferences ───────────────────────────────────────────────
const prefs = {
    get(key, fallback) {
        try { return localStorage.getItem(`subtitle-library.${key}`) ?? fallback; } catch { return fallback; }
    },
    set(key, value) {
        try { localStorage.setItem(`subtitle-library.${key}`, value); } catch { /* private mode */ }
    },
};

// 4. Pickers ───────────────────────────────────────────────────
function renderServerTabs() {
    const tabs = $('server-tabs');
    const servers = state.config.servers;
    tabs.classList.toggle('hidden', servers.length < 2);
    tabs.innerHTML = '';
    for (const s of servers) {
        const b = el('button', { textContent: s.label, className: s.name === state.server ? 'active' : '' });
        b.onclick = () => {
            if (s.name === state.server) return;
            state.server = s.name;
            prefs.set('server', s.name);
            state.selection.clear();
            renderServerTabs();
            loadLibrary();
        };
        tabs.append(b);
    }
}

function renderLanguages() {
    const select = $('lang');
    select.innerHTML = '';
    for (const l of state.config.languages) {
        select.append(el('option', { value: l.code, textContent: `${l.name} (${l.code})` }));
    }
    select.value = state.lang;
    select.onchange = () => {
        state.lang = select.value;
        prefs.set('lang', state.lang);
        loadLibrary();
    };
}

async function loadModels() {
    const select = $('model');
    const note = $('model-note');
    select.innerHTML = '<option>Loading models…</option>';
    let body;
    try {
        body = await api('/models');
    } catch (e) {
        body = { local: [], local_error: e.message, gemini: null };
    }
    select.innerHTML = '';
    if (body.local.length) {
        const group = el('optgroup', { label: 'Local (llama-swap)' });
        for (const m of body.local) {
            group.append(el('option', {
                value: `llama-swap|${m.id}`,
                textContent: m.loaded ? `${m.name} ● loaded` : m.name,
            }));
        }
        select.append(group);
    }
    if (body.gemini) {
        const group = el('optgroup', { label: 'Cloud' });
        group.append(el('option', { value: `gemini|${body.gemini.id}`, textContent: body.gemini.name }));
        select.append(group);
    }
    if (!select.options.length) {
        select.append(el('option', { value: '', textContent: 'No model available' }));
    }
    note.classList.toggle('hidden', !body.local_error);
    note.textContent = body.local_error ? `Local models unavailable: ${body.local_error}` : '';

    // Last pick, else the configured default, else any TranslateGemma model, else the first.
    const values = [...select.options].map(o => o.value);
    const preferred = [prefs.get('model', ''), `llama-swap|${state.config.default_model}`,
                       values.find(v => /translategemma/i.test(v))];
    select.value = preferred.find(v => v && values.includes(v)) || values[0];
    select.onchange = () => prefs.set('model', select.value);
    updateRunButton();
}

function setOp(op) {
    state.op = op;
    for (const b of document.querySelectorAll('#op-tabs .op')) b.classList.toggle('active', b.dataset.op === op);
    for (const name of Object.keys(OPS)) {
        for (const node of document.querySelectorAll(`.op-${name}`)) node.classList.toggle('hidden', name !== op);
    }
    updateRunButton();
}

// 5. Library grid ──────────────────────────────────────────────
async function loadLibrary() {
    const status = $('library-state');
    if (!state.server) {
        status.textContent = 'No media server configured. Set JELLYFIN_URL or PLEX_URL in .env and restart the backend.';
        status.classList.remove('hidden');
        $('grid').innerHTML = '';
        return;
    }
    status.textContent = 'Loading library…';
    status.classList.remove('hidden');
    try {
        const body = await api(`/library/${state.server}?lang=${encodeURIComponent(state.lang)}`);
        state.titles = body.titles;
        renderGrid();
    } catch (e) {
        state.titles = [];
        $('grid').innerHTML = '';
        status.textContent = `Could not load the library: ${e.message}`;
    }
}

function visibleTitles() {
    const q = state.query.trim().toLowerCase();
    return state.titles.filter(t =>
        (state.kind === 'all' || t.kind === state.kind) && (!q || t.name.toLowerCase().includes(q)));
}

function badgeText(t) {
    if (t.kind === 'show' && (t.status === 'partial' || t.status === 'done')) return `${t.done}/${t.total}`;
    return STATUS_LABEL[t.status] || t.status;
}

function renderGrid() {
    const grid = $('grid');
    const titles = visibleTitles();
    const status = $('library-state');
    status.classList.toggle('hidden', titles.length > 0);
    status.textContent = state.titles.length ? 'No titles match.' : 'This library is empty.';
    grid.innerHTML = '';
    for (const t of titles) grid.append(card(t));
}

function card(t) {
    const key = selectionKey(t.kind, t.id);
    const node = el('article', {
        className: `card ${t.status === 'unavailable' ? 'unavailable' : ''} ${state.selection.has(key) ? 'selected' : ''}`,
    });
    node.dataset.key = key;

    const wrap = el('div', { className: 'poster-wrap' });
    const img = el('img', { className: 'poster', loading: 'lazy', alt: '', src: `${API_BASE}${t.poster}` });
    img.onerror = () => { img.remove(); wrap.prepend(el('div', { className: 'poster-fallback', textContent: t.name })); };
    wrap.append(
        img,
        el('span', { className: `badge ${t.status}`, textContent: badgeText(t) }),
        el('span', { className: 'select-box', title: 'Select' }),
        el('span', { className: 'kind-tag', textContent: t.kind === 'show' ? 'Series' : 'Movie' }),
    );
    if (t.kind === 'show') {
        const open = el('button', { className: 'open-show', textContent: 'Episodes' });
        open.onclick = e => { e.stopPropagation(); openShow(t); };
        wrap.append(open);
    }
    node.append(wrap, el('div', { className: 'card-body' },
        el('div', { className: 'card-title', textContent: t.name, title: t.name }),
        el('div', { className: 'card-meta', textContent: [t.year, t.kind === 'show' && t.total ? `${t.total} episode${t.total === 1 ? '' : 's'}` : ''].filter(Boolean).join(' · ') }),
    ));

    node.onclick = () => {
        if (t.status === 'unavailable') return toast(`${t.name} has no file this app can reach`);
        toggleSelection({ kind: t.kind, id: t.id, label: t.name });
        node.classList.toggle('selected', state.selection.has(key));
    };
    return node;
}

// 6. Selection ─────────────────────────────────────────────────
function selectionKey(kind, id) { return `${kind}:${id}`; }

function toggleSelection(item, on) {
    const key = selectionKey(item.kind, item.id);
    const select = on ?? !state.selection.has(key);
    if (select) state.selection.set(key, item); else state.selection.delete(key);
    updateRunButton();
}

function updateRunButton() {
    const n = state.selection.size;
    const items = [...state.selection.values()];
    const shows = items.filter(i => i.kind === 'show').length;
    const episodes = items.filter(i => i.kind === 'episode').length;
    const movies = n - shows - episodes;
    const parts = [movies && `${movies} movie${movies > 1 ? 's' : ''}`,
                   shows && `${shows} show${shows > 1 ? 's' : ''}`,
                   episodes && `${episodes} episode${episodes > 1 ? 's' : ''}`].filter(Boolean);
    $('selection-count').textContent = n ? `Selected: ${parts.join(', ')}` : 'Nothing selected';
    $('clear-selection').classList.toggle('hidden', !n);
    const run = $('run');
    run.textContent = n ? `${OPS[state.op].label} ${n} selected` : OPS[state.op].label;
    run.disabled = !n || (state.op === 'translate' && !$('model').value);
}

function clearSelection() {
    state.selection.clear();
    for (const c of document.querySelectorAll('.card.selected')) c.classList.remove('selected');
    updateRunButton();
}

// 7. Show dialog ───────────────────────────────────────────────
async function openShow(t) {
    state.show = t;
    $('show-title').textContent = t.year ? `${t.name} (${t.year})` : t.name;
    $('show-poster').src = `${API_BASE}${t.poster}`;
    $('show-summary').textContent = `${state.lang} subtitles: ${t.done ?? 0} of ${t.total ?? 0} episodes`;
    $('show-seasons').innerHTML = '<p class="muted">Loading episodes…</p>';
    const whole = $('show-all');
    whole.checked = state.selection.has(selectionKey('show', t.id));
    whole.onchange = () => {
        toggleSelection({ kind: 'show', id: t.id, label: t.name }, whole.checked);
        if (whole.checked) clearEpisodesOf(t.id);
        renderSeasons(state.show.seasons || []);
        syncCard(t);
    };
    $('show-dialog').showModal();
    try {
        const body = await api(`/library/${state.server}/shows/${encodeURIComponent(t.id)}?lang=${encodeURIComponent(state.lang)}`);
        state.show.seasons = body.seasons;
        renderSeasons(body.seasons);
    } catch (e) {
        $('show-seasons').innerHTML = '';
        $('show-seasons').append(el('p', { className: 'muted', textContent: `Could not load episodes: ${e.message}` }));
    }
}

function clearEpisodesOf(showId) {
    for (const [key, item] of state.selection) if (item.kind === 'episode' && item.show_id === showId) state.selection.delete(key);
}

function renderSeasons(seasons) {
    const box = $('show-seasons');
    const show = state.show;
    const wholeShow = state.selection.has(selectionKey('show', show.id));
    box.innerHTML = '';
    for (const s of seasons) {
        const available = s.episodes.filter(e => e.status !== 'unavailable');
        const picked = available.filter(e => wholeShow || state.selection.has(selectionKey('episode', e.id)));
        const seasonBox = el('input', { type: 'checkbox', disabled: wholeShow || !available.length });
        seasonBox.checked = available.length > 0 && picked.length === available.length;
        seasonBox.indeterminate = picked.length > 0 && picked.length < available.length;
        seasonBox.onchange = () => {
            for (const e of available) toggleSelection(episodeItem(show, s, e), seasonBox.checked);
            renderSeasons(seasons);
        };
        const section = el('section', { className: 'season' },
            el('label', { className: 'season-head' }, seasonBox,
                el('span', { textContent: s.season === 0 ? 'Specials' : `Season ${s.season}` }),
                el('span', { className: 'muted', textContent: `${s.episodes.filter(e => e.status === 'done').length}/${available.length} done` })));
        for (const e of s.episodes) {
            const unavailable = e.status === 'unavailable';
            const box = el('input', { type: 'checkbox', disabled: wholeShow || unavailable });
            box.checked = !unavailable && (wholeShow || state.selection.has(selectionKey('episode', e.id)));
            box.onchange = () => { toggleSelection(episodeItem(show, s, e), box.checked); renderSeasons(seasons); };
            section.append(el('label', { className: `episode ${unavailable ? 'unavailable' : ''}` },
                box,
                el('span', { textContent: `E${String(e.number).padStart(2, '0')}` }),
                el('span', { className: 'ep-name', textContent: e.name }),
                el('span', { className: `badge pill ${e.status}`, textContent: STATUS_LABEL[e.status] || e.status })));
        }
        box.append(section);
    }
    syncCard(show);
}

function episodeItem(show, season, e) {
    return { kind: 'episode', id: e.id, show_id: show.id,
             label: `${show.name} S${String(season.season).padStart(2, '0')}E${String(e.number).padStart(2, '0')}` };
}

function syncCard(t) {
    const node = document.querySelector(`.card[data-key="${CSS.escape(selectionKey('show', t.id))}"]`);
    const anyEpisode = [...state.selection.values()].some(i => i.kind === 'episode' && i.show_id === t.id);
    if (node) node.classList.toggle('selected', state.selection.has(selectionKey('show', t.id)) || anyEpisode);
}

// 8. Running jobs ──────────────────────────────────────────────
async function runSelected() {
    const items = [...state.selection.values()].map(({ kind, id, show_id }) => ({ kind, id, show_id }));
    const body = { server: state.server, op: state.op, lang: state.lang, items };
    if (state.op === 'translate') {
        const [backend, model] = $('model').value.split(/\|(.*)/s);
        Object.assign(body, { backend, model, overwrite: $('overwrite').checked });
    } else if (state.op === 'shift') {
        body.offset_ms = parseInt($('offset').value, 10) || 0;
    } else {
        body.priority = $('priority').value;
    }
    $('run').disabled = true;
    try {
        const result = await api('/jobs', { method: 'POST', body: JSON.stringify(body) });
        const parts = [`Queued ${result.created} job${result.created === 1 ? '' : 's'}`];
        if (result.already_queued) parts.push(`${result.already_queued} already queued`);
        if (result.unavailable.length) parts.push(`${result.unavailable.length} without a file`);
        toast(parts.join(' · '));
        clearSelection();
        refreshJobs();
        loadLibrary();
    } catch (e) {
        toast(`Could not queue: ${e.message}`);
        updateRunButton();
    }
}

// 9. Jobs panel ────────────────────────────────────────────────
let jobsTimer = null;

async function refreshJobs() {
    clearTimeout(jobsTimer);
    let list = [];
    try {
        list = (await api('/jobs')).jobs;
    } catch { /* keep polling */ }
    const active = list.filter(j => j.status === 'queued' || j.status === 'running').length;
    if (state.activeJobs && active < state.activeJobs) loadLibrary();   // a job finished: badges changed
    state.activeJobs = active;
    renderJobs(list, active);
    jobsTimer = setTimeout(refreshJobs, active ? 2000 : 10000);
}

function renderJobs(list, active) {
    const badge = $('jobs-badge');
    badge.textContent = active;
    badge.classList.toggle('hidden', !active);
    const ol = $('jobs-list');
    ol.innerHTML = '';
    if (!list.length) {
        ol.append(el('li', { className: 'muted job', textContent: 'No jobs yet. Select posters and press the button above.' }));
        return;
    }
    for (const j of list) {
        const what = j.op === 'translate' ? `→ ${j.target_lang} with ${j.model}`
                   : j.op === 'shift' ? `${j.target_lang} shift ${j.params.offset_ms > 0 ? '+' : ''}${j.params.offset_ms} ms`
                   : `${j.target_lang} sync (${j.params.priority})`;
        const li = el('li', { className: 'job' },
            el('div', {}, el('span', { className: 'job-title', textContent: j.title || j.video_path }),
                el('span', { className: 'job-meta', textContent: `  ${OPS[j.op]?.label || j.op} ${what}` })),
            el('div', { className: `job-status ${j.status}`, textContent: j.status === 'running' ? `${Math.round(j.progress)}%` : j.status }));
        if (j.status === 'running') {
            const bar = el('div', { className: 'bar' }, el('div'));
            bar.firstChild.style.width = `${j.progress}%`;
            li.append(bar);
        }
        if (j.message) li.append(el('div', { className: 'job-msg', textContent: j.message }));
        ol.append(li);
    }
}

// 10. Utilities ────────────────────────────────────────────────
function el(tag, props = {}, ...children) {
    const node = Object.assign(document.createElement(tag), props);
    node.append(...children);
    return node;
}

let toastTimer = null;
function toast(text) {
    const t = $('toast');
    t.textContent = text;
    t.classList.remove('hidden');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.add('hidden'), 4000);
}

// 11. Init ─────────────────────────────────────────────────────
async function init() {
    try {
        state.config = await api('/config');
    } catch (e) {
        $('library-state').textContent = `Backend unreachable: ${e.message}`;
        return;
    }
    const names = state.config.servers.map(s => s.name);
    const savedServer = prefs.get('server', '');
    state.server = names.includes(savedServer) ? savedServer : names[0] || null;
    const codes = state.config.languages.map(l => l.code);
    const savedLang = prefs.get('lang', state.config.default_lang);
    state.lang = codes.includes(savedLang) ? savedLang : state.config.default_lang;

    renderServerTabs();
    renderLanguages();
    for (const b of document.querySelectorAll('#op-tabs .op')) b.onclick = () => setOp(b.dataset.op);
    for (const b of document.querySelectorAll('#kind-filter .chip')) {
        b.onclick = () => {
            state.kind = b.dataset.kind;
            for (const c of document.querySelectorAll('#kind-filter .chip')) c.classList.toggle('active', c === b);
            renderGrid();
        };
    }
    $('search').oninput = e => { state.query = e.target.value; renderGrid(); };
    $('run').onclick = runSelected;
    $('clear-selection').onclick = clearSelection;
    $('show-apply').onclick = () => $('show-dialog').close();
    $('jobs-toggle').onclick = () => $('jobs-panel').classList.toggle('collapsed');
    $('jobs-clear').onclick = async () => { await api('/jobs/finished', { method: 'DELETE' }); refreshJobs(); };
    setOp('translate');

    loadModels();
    loadLibrary();
    refreshJobs();
}

init();
