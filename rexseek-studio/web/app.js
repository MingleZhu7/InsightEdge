/* ============================================================
   RexSeek Studio — 前端交互
   ============================================================ */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);

  const AGENT_SVG = `<svg viewBox="0 0 32 32" fill="none"><circle cx="16" cy="16" r="8" stroke="#5eead4" stroke-width="1.4" opacity=".9"/><circle cx="16" cy="16" r="2.6" fill="#5eead4"/><path d="M16 4.5v3M16 24.5v3M4.5 16h3M24.5 16h3" stroke="#7cc4ff" stroke-width="1.4" stroke-linecap="round"/></svg>`;
  const ARROW_SVG = `<svg class="arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h13"/><path d="m13 6 6 6-6 6"/></svg>`;
  const COPY_SVG = `<svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/></svg>`;

  const els = {
    stage: $("#stage"),
    mediaBox: $("#mediaBox"),
    stageImage: $("#stageImage"),
    stageVideo: $("#stageVideo"),
    overlay: $("#overlay"),
    stageEmpty: $("#stageEmpty"),
    mediaType: $("#mediaType"),
    mediaDim: $("#mediaDim"),
    targetCount: $("#targetCount"),
    indexChip: $("#indexChip"),
    sessionChip: $("#sessionChip"),
    videoBar: $("#videoBar"),
    playBtn: $("#playBtn"),
    vProgress: $("#vProgress"),
    vFill: $("#vFill"),
    vThumb: $("#vThumb"),
    vTime: $("#vTime"),
    sampleStrip: $("#sampleStrip"),
    stripHint: $("#stripHint"),
    segStrip: $("#segStrip"),
    segMeta: $("#segMeta"),
    segItems: $("#segItems"),
    sysChip: $("#sysChip"),
    engineBadge: $("#engineBadge"),
    engineText: $("#engineText"),
    modelChip: $("#modelChip"),
    agentSub: $("#agentSub"),
    chatScroll: $("#chatScroll"),
    messages: $("#messages"),
    welcome: $("#welcome"),
    welcomeDesc: $("#welcomeDesc"),
    welcomeScene: $("#welcomeScene"),
    welcomeSuggestions: $("#welcomeSuggestions"),
    quickRow: $("#quickRow"),
    input: $("#input"),
    sendBtn: $("#sendBtn"),
    uploadBtn: $("#uploadBtn"),
    fileInput: $("#fileInput"),
    toggleBoxesBtn: $("#toggleBoxesBtn"),
    restartBtn: $("#restartBtn"),
    clearBtn: $("#clearBtn"),
    toast: $("#toast"),
  };

  const state = {
    engine: null,
    samples: [],
    sampleMap: new Map(),
    media: null,
    detections: [],
    detEls: new Map(),
    history: [],
    busy: false,
    showBoxes: true,
    objectUrl: null,
    imageDataCache: new Map(),
    raf: 0,
    seeking: false,
    sessionId: "",
    indexAbort: null,
    uploading: false,
    eventRef: null,
    segment: null,
  };

  /* ------------------------------------------------------------- utils */
  const escapeHtml = (str) =>
    String(str ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const escapeAttr = (str) => escapeHtml(str).replace(/`/g, "&#96;");

  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  function inline(text) {
    return escapeHtml(text)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  }

  function formatText(text) {
    const blocks = String(text ?? "").trim().split(/\n{2,}/);
    return blocks
      .map((block) => {
        const lines = block.split("\n");
        if (lines.length && lines.every((line) => /^\s*[-•]\s+/.test(line))) {
          return `<ul>${lines
            .map((line) => `<li>${inline(line.replace(/^\s*[-•]\s+/, ""))}</li>`)
            .join("")}</ul>`;
        }
        return `<p>${lines.map(inline).join("<br>")}</p>`;
      })
      .join("");
  }

  let toastTimer = 0;
  function toast(message) {
    clearTimeout(toastTimer);
    els.toast.textContent = message;
    els.toast.classList.add("show");
    toastTimer = setTimeout(() => els.toast.classList.remove("show"), 2400);
  }

  function newSessionId() {
    if (crypto?.randomUUID) return "s_" + crypto.randomUUID().replace(/-/g, "").slice(0, 16);
    return "s_" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
  }

  function setSessionChip(text, id = "") {
    if (!els.sessionChip) return;
    els.sessionChip.textContent = text;
    els.sessionChip.title = id ? `session: ${id}` : "";
  }

  function persistSessionLocal() {
    try { localStorage.setItem("rexseek_session", state.sessionId); } catch { /* ignore */ }
  }

  function resetSession() {
    state.sessionId = newSessionId();
    persistSessionLocal();
    setSessionChip("新会话", state.sessionId);
    if (state.engine && state.engine.demo === false) {
      fetch("/api/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId }),
      }).catch(() => {});
    }
  }

  async function ensureSession() {
    let saved = "";
    try { saved = localStorage.getItem("rexseek_session") || ""; } catch { /* ignore */ }
    if (!state.engine || state.engine.demo) {
      state.sessionId = saved || newSessionId();
      persistSessionLocal();
      setSessionChip("本地会话", state.sessionId);
      return;
    }
    if (saved) {
      try {
        const res = await fetch(`/api/session/${encodeURIComponent(saved)}`);
        if (res.ok) {
          const d = await res.json();
          if (d.exists) {
            state.sessionId = d.session_id;
            persistSessionLocal();
            setSessionChip(d.turns ? `已恢复 ${d.turns} 轮` : "会话已保存", d.session_id);
            return;
          }
        }
      } catch { /* fall through */ }
    }
    try {
      const res = await fetch("/api/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: saved || undefined }),
      });
      const d = await res.json();
      if (d.session_id) {
        state.sessionId = d.session_id;
        persistSessionLocal();
        setSessionChip(d.resumed && d.turns ? `已恢复 ${d.turns} 轮` : "新会话", d.session_id);
        return;
      }
    } catch { /* ignore */ }
    state.sessionId = saved || newSessionId();
    persistSessionLocal();
    setSessionChip("会话已保存", state.sessionId);
  }

  function scrollToBottom(smooth = true) {
    els.chatScroll.scrollTo({ top: els.chatScroll.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  }

  function hideWelcome() {
    els.welcome.hidden = true;
    renderQuick();
  }

  function autosize() {
    const el = els.input;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 132)}px`;
  }

  function updateComposer() {
    els.sendBtn.disabled = state.busy || !els.input.value.trim();
    els.input.disabled = state.busy;
    els.sendBtn.setAttribute("aria-busy", String(state.busy));
  }

  async function loadStatus() {
    if (!els.sysChip) return;
    try {
      const res = await fetch("/api/status", { cache: "no-store" });
      const d = await res.json();
      const bad = Object.entries(d.services || {}).filter(([, v]) => !v.ok).map(([k]) => k);
      els.sysChip.textContent = d.ok ? "系统在线 · 全离线" : `服务异常：${bad.join(" / ")}`;
      els.sysChip.classList.toggle("bad", !d.ok);
      els.sysChip.classList.toggle("g", Boolean(d.ok));
      els.sysChip.title = Object.entries(d.services || {})
        .map(([k, v]) => `${k}: ${v.ok ? `${v.ms || 0}ms` : (v.error || "down")}`).join("\n");
    } catch {
      els.sysChip.textContent = "网关异常";
      els.sysChip.classList.add("bad");
    }
  }

  /* ------------------------------------------------------------- boot */
  async function init() {
    bindEvents();
    startVideoLoop();
    loadStatus();
    setInterval(loadStatus, 15000);

    try {
      const [health, catalog] = await Promise.all([
        fetch("/api/health", { cache: "no-store" }).then((r) => {
          if (!r.ok) throw new Error(`health ${r.status}`);
          return r.json();
        }),
        fetch("/api/samples", { cache: "no-store" }).then((r) => r.json()),
      ]);

      state.engine = health;
      state.samples = catalog.samples || [];
      state.samples.forEach((s) => state.sampleMap.set(s.id, s));
      applyEngine(health);
      await ensureSession();
      renderStrip();
      if (els.stripHint) els.stripHint.textContent = `${state.samples.length} 组预置场景`;

      if (state.samples.length) loadSample(state.samples[0].id, { silent: true });
      renderWelcome();
      renderQuick();

      // 深链接：?event=<id>（值守台台账跳转）/ ?sample=... / ?img=...&ask=...
      const params = new URLSearchParams(location.search);
      const sampleId = params.get("sample");
      if (sampleId && state.sampleMap.has(sampleId) && sampleId !== state.media?.sampleId) {
        loadSample(sampleId, { silent: true });
      }
      const evId = params.get("event");
      if (evId) {
        await loadEventDeepLink(evId);
      } else {
        const imgParam = params.get("img");
        if (imgParam) await loadExternalImage(imgParam, "事件证据");
      }
      const ask = params.get("ask");
      if (ask) {
        if (evId) whenStageReady().then(() => send(ask));
        else setTimeout(() => send(ask), 350);
      }
    } catch (err) {
      console.error(err);
      applyEngine({ name: "offline", model: "RexSeek-3B", demo: true });
      addError(
        "无法连接本地服务。请在项目目录运行 <code>python3 server.py</code>，再打开 http://127.0.0.1:8848。"
      );
      els.input.disabled = true;
    }
  }

  function applyEngine(engine) {
    const demo = engine.demo !== false;
    els.engineBadge.classList.toggle("live", !demo);
    els.engineText.textContent = demo ? "演示模式 · 内置样例" : `在线 · ${engine.model || "RexSeek-3B"}`;
    els.engineBadge.title = engine.note || (demo ? "内置样例 + 规则路由，无需 GPU" : `已连接 ${engine.endpoint || "推理服务"}`);
    const modelName = String(engine.model || "RexSeek-3B").replace(/\s*·\s*demo$/i, "");
    els.modelChip.textContent = modelName;
    els.agentSub.textContent = demo ? "视觉指代定位 · 演示模式" : `视觉指代定位 · ${modelName}`;
  }

  /* ------------------------------------------------------------- media */
  function renderStrip() {
    els.sampleStrip.innerHTML = "";
    state.samples.forEach((s) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "thumb";
      btn.dataset.id = s.id;
      btn.title = `${s.title} · ${s.subtitle}`;
      const thumbSrc = s.kind === "video" ? s.poster || s.file : s.file;
      btn.innerHTML = `
        <img src="${thumbSrc}" alt="${escapeAttr(s.title)}" loading="lazy" />
        <span class="thumb-label">${escapeHtml(s.title)}</span>
        ${s.kind === "video" ? '<span class="thumb-kind">VIDEO</span>' : ""}
      `;
      btn.addEventListener("click", () => selectSample(s.id));
      els.sampleStrip.append(btn);
    });
  }

  function renderStripActive() {
    const id = state.media?.sampleId;
    [...els.sampleStrip.children].forEach((btn) => btn.classList.toggle("is-active", btn.dataset.id === id));
  }

  function selectSample(id) {
    if (state.busy) return toast("等当前回答结束再切换");
    if (state.media?.sampleId === id) return;
    loadSample(id);
  }

  function loadSample(id, options = {}) {
    const sample = state.sampleMap.get(id);
    if (!sample) return;
    state.eventRef = null;

    if (state.objectUrl) {
      URL.revokeObjectURL(state.objectUrl);
      state.objectUrl = null;
    }

    state.media = {
      kind: sample.kind,
      sampleId: sample.id,
      url: sample.file,
      width: sample.width,
      height: sample.height,
      duration: sample.duration || 0,
      title: sample.title,
      name: sample.title,
      poster: sample.poster || null,
      uploaded: false,
      dataUrl: null,
      startAt: Number.isFinite(options.startAt) ? options.startAt : null,
      segStart: Number.isFinite(options.startAt) ? options.startAt : null,
      segEnd: Number.isFinite(options.endAt) ? options.endAt : null,
    };
    state.detections = [];
    clearOverlay();
    renderStage();
    updateHud();
    renderStripActive();
    renderWelcome();
    renderQuick();
    if (!options.silent) resetSession();
    indexMedia();

    if (!options.silent && state.history.length) {
      addDivider(`切换到「${sample.title}」`);
    }
  }

  /* ------------------------------------------------ 事件跳转（值守台 → 这里） */
  async function loadExternalImage(url, title) {
    if (!url) return;
    const dims = await imageDims(url);
    if (state.objectUrl) {
      URL.revokeObjectURL(state.objectUrl);
      state.objectUrl = null;
    }
    state.media = {
      kind: "image",
      sampleId: null,
      url,
      width: dims.width,
      height: dims.height,
      duration: 0,
      title: title || "事件证据",
      name: title || "事件证据",
      poster: null,
      uploaded: false,
      dataUrl: null,
      startAt: null,
    };
    state.detections = [];
    clearOverlay();
    renderStage();
    updateHud();
    renderStripActive();
    renderWelcome();
    renderQuick();
    indexMedia();
  }

  async function loadEventDeepLink(rawId) {
    let ev = null;
    let seg = null;
    try {
      const [evRes, segRes] = await Promise.all([
        fetch(`/api/guard/event/${encodeURIComponent(rawId)}`, { cache: "no-store" }),
        fetch(`/api/guard/segment/${encodeURIComponent(rawId)}`, { cache: "no-store" }),
      ]);
      if (evRes.ok) ev = (await evRes.json()).event;
      if (segRes.ok) seg = await segRes.json();
    } catch (err) {
      console.warn("event deep link failed", err);
    }
    if (!ev) {
      addError(`没有找到事件 #${escapeHtml(String(rawId))}，台账可能已被重置。`);
      return null;
    }
    state.segment = seg;
    renderSegment(seg);
    state.eventRef = ev;

    const v0 = Number(ev.video_enter);
    const v1 = Number(ev.video_exit);
    const sample = ev.sample ? state.sampleMap.get(ev.sample) : null;
    if (sample && Number.isFinite(v0)) {
      // 同一段视频 + 只回放"进入→离开"这一段
      loadSample(sample.id, { silent: true, startAt: Math.max(0, v0), endAt: Number.isFinite(v1) ? v1 : undefined });
      state.eventRef = ev;
      const v = els.stageVideo;
      const playSeg = () => {
        try {
          v.currentTime = Math.max(0, v0);
          v.play().catch(() => {});
        } catch { /* ignore */ }
      };
      v.addEventListener("loadeddata", playSeg, { once: true });
      setTimeout(playSeg, 500);
    } else if (seg && seg.frames && seg.frames.length) {
      // 历史事件没有源视频：用抽帧回放
      await loadExternalImage(seg.frames[0].url, `事件 #${ev.id} · ${seg.frames[0].hms}`);
      state.eventRef = ev;
      markSegItem(0);
    } else if (ev.evidence) {
      await loadExternalImage(ev.evidence, `事件 #${ev.id} · 进入取证`);
      state.eventRef = ev;
    }
    addEventCard(ev, seg);
    return ev;
  }

  /* ---------------------------------------------- 事件片段（抽帧 / 回放） */
  function renderSegment(seg) {
    if (!els.segStrip) return;
    if (!seg || !seg.frames || !seg.frames.length) {
      els.segStrip.hidden = true;
      return;
    }
    els.segStrip.hidden = false;
    const sm = seg.summary || {};
    const people = sm.persons_total != null ? sm.persons_total : "-";
    const helmet = sm.helmet_ok != null ? `${sm.helmet_ok}/${people}` : "-";
    els.segMeta.textContent = `${seg.frames.length} 帧 · 期间 ${people} 人（去重） · 安全帽 ${helmet}`;
    els.segItems.innerHTML = seg.frames.map((f, i) =>
      `<div class="seg-item" data-i="${i}" title="${escapeAttr(f.hms || "")} · 禁区内 ${f.inside ?? 0} 人">
         <img src="${escapeAttr(f.url)}" loading="lazy" alt="帧 ${i + 1}" />
         <span class="t">${escapeHtml(f.hms || "")}</span>
       </div>`
    ).join("");
    els.segItems.querySelectorAll(".seg-item").forEach((el) => {
      el.addEventListener("click", () => onSegmentFrame(seg, Number(el.dataset.i)));
    });
  }

  function markSegItem(idx) {
    if (!els.segItems) return;
    els.segItems.querySelectorAll(".seg-item").forEach((el, j) => el.classList.toggle("on", j === idx));
  }

  async function onSegmentFrame(seg, idx) {
    const f = seg?.frames?.[idx];
    if (!f) return;
    markSegItem(idx);
    if (state.media?.kind === "video" && Number.isFinite(Number(f.video_t))) {
      try {
        els.stageVideo.pause();
        els.stageVideo.currentTime = Number(f.video_t);
      } catch { /* ignore */ }
      updateVideoUI();
      return;
    }
    const keepEvent = state.eventRef;
    await loadExternalImage(f.url, `事件 #${seg.event?.id || ""} · ${f.hms}`);
    state.eventRef = keepEvent;
  }

  function addEventCard(ev, seg) {
    hideWelcome();
    const row = document.createElement("div");
    row.className = "msg agent event-card";
    const span = ev.exit_hms
      ? `进入 <b>${escapeHtml(ev.enter_hms || "-")}</b> · 离开 <b>${escapeHtml(ev.exit_hms)}</b> · 持续 <b>${escapeHtml(ev.duration_text || "-")}</b>`
      : `进入 <b>${escapeHtml(ev.enter_hms || "-")}</b> · 值守中（已持续 ${escapeHtml(ev.duration_text || "-")}）`;
    const sm = seg?.summary || {};
    const total = sm.persons_total ?? ev.persons_total;
    const duty = (ev.kind === "duty") || ev.mission === "duty_zone";
    const stat = [
      duty ? "值守区无人告警" : `最多 <b>${escapeHtml(String(ev.persons_max ?? 1))}</b> 人`,
      !duty && total != null ? `期间共 <b>${escapeHtml(String(total))}</b> 人（去重）` : "",
      seg?.frames?.length ? `抽帧 <b>${seg.frames.length}</b> 张` : "",
    ].filter(Boolean).join(" · ");
    row.innerHTML = `
      <div class="msg-body">
        <div class="bubble event-bubble">
          <div class="event-title">事件 #${escapeHtml(String(ev.id))} · ${duty ? "值守区无人" : "危险区入侵"}</div>
          <div class="event-meta">${span}</div>
          <div class="event-meta">${stat}</div>
          ${ev.evidence ? `<a class="event-shot" href="${escapeAttr(ev.evidence)}" target="_blank"><img src="${escapeAttr(ev.evidence)}" alt="事件取证图" /></a>` : ""}
          <div class="event-note">${ev.sample
            ? "左侧正在回放该事件的视频片段；下方是逐帧抽帧结果，可点击任意帧跳转。"
            : "该事件没有关联源视频，已载入抽帧片段，可点击下方任意帧查看。"}</div>
          <div class="event-actions">
            ${ev.sample ? '<button type="button" data-act="replay">回放这一段</button>' : ""}
            <button type="button" data-act="clip">生成证据片段 MP4</button>
          </div>
        </div>
      </div>`;
    row.querySelector('[data-act="replay"]')?.addEventListener("click", () => {
      const v = els.stageVideo;
      if (!v || state.media?.kind !== "video") return;
      try {
        v.currentTime = Math.max(0, Number(ev.video_enter) || 0);
        v.play().catch(() => {});
      } catch { /* ignore */ }
    });
    row.querySelector('[data-act="clip"]')?.addEventListener("click", async (e) => {
      const btn = e.currentTarget;
      btn.disabled = true;
      btn.textContent = "切片中…";
      try {
        const res = await fetch(`/api/guard/clip/${encodeURIComponent(ev.id)}`, { cache: "no-store" });
        const d = await res.json();
        if (!res.ok) throw new Error(d.error || `HTTP ${res.status}`);
        btn.textContent = "已生成 · 打开片段";
        btn.disabled = false;
        btn.onclick = () => window.open(d.clip, "_blank");
        toast("证据片段已生成");
      } catch (err) {
        btn.textContent = "切片失败";
        btn.disabled = false;
        toast(`无法生成片段：${err.message || err}`);
      }
    });
    els.messages.append(row);
    scrollToBottom();
  }

  async function whenStageReady(timeout = 4500) {
    const t0 = performance.now();
    while (performance.now() - t0 < timeout) {
      const media = state.media;
      if (!media || media.kind !== "video") return;
      const v = els.stageVideo;
      const startAt = Number.isFinite(media.startAt) ? media.startAt : 0;
      if (v.readyState >= 2 && Math.abs((v.currentTime || 0) - startAt) < 0.5) return;
      await wait(120);
    }
  }

  function renderStage() {
    const media = state.media;
    if (!media) return;

    els.stageEmpty.hidden = true;
    els.mediaBox.hidden = false;
    els.mediaBox.style.animation = "none";
    void els.mediaBox.offsetWidth;
    els.mediaBox.style.animation = "";

    if (media.kind === "video") {
      els.stageImage.hidden = true;
      els.stageVideo.hidden = false;
      els.videoBar.hidden = false;
      els.stageVideo.poster = media.poster || "";
      els.stageVideo.src = media.url;
      els.stageVideo.load();
      els.videoBar.classList.remove("playing");
      const startAt = Number.isFinite(media.startAt) ? media.startAt : null;
      if (startAt !== null) {
        // 事件跳转：定位到事发时刻并停住，上方框与画面严格同帧
        const seekOnce = () => {
          try { els.stageVideo.currentTime = Math.max(0, startAt); } catch { /* ignore */ }
          els.stageVideo.pause();
          updateVideoUI();
        };
        els.stageVideo.addEventListener("loadeddata", seekOnce, { once: true });
        els.stageVideo.pause();
        setTimeout(seekOnce, 500);
      } else {
        els.stageVideo.play().catch(() => {});
      }
    } else {
      els.stageVideo.pause();
      els.stageVideo.hidden = true;
      els.videoBar.hidden = true;
      els.stageImage.hidden = false;
      els.stageImage.src = media.url;
      els.stageImage.onload = () => layoutMedia();
    }

    layoutMedia();
    requestAnimationFrame(layoutMedia);
  }

  function layoutMedia() {
    const media = state.media;
    if (!media) return;
    const stage = els.stage;
    const padX = 56;
    const padY = 56;
    const reserved = media.kind === "video" ? 36 : 0;
    const cw = Math.max(160, stage.clientWidth - padX);
    const ch = Math.max(120, stage.clientHeight - padY - reserved);
    const ar = media.width / media.height || 1;
    let w = cw;
    let h = w / ar;
    if (h > ch) {
      h = ch;
      w = h * ar;
    }
    els.mediaBox.style.width = `${Math.round(w)}px`;
    els.mediaBox.style.height = `${Math.round(h)}px`;
  }

  function updateHud() {
    const media = state.media;
    if (!media) return;
    els.mediaType.textContent = media.kind === "video" ? "VIDEO" : "IMAGE";
    const dur = media.kind === "video" ? media.duration || els.stageVideo.duration || 0 : 0;
    els.mediaDim.textContent =
      media.kind === "video"
        ? `${media.width} × ${media.height} · ${Number(dur).toFixed(1)}s`
        : `${media.width} × ${media.height}`;

    const count = state.detections.length;
    const tracking = media.kind === "video" && state.detections.some((d) => d.track);
    els.targetCount.textContent = count ? `${count} 个目标${tracking ? " · 跟踪中" : ""}` : "等待提问";
    els.targetCount.classList.toggle("live", Boolean(tracking));
  }

  /* ------------------------------------------------------------- boxes */
  function clearOverlay() {
    els.overlay.innerHTML = "";
    state.detEls.clear();
  }

  function boxAt(obj, t) {
    const track = obj.track;
    if (!track || !track.length) return obj.box;
    if (t < track[0].t - 0.03 || t > track[track.length - 1].t + 0.03) return null;
    if (t <= track[0].t) return track[0].box;
    for (let i = 0; i < track.length - 1; i += 1) {
      const a = track[i];
      const b = track[i + 1];
      if (t >= a.t && t <= b.t) {
        const k = (t - a.t) / (b.t - a.t || 1);
        return a.box.map((v, j) => v + (b.box[j] - v) * k);
      }
    }
    return track[track.length - 1].box;
  }

  function applyBox(el, box) {
    el.style.left = `${box[0] * 100}%`;
    el.style.top = `${box[1] * 100}%`;
    el.style.width = `${Math.max(0.004, box[2] - box[0]) * 100}%`;
    el.style.height = `${Math.max(0.004, box[3] - box[1]) * 100}%`;
    el.classList.toggle("below", box[1] < 0.055);
  }

  function setDetections(objects) {
    state.detections = Array.isArray(objects) ? objects : [];
    clearOverlay();
    const compact = state.detections.length > 3;
    state.detections.forEach((obj, index) => {
      const el = document.createElement("div");
      el.className = compact ? "det compact" : "det";
      el.dataset.id = obj.id;
      el.style.setProperty("--c", obj.color || "#5eead4");
      el.style.animationDelay = `${index * 70}ms`;
      const tag = document.createElement("span");
      tag.className = "det-tag";
      tag.innerHTML = `<b>${index + 1}</b><span class="tag-label">${escapeHtml(obj.short || obj.label || "目标")}</span>`;
      el.append(tag);

      const box = obj.track ? boxAt(obj, currentVideoTime()) : obj.box;
      if (!box) el.classList.add("hidden");
      else applyBox(el, box);

      el.addEventListener("mouseenter", () => highlightFinding(obj.id, true));
      el.addEventListener("mouseleave", () => highlightFinding(obj.id, false));
      el.addEventListener("click", (ev) => {
        ev.stopPropagation();
        focusFinding(obj.id);
      });

      els.overlay.append(el);
      state.detEls.set(obj.id, el);
    });
    updateHud();
    if (!state.showBoxes) els.overlay.style.visibility = "hidden";
  }

  function pulseBox(id) {
    const el = state.detEls.get(id);
    if (!el || el.classList.contains("hidden")) {
      toast("该目标在当前时间点不可见");
      return;
    }
    el.classList.remove("pulse");
    void el.offsetWidth;
    el.classList.add("pulse");
    setTimeout(() => el.classList.remove("pulse"), 900);
  }

  function highlightFinding(id, on) {
    document.querySelectorAll(`.finding[data-id="${CSS.escape(id)}"]`).forEach((row) => {
      row.classList.toggle("flash", on);
    });
  }

  function focusFinding(id) {
    const rows = document.querySelectorAll(`.finding[data-id="${CSS.escape(id)}"]`);
    const row = rows[rows.length - 1];
    if (!row) return;
    row.scrollIntoView({ block: "nearest", behavior: "smooth" });
    row.classList.add("flash");
    setTimeout(() => row.classList.remove("flash"), 1100);
  }

  function currentVideoTime() {
    return state.media?.kind === "video" ? els.stageVideo.currentTime || 0 : 0;
  }

  function updateTrackedBoxes() {
    if (!state.detections.length) return;
    const t = currentVideoTime();
    state.detections.forEach((obj) => {
      if (!obj.track) return;
      const el = state.detEls.get(obj.id);
      if (!el) return;
      const box = boxAt(obj, t);
      if (!box) {
        el.classList.add("hidden");
      } else {
        el.classList.remove("hidden");
        applyBox(el, box);
      }
    });
  }

  /* ------------------------------------------------------------- video */
  function updateVideoUI() {
    const media = state.media;
    if (!media || media.kind !== "video" || els.videoBar.hidden) return;
    const v = els.stageVideo;
    const dur = v.duration || media.duration || 0;
    const cur = v.currentTime || 0;
    const ratio = dur ? Math.min(1, Math.max(0, cur / dur)) : 0;
    els.vFill.style.width = `${ratio * 100}%`;
    els.vThumb.style.left = `${ratio * 100}%`;
    els.vTime.textContent = `${cur.toFixed(1)}s / ${dur ? dur.toFixed(1) : "0.0"}s`;
  }

  function startVideoLoop() {
    cancelAnimationFrame(state.raf);
    const tick = () => {
      if (state.media?.kind === "video") {
        const m = state.media;
        if (Number.isFinite(m.segEnd) && Number.isFinite(m.segStart)
            && els.stageVideo.currentTime > m.segEnd + 0.05 && !els.stageVideo.paused) {
          try { els.stageVideo.currentTime = m.segStart; } catch { /* ignore */ }
        }
        updateVideoUI();
        updateTrackedBoxes();
      }
      state.raf = requestAnimationFrame(tick);
    };
    state.raf = requestAnimationFrame(tick);
  }

  function seekTo(clientX) {
    const v = els.stageVideo;
    const dur = v.duration || state.media?.duration || 0;
    if (!dur) return;
    const rect = els.vProgress.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    v.currentTime = ratio * dur;
    updateVideoUI();
    updateTrackedBoxes();
  }

  /* ------------------------------------------------------------- chat */
  function addDivider(text) {
    const div = document.createElement("div");
    div.className = "divider";
    div.textContent = text;
    els.messages.append(div);
  }

  function addUserMessage(text) {
    const row = document.createElement("div");
    row.className = "msg user";
    row.innerHTML = `<div class="msg-body"><div class="bubble">${formatText(text)}</div></div>`;
    els.messages.append(row);
  }

  function buildFindings(objects) {
    const wrap = document.createElement("div");
    wrap.className = "findings";
    wrap.innerHTML = `<div class="findings-head"><span>DETECTIONS · 定位结果</span><span class="count">${objects.length} 个目标</span></div>`;

    objects.forEach((obj, index) => {
      const row = document.createElement("div");
      row.className = "finding";
      row.dataset.id = obj.id;
      row.style.setProperty("--c", obj.color || "#5eead4");
      row.innerHTML = `
        <span class="f-index">${index + 1}</span>
        <div class="f-main">
          <div class="f-label">${escapeHtml(obj.label || obj.short || "目标")}</div>
          <div class="f-where">${escapeHtml(obj.where || "画面内")}</div>
        </div>
        <span class="f-score">${Number(obj.score || 0).toFixed(2)}</span>
      `;
      row.addEventListener("click", () => pulseBox(obj.id));
      row.addEventListener("mouseenter", () => {
        const el = state.detEls.get(obj.id);
        if (el && !el.classList.contains("hidden")) el.classList.add("pulse");
      });
      row.addEventListener("mouseleave", () => {
        const el = state.detEls.get(obj.id);
        if (el) el.classList.remove("pulse");
      });
      wrap.append(row);
    });
    return wrap;
  }

  function buildTrace(trace, meta) {
    if (!Array.isArray(trace) || !trace.length) return null;
    const thought = trace.find((t) => t.type === "thought");
    const plan = trace.find((t) => t.type === "plan");
    const tools = trace.filter((t) => t.type === "tool" && t.tool !== "chat");
    const recoveries = trace.filter((t) => t.type === "recovery");
    const memory = tools.some((t) => t.tool === "resolve" || t.tool === "relative");
    const toolName = { ground: "定位", count: "计数", describe: "概述", relative: "相对位置", resolve: "指代解析", clarify: "追问" };
    const argsText = (t) => {
      const a = t.args || {};
      if (t.tool === "ground") return a.target || "";
      if (t.tool === "count") return a.category || "";
      if (t.tool === "relative") {
        const d = { left: "左侧", right: "右侧", above: "上方", below: "下方", side: "旁边" }[a.direction] || a.direction || "";
        return `${a.target || ""} · ${d}`;
      }
      if (t.tool === "resolve") return a.phrase || "";
      return "";
    };
    const byText = (i) => {
      const by = plan?.steps?.[i]?.chosen_by;
      if (by === "agent") return '<em class="trace-by">Agent 决策</em>';
      if (by === "guardrail") return '<em class="trace-by rule">规则护栏</em>';
      if (by === "qwen" || by === "qwen-json") {
        return `<em class="trace-by">${by === "qwen-json" ? "Qwen JSON" : "Qwen 决策"}</em>`;
      }
      if (by === "rule") return '<em class="trace-by rule">规则兜底</em>';
      return "";
    };

    const wrap = document.createElement("div");
    wrap.className = "trace";
    const rows = tools.map((t, i) => {
      const status = t.status === "ok" ? "ok" : t.status === "empty" ? "empty" : "error";
      return `<div class="trace-step ${status}">
        <span class="trace-dot"></span>
        <div class="trace-step-main">
          <div class="trace-tool"><b>${toolName[t.tool] || t.tool}</b><span>${escapeHtml(argsText(t))}</span>${byText(i)}</div>
          <div class="trace-obs">${escapeHtml(t.observation || "完成")}</div>
        </div>
        <span class="trace-time">${t.duration_ms ? `${t.duration_ms}ms` : ""}</span>
      </div>`;
    }).join("");
    const recRows = recoveries.map((r) => `<div class="trace-recovery">${escapeHtml(r.text || "")}</div>`).join("");
    const duration = meta?.latency_ms ? `${(meta.latency_ms / 1000).toFixed(1)}s` : "";
    wrap.innerHTML = `
      <div class="trace-head">
        <span class="trace-title"><span class="trace-icon">✦</span> 思考过程</span>
        <span class="trace-tags">${tools.length ? `${tools.length} 个工具` : "无需工具"}${memory ? "<em>引用上文</em>" : ""}${duration ? `<i>${duration}</i>` : ""}</span>
        <button type="button" class="trace-toggle">收起</button>
      </div>
      <div class="trace-body">
        ${thought?.text ? `<div class="trace-thought">${escapeHtml(thought.text)}</div>` : ""}
        <div class="trace-steps">${rows}${recRows}</div>
      </div>`;
    wrap.querySelector(".trace-toggle").addEventListener("click", (ev) => {
      wrap.classList.toggle("collapsed");
      ev.currentTarget.textContent = wrap.classList.contains("collapsed") ? "展开" : "收起";
    });
    return wrap;
  }

  function addAgentMessage(data, clientLatency) {
    const reply = data.reply || "（没有返回内容）";
    const objects = Array.isArray(data.objects) ? data.objects : [];
    const meta = data.meta || {};

    const row = document.createElement("div");
    row.className = "msg agent";
    row.innerHTML = `<div class="mini-avatar">${AGENT_SVG}</div><div class="msg-body"><div class="bubble"></div></div>`;

    const traceEl = buildTrace(data.trace, meta);
    if (traceEl) $(".msg-body", row).prepend(traceEl);

    const bubble = $(".bubble", row);
    bubble.innerHTML = formatText(reply);
    if (objects.length) bubble.append(buildFindings(objects));

    const engineName = meta.demo === false ? meta.model || "RexSeek-3B" : "RexSeek-3B";
    const seconds = (Math.max(0, clientLatency) / 1000).toFixed(2);
    const metaRow = document.createElement("div");
    metaRow.className = "msg-meta";
    const tag = meta.cached ? "缓存" : meta.demo === false ? "在线" : "演示";
    metaRow.innerHTML = `<span>${escapeHtml(engineName)}</span><span class="sep"></span><span>${tag}</span><span class="sep"></span><span>${seconds}s</span>`;

    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "copy-btn";
    copyBtn.innerHTML = `${COPY_SVG}<span>复制</span>`;
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(reply);
        copyBtn.querySelector("span").textContent = "已复制";
        setTimeout(() => (copyBtn.querySelector("span").textContent = "复制"), 1600);
      } catch {
        toast("复制失败，请手动选择文本");
      }
    });
    metaRow.append(copyBtn);

    $(".msg-body", row).append(metaRow);
    els.messages.append(row);
  }

  function addError(message) {
    const row = document.createElement("div");
    row.className = "msg agent error";
    row.innerHTML = `<div class="mini-avatar">${AGENT_SVG}</div><div class="msg-body"><div class="bubble">${formatText(message)}</div></div>`;
    els.messages.append(row);
  }

  function addThinking() {
    const steps = ["理解问题", "检索画面", "组织回答"];
    const row = document.createElement("div");
    row.className = "msg agent";
    row.innerHTML = `
      <div class="mini-avatar">${AGENT_SVG}</div>
      <div class="msg-body"><div class="bubble thinking-bubble">
        <div class="think-steps">
          ${steps.map((s, i) => `<div class="think-step" data-i="${i}"><i></i><span>${s}</span></div>`).join("")}
        </div>
      </div></div>`;
    els.messages.append(row);
    const stepEls = [...row.querySelectorAll(".think-step")];
    const timers = [];
    stepEls[0].classList.add("active");
    timers.push(
      setTimeout(() => {
        stepEls[0].classList.remove("active");
        stepEls[0].classList.add("done");
        stepEls[1].classList.add("active");
      }, 430)
    );
    timers.push(
      setTimeout(() => {
        stepEls[1].classList.remove("active");
        stepEls[1].classList.add("done");
        stepEls[2].classList.add("active");
      }, 900)
    );
    scrollToBottom();
    return {
      remove() {
        timers.forEach(clearTimeout);
        row.remove();
      },
    };
  }

  function renderWelcome() {
    const media = state.media;
    if (!media) return;
    const sample = media.sampleId ? state.sampleMap.get(media.sampleId) : null;

    if (sample) {
      const live = state.engine && state.engine.demo === false;
      els.welcomeDesc.innerHTML = live
        ? `当前画面是「<strong>${escapeHtml(sample.title)}</strong>」——${escapeHtml(
            sample.subtitle
          )}。用自然语言指代画面里的任意人、物或区域，我会用 RexSeek-3B 实时定位并框出结果。`
        : `当前画面是「<strong>${escapeHtml(sample.title)}</strong>」——${escapeHtml(
            sample.subtitle
          )}。用自然语言指代画面里的任意人、物或区域，我会把对应位置框出来，并给出判断依据。`;
      els.welcomeScene.textContent = sample.title;
      setSuggestions(suggestionsFor(sample));
    } else {
      const live = state.engine && state.engine.demo === false;
      els.welcomeDesc.innerHTML = live
        ? `已载入自定义素材「<strong>${escapeHtml(
            media.title
          )}</strong>」。直接提问即可，右侧会返回真实 RexSeek 的定位结果。`
        : `已载入自定义素材「<strong>${escapeHtml(
            media.title
          )}</strong>」。当前为演示模式，自定义素材暂不参与定位推理；接入真实 RexSeek 后端后即可直接提问。`;
      els.welcomeScene.textContent = "自定义素材";
      setSuggestions([]);
    }
  }

  function suggestionsFor(sample) {
    if (!sample) return [];
    const live = state.engine && state.engine.demo === false;
    if (live && Array.isArray(sample.live_suggestions) && sample.live_suggestions.length) {
      return sample.live_suggestions;
    }
    return sample.suggestions || [];
  }

  function setSuggestions(list) {
    els.welcomeSuggestions.innerHTML = list
      .map(
        (text) => `<button type="button" data-prompt="${escapeAttr(text)}"><span>${escapeHtml(
          text
        )}</span>${ARROW_SVG}</button>`
      )
      .join("");
  }

  function renderQuick() {
    const media = state.media;
    const sample = media?.sampleId ? state.sampleMap.get(media.sampleId) : null;
    const list = suggestionsFor(sample);
    const show = list.length > 0 && state.history.length > 0;
    els.quickRow.hidden = !show;
    if (!show) {
      els.quickRow.innerHTML = "";
      return;
    }
    els.quickRow.innerHTML = list
      .map((text) => `<button type="button" class="quick-chip" data-prompt="${escapeAttr(text)}">${escapeHtml(text)}</button>`)
      .join("");
  }

  /* ------------------------------------------------------------- visual index */
  function setIndexState(text, cls = "") {
    if (!els.indexChip) return;
    els.indexChip.hidden = !text;
    els.indexChip.textContent = text;
    els.indexChip.className = "hud-chip" + (cls ? ` ${cls}` : "");
  }

  async function indexMedia() {
    const media = state.media;
    if (!media || media.kind !== "image" || !state.engine || state.engine.demo) {
      setIndexState("");
      return;
    }
    if (state.indexAbort) state.indexAbort.abort();
    const ctrl = new AbortController();
    state.indexAbort = ctrl;
    setIndexState("索引中", "indexing");
    try {
      const dataUrl = media.uploaded ? media.dataUrl : await imageToDataUrl(media.url);
      const res = await fetch("/api/index", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: ctrl.signal,
        body: JSON.stringify({
          media: {
            sample_id: media.sampleId,
            kind: media.kind,
            name: media.title,
            width: media.width,
            height: media.height,
            data_url: dataUrl,
          },
        }),
      });
      if (!res.ok) throw new Error(`index ${res.status}`);
      const data = await res.json();
      if (state.media !== media) return;
      setIndexState(data.cached ? "已缓存" : "已索引", "cached");
    } catch (err) {
      if (err.name === "AbortError") return;
      setIndexState("索引失败", "");
    }
  }

  /* ------------------------------------------------------------- payload */
  async function imageToDataUrl(url) {
    if (state.imageDataCache.has(url)) return state.imageDataCache.get(url);
    const promise = new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const max = 1280;
        const scale = Math.min(1, max / Math.max(img.naturalWidth, img.naturalHeight));
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
        canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
        canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL("image/jpeg", 0.86));
      };
      img.onerror = () => resolve(null);
      img.src = url;
    });
    state.imageDataCache.set(url, promise);
    return promise;
  }

  function captureVideoFrame() {
    const v = els.stageVideo;
    if (!v.videoWidth) return null;
    const max = 1280;
    const scale = Math.min(1, max / Math.max(v.videoWidth, v.videoHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(v.videoWidth * scale);
    canvas.height = Math.round(v.videoHeight * scale);
    try {
      canvas.getContext("2d").drawImage(v, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL("image/jpeg", 0.86);
    } catch {
      return null;
    }
  }

  async function buildPayload(text) {
    const media = state.media;
    let dataUrl = media.dataUrl || null;
    if (!dataUrl && media.kind === "image") {
      dataUrl = await imageToDataUrl(media.url);
    }
    if (media.kind === "video") {
      dataUrl = captureVideoFrame();
      if (!dataUrl) {
        await new Promise((resolve) => {
          if (els.stageVideo.readyState >= 2) return resolve();
          els.stageVideo.addEventListener("loadeddata", resolve, { once: true });
          setTimeout(resolve, 1200);
        });
        dataUrl = captureVideoFrame();
      }
    }
    return {
      session_id: state.sessionId,
      messages: [...state.history, { role: "user", content: text }].map(({ role, content }) => ({ role, content })),
      media: {
        sample_id: media.sampleId,
        kind: media.kind,
        name: media.title,
        width: media.width,
        height: media.height,
        data_url: dataUrl,
      },
      video_time: media.kind === "video" ? els.stageVideo.currentTime || 0 : 0,
      event_id: state.eventRef?.id ?? null,
    };
  }

  /* ------------------------------------------------------------- send */
  async function send(text) {
    const content = (text ?? els.input.value).trim();
    if (!content || state.busy) return;
    if (state.media?.kind === "video" && state.engine && state.engine.demo === false) {
      els.stageVideo.pause();
    }
    if (!state.media) {
      toast("请先选择或上传一个画面");
      return;
    }
    if (state.engine && state.engine.demo === false) {
      return sendStream(content);
    }

    state.busy = true;
    updateComposer();
    els.input.value = "";
    autosize();
    hideWelcome();
    addUserMessage(content);
    scrollToBottom();
    const thinking = addThinking();
    els.stage.classList.add("analyzing");

    const started = performance.now();
    try {
      const payload = await buildPayload(content);
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        let message = `请求失败（${res.status}）`;
        try {
          const body = await res.json();
          if (body.error) message = body.error;
        } catch {
          /* ignore */
        }
        throw new Error(message);
      }
      const data = await res.json();
      const elapsed = performance.now() - started;
      const minThink = state.engine && state.engine.demo ? 300 : 0;
      await wait(Math.max(0, minThink - elapsed));
      thinking.remove();

      state.history.push({ role: "user", content });
      state.history.push({ role: "assistant", content: data.reply || "" });
      addAgentMessage(data, data.meta?.latency_ms || performance.now() - started);
      setDetections(data.objects || []);
      scrollToBottom();
      renderQuick();
    } catch (err) {
      thinking.remove();
      addError(`**请求没有完成。** ${err.message || "未知错误"}\n\n请确认本地服务仍在运行，或稍后重试。`);
      scrollToBottom();
      toast("请求失败");
    } finally {
      state.busy = false;
      els.stage.classList.remove("analyzing");
      updateComposer();
      els.input.focus();
    }
  }


  async function consumeSSE(response, onEvent) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        for (const line of block.split(/\r?\n/)) {
          if (!line.startsWith("data:")) continue;
          const raw = line.slice(5).trim();
          if (raw) onEvent(JSON.parse(raw));
        }
      }
    }
  }

  function createLiveMessage() {
    const row = document.createElement("div");
    row.className = "msg agent live";
    row.innerHTML = `<div class="mini-avatar">${AGENT_SVG}</div>
      <div class="msg-body">
        <div class="live-status"><span class="live-dot"></span><span class="live-status-text">收到，先看一下画面…</span></div>
        <div class="live-trace"></div>
        <div class="bubble live-bubble"></div>
      </div>`;
    els.messages.append(row);
    const statusEl = row.querySelector(".live-status-text");
    const traceEl = row.querySelector(".live-trace");
    const bubble = row.querySelector(".live-bubble");
    const toolName = { ground: "定位", count: "计数", describe: "概述", relative: "相对位置", resolve: "指代解析", clarify: "追问" };

    const addStep = (text, cls = "") => {
      const line = document.createElement("div");
      line.className = "live-step" + (cls ? ` ${cls}` : "");
      line.innerHTML = text;
      traceEl.append(line);
    };

    return {
      status(text) {
        statusEl.textContent = text;
        scrollToBottom(false);
      },
      thought(text) {
        if (text) addStep(`<b>✦</b><span>${escapeHtml(text)}</span>`);
      },
      plan(steps) {
        const names = (steps || []).map((s) => toolName[s.tool] || s.tool).join(" → ");
        if (names) addStep(`<b>计划</b><span>${escapeHtml(names)}</span>`);
      },
      toolStart(ev) {
        this.status(`正在${toolName[ev.tool] || ev.tool}…`);
      },
      tool(ev) {
        const ok = ev.status === "ok";
        const label = toolName[ev.tool] || ev.tool;
        addStep(`<b class="${ok ? "ok" : ""}">${ok ? "✓" : "!"}</b><span>${escapeHtml(label)} · ${escapeHtml(ev.observation || "")}${ev.duration_ms ? ` · ${ev.duration_ms}ms` : ""}</span>`);
      },
      recovery(text) {
        addStep(`<b style="color:#ffb547">!</b><span style="color:#ffd7a1">${escapeHtml(text)}</span>`);
      },
      token(text) {
        if (!bubble.textContent) bubble.classList.add("has-text");
        bubble.textContent += text;
        this.status("正在整理回答…");
        scrollToBottom(false);
      },
      remove() {
        row.remove();
      },
    };
  }

  async function sendStream(content) {
    state.busy = true;
    updateComposer();
    els.input.value = "";
    autosize();
    hideWelcome();
    addUserMessage(content);
    scrollToBottom();
    els.stage.classList.add("analyzing");

    const live = createLiveMessage();
    const started = performance.now();
    let payload;
    let finalData = null;

    try {
      payload = await buildPayload(content);
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok || !res.body) {
        let message = `请求失败（${res.status}）`;
        try {
          const body = await res.json();
          if (body.error) message = body.error;
        } catch {
          /* ignore */
        }
        throw new Error(message);
      }

      await consumeSSE(res, (ev) => {
        if (ev.type === "thought") live.thought(ev.text);
        else if (ev.type === "plan") live.plan(ev.steps);
        else if (ev.type === "tool_start") live.toolStart(ev);
        else if (ev.type === "tool") live.tool(ev);
        else if (ev.type === "recovery") live.recovery(ev.text);
        else if (ev.type === "boxes") {
          setDetections(ev.objects || []);
          live.status(`已找到 ${ev.objects?.length || 0} 个目标，正在整理回答…`);
        } else if (ev.type === "token") live.token(ev.text || "");
        else if (ev.type === "final") finalData = ev;
        else if (ev.type === "error") throw new Error(ev.error || "推理服务错误");
      });

      if (!finalData) throw new Error("流式响应没有返回最终结果");
      live.remove();
      state.history.push({ role: "user", content });
      state.history.push({ role: "assistant", content: finalData.reply || "" });
      addAgentMessage(finalData, finalData.meta?.latency_ms || performance.now() - started);
      setDetections(finalData.objects || []);
      scrollToBottom();
      renderQuick();
    } catch (err) {
      live.remove();
      addError(`**请求没有完成。** ${err.message || "未知错误"}\n\n请确认本地服务仍在运行，或稍后重试。`);
      scrollToBottom();
      toast("请求失败");
    } finally {
      state.busy = false;
      els.stage.classList.remove("analyzing");
      updateComposer();
      els.input.focus();
    }
  }

  /* ------------------------------------------------------------- upload */
  function imageDims(url) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
      img.onerror = () => resolve({ width: 1280, height: 720 });
      img.src = url;
    });
  }

  function videoDims(url) {
    return new Promise((resolve) => {
      const v = document.createElement("video");
      v.preload = "metadata";
      v.muted = true;
      const done = () =>
        resolve({ width: v.videoWidth || 1280, height: v.videoHeight || 720, duration: v.duration || 0 });
      v.onloadedmetadata = done;
      v.onerror = () => resolve({ width: 1280, height: 720, duration: 0 });
      setTimeout(done, 3000);
      v.src = url;
    });
  }

  async function downscaleImage(url, max) {
    const img = await new Promise((resolve) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => resolve(null);
      el.src = url;
    });
    if (!img) return null;
    const scale = Math.min(1, max / Math.max(img.naturalWidth, img.naturalHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#0a0d12";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.86);
  }

  function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(fr.result);
      fr.onerror = () => reject(fr.error || new Error("读取文件失败"));
      fr.readAsDataURL(file);
    });
  }

  // 上传一次 mp4 → 固化成本地默认素材（两页共用），事件跳转才能回到同一段视频
  async function persistVideo(file) {
    if (!file || !file.type.startsWith("video/") || state.uploading) return;
    if (file.size > 118 * 1024 * 1024) return toast("视频超过 120MB，本次只在本页播放，不保存为默认素材");
    state.uploading = true;
    toast("正在保存为本地默认素材…");
    try {
      const dataUrl = await fileToDataUrl(file);
      const poster = captureVideoFrame();
      const v = els.stageVideo;
      const res = await fetch("/api/media/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: file.name,
          kind: "video",
          data_url: dataUrl,
          poster,
          width: v?.videoWidth || state.media?.width || 0,
          height: v?.videoHeight || state.media?.height || 0,
          duration: v?.duration || state.media?.duration || 0,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      if (state.media) {
        state.media.sampleId = data.sample.id;
        state.media.title = data.sample.title;
        state.media.poster = data.sample.poster;
      }
      try {
        const catalog = await fetch("/api/samples", { cache: "no-store" }).then((r) => r.json());
        if (catalog?.samples) {
          state.samples = catalog.samples;
          state.sampleMap = new Map(catalog.samples.map((s) => [s.id, s]));
          renderStrip();
          renderStripActive();
          if (els.stripHint) els.stripHint.textContent = `${state.samples.length} 组预置场景`;
        }
      } catch { /* ignore */ }
      toast("已保存为本地默认素材：值守台也默认用它");
    } catch (err) {
      toast(`保存默认素材失败：${err.message || err}`);
    } finally {
      state.uploading = false;
    }
  }

  async function handleFiles(fileList) {
    const file = fileList?.[0];
    if (!file) return;
    if (state.busy) return toast("等当前回答结束再上传");
    const isVideo = file.type.startsWith("video/");
    const isImage = file.type.startsWith("image/");
    if (!isVideo && !isImage) return toast("只支持图片或视频文件");
    if (file.size > 120 * 1024 * 1024) return toast("文件太大（上限 120MB）");
    const url = URL.createObjectURL(file);
    const dims = isVideo ? await videoDims(url) : await imageDims(url);
    const dataUrl = isImage ? await downscaleImage(url, 1280) : null;

    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = url;
    state.media = {
      kind: isVideo ? "video" : "image",
      sampleId: null,
      url,
      width: dims.width,
      height: dims.height,
      duration: dims.duration || 0,
      title: file.name,
      name: file.name,
      poster: null,
      uploaded: true,
      dataUrl,
    };
    state.eventRef = null;
    state.detections = [];
    clearOverlay();
    renderStage();
    updateHud();
    renderStripActive();
    renderWelcome();
    renderQuick();
    resetSession();
    indexMedia();
    if (state.history.length) addDivider(`已载入自定义素材「${file.name}」`);
    scrollToBottom();
    if (state.engine && state.engine.demo === false) {
      toast(isVideo ? "已载入视频，将对当前帧做真实推理" : "已载入图片，可直接提问");
    } else {
      toast(isVideo ? "已载入视频；演示模式下仅示例素材支持跟踪" : "已载入图片；演示模式下仅示例素材支持定位");
    }
    if (isVideo) setTimeout(() => persistVideo(file), 600);
  }

  /* ------------------------------------------------------------- reset */
  function resetChat() {
    if (state.busy) return toast("等当前回答结束再重新开始");
    state.history = [];
    resetSession();
    [...els.messages.children].forEach((node) => {
      if (node.id !== "welcome") node.remove();
    });
    els.welcome.hidden = false;
    state.detections = [];
    clearOverlay();
    updateHud();
    renderWelcome();
    renderQuick();
    els.input.value = "";
    autosize();
    updateComposer();
    toast("已重新开始");
    els.input.focus();
  }

  function restartAll() {
    if (state.busy) return toast("等当前回答结束再重新开始");
    resetChat();
    if (state.samples.length) loadSample(state.samples[0].id, { silent: true });
    els.chatScroll.scrollTop = 0;
  }

  /* ------------------------------------------------------------- events */
  function bindEvents() {
    els.sendBtn.addEventListener("click", () => send());
    els.input.addEventListener("input", () => {
      autosize();
      updateComposer();
    });
    els.input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) {
        ev.preventDefault();
        send();
      }
    });

    els.welcomeSuggestions.addEventListener("click", (ev) => {
      const btn = ev.target.closest("button[data-prompt]");
      if (btn) send(btn.dataset.prompt);
    });
    els.quickRow.addEventListener("click", (ev) => {
      const btn = ev.target.closest("button[data-prompt]");
      if (btn) send(btn.dataset.prompt);
    });

    els.uploadBtn.addEventListener("click", () => els.fileInput.click());
    els.fileInput.addEventListener("change", (ev) => {
      handleFiles(ev.target.files);
      ev.target.value = "";
    });

    els.toggleBoxesBtn.addEventListener("click", () => {
      state.showBoxes = !state.showBoxes;
      els.overlay.style.visibility = state.showBoxes ? "visible" : "hidden";
      els.toggleBoxesBtn.classList.toggle("is-active", state.showBoxes);
      els.toggleBoxesBtn.classList.toggle("is-off", !state.showBoxes);
      els.toggleBoxesBtn.setAttribute("aria-pressed", String(state.showBoxes));
      if (!state.showBoxes) toast("已隐藏标注");
    });

    els.restartBtn.addEventListener("click", restartAll);
    els.clearBtn.addEventListener("click", resetChat);

    els.playBtn.addEventListener("click", () => {
      const v = els.stageVideo;
      if (v.paused) v.play().catch(() => {});
      else v.pause();
    });
    els.stageVideo.addEventListener("play", () => els.videoBar.classList.add("playing"));
    els.stageVideo.addEventListener("pause", () => els.videoBar.classList.remove("playing"));
    els.stageVideo.addEventListener("loadedmetadata", () => {
      if (state.media?.kind === "video") {
        state.media.duration = els.stageVideo.duration || state.media.duration;
        updateHud();
        layoutMedia();
      }
    });

    const progressPointerDown = (ev) => {
      if (!state.media || state.media.kind !== "video") return;
      state.seeking = true;
      els.vProgress.setPointerCapture?.(ev.pointerId);
      seekTo(ev.clientX);
    };
    const progressPointerMove = (ev) => {
      if (state.seeking) seekTo(ev.clientX);
    };
    const progressPointerUp = (ev) => {
      state.seeking = false;
      els.vProgress.releasePointerCapture?.(ev.pointerId);
    };
    els.vProgress.addEventListener("pointerdown", progressPointerDown);
    els.vProgress.addEventListener("pointermove", progressPointerMove);
    els.vProgress.addEventListener("pointerup", progressPointerUp);
    els.vProgress.addEventListener("pointercancel", progressPointerUp);

    // 拖拽上传
    ["dragenter", "dragover"].forEach((type) =>
      els.stage.addEventListener(type, (ev) => {
        ev.preventDefault();
        els.stage.classList.add("dragover");
      })
    );
    ["dragleave", "drop"].forEach((type) =>
      els.stage.addEventListener(type, (ev) => {
        ev.preventDefault();
        if (type === "dragleave" && els.stage.contains(ev.relatedTarget)) return;
        els.stage.classList.remove("dragover");
      })
    );
    els.stage.addEventListener("drop", (ev) => {
      const files = ev.dataTransfer?.files;
      if (files?.length) handleFiles(files);
    });

    // 键盘：/ 聚焦输入框
    document.addEventListener("keydown", (ev) => {
      if (ev.key === "/" && document.activeElement !== els.input && !ev.metaKey && !ev.ctrlKey) {
        ev.preventDefault();
        els.input.focus();
      }
    });

    const ro = new ResizeObserver(() => layoutMedia());
    ro.observe(els.stage);
    window.addEventListener("resize", layoutMedia);
  }

  /* ------------------------------------------------------------- go */
  init();
})();
