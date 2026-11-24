(() => {
  const $ = (s) => document.querySelector(s);
  const canvas = $("#scene"), ctx = canvas.getContext("2d");
  const DEFAULT_ZONE = [[0.28, 0.45], [0.78, 0.45], [0.88, 0.95], [0.18, 0.95]];
  const DEFAULT_DUTY = [[0.34, 0.60], [0.66, 0.60], [0.72, 0.96], [0.28, 0.96]];
  const state = {
    samples: [], media: null, source: null, mode: "auto", objects: [],
    guard: null, busy: false, session: localStorage.getItem("insight_session_v2") || "",
    frozenImage: null, frameTs: 0, detectMs: 0, tickNo: 0, tickMs: 2000, rate: 0.5, poster: null,
    episode: null, episodeKey: "", events: [], total: 0, uploading: false,
    zones: { danger: DEFAULT_ZONE.map((p) => p.slice()), duty: null },
    editKind: null, dragIdx: -1, zoneTimer: 0, alarm: { danger: false, duty: false },
  };
  if (!state.session) {
    state.session = "insight_" + Math.random().toString(36).slice(2, 10);
    localStorage.setItem("insight_session_v2", state.session);
  }

  /* ---------------------------------------------------------- 示例素材 */
  function videoPlaceholder() {
    const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='140' height='88'>
      <defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
      <stop stop-color='#12313f'/><stop offset='1' stop-color='#08131e'/></linearGradient></defs>
      <rect width='140' height='88' fill='url(#g)'/>
      <circle cx='70' cy='44' r='16' fill='none' stroke='#35e0ff' stroke-width='2'/>
      <path d='M65 37l13 7-13 7z' fill='#35e0ff'/></svg>`;
    return "data:image/svg+xml;utf8," + encodeURIComponent(svg);
  }
  function thumbFor(s) {
    if (s.kind !== "video") return s.file;
    return s.poster || videoPlaceholder();
  }

  async function loadSamples(keepSelection) {
    try {
      const r = await fetch("/api/samples", { cache: "no-store" });
      const d = await r.json();
      state.samples = d.samples || [];
      renderSampleThumbs();
      if (keepSelection) markActiveSample();
      else if (state.samples.length) selectSample(0);
    } catch (e) { $("#hint").textContent = "加载示例失败：" + e.message; }
  }

  function renderSampleThumbs() {
    const box = $("#samples");
    if (!box) return;
    box.innerHTML = state.samples.map((s, i) =>
      `<img class="sample" data-i="${i}" src="${thumbFor(s)}" title="${s.title}${s.kind === "video" ? " · 视频" : ""}">`
    ).join("");
    box.querySelectorAll(".sample").forEach((el) => {
      el.onclick = () => selectSample(+el.dataset.i);
      el.onerror = () => { el.src = videoPlaceholder(); };
    });
  }

  function markActiveSample() {
    const id = state.media?.sampleId;
    document.querySelectorAll(".sample").forEach((el) => {
      const s = state.samples[+el.dataset.i];
      el.classList.toggle("active", !!s && s.id === id);
    });
  }

  function selectSample(i) {
    stopGuard();
    const s = state.samples[i];
    if (!s) return;
    state.media = { kind: s.kind, title: s.title, sampleId: s.id, width: s.width, height: s.height, file: s.file, poster: s.poster };
    state.objects = [];
    state.episode = null;
    state.poster = null;
    markActiveSample();
    renderChips(s);
    if (s.kind === "video") {
      if (s.poster) {
        const p = new Image();
        p.onload = () => { state.poster = p; if (state.media && state.media.sampleId === s.id) drawScene(); };
        p.src = s.poster;
      }
      const v = document.createElement("video");
      v.src = s.file; v.poster = s.poster || ""; v.muted = true; v.loop = true; v.playsInline = true;
      v.onloadeddata = () => {
        state.source = v;
        if (!state.media.width) state.media.width = v.videoWidth;
        if (!state.media.height) state.media.height = v.videoHeight;
        drawScene();
        v.play().then(drawScene).catch(() => {});
        $("#hint").textContent = "视频已加载；可以开始值守（一次告警只记一条台账）";
      };
      v.oncanplay = drawScene; v.onseeked = drawScene; v.onplaying = drawScene;
      v.load();
    } else {
      const im = new Image();
      im.onload = () => { state.source = im; drawScene(); $("#hint").textContent = "图片已加载"; };
      im.src = s.file;
    }
  }

  function renderChips(sample) {
    const el = $("#chips"); if (!el) return;
    el.innerHTML = "";
    (sample?.live_suggestions || []).forEach((q) => {
      const b = document.createElement("button"); b.textContent = q;
      b.onclick = () => quick(q); el.append(b);
    });
    if (sample?.live_suggestions?.length && $("#query")) $("#query").value = sample.live_suggestions[0];
  }

  /* ------------------------------------------------------------ 上传素材 */
  function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(fr.result);
      fr.onerror = () => reject(fr.error || new Error("读取文件失败"));
      fr.readAsDataURL(file);
    });
  }

  function videoPoster(video) {
    try {
      const w = video.videoWidth || 1280, h = video.videoHeight || 720;
      const scale = Math.min(1, 1280 / Math.max(w, h));
      const c = document.createElement("canvas");
      c.width = Math.max(1, Math.round(w * scale));
      c.height = Math.max(1, Math.round(h * scale));
      c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
      return c.toDataURL("image/jpeg", 0.8);
    } catch (e) { return null; }
  }

  // 上传一次 → 固化成本地素材：值守台 / 对话分析共用，事件才能跳回同一段视频
  async function persistVideo(file) {
    if (!file || !file.type.startsWith("video/") || state.uploading) return;
    if (file.size > 118 * 1024 * 1024) {
      $("#hint").textContent = "视频超过 120MB，本次只在本页播放，不保存为默认素材";
      return;
    }
    state.uploading = true;
    $("#hint").textContent = "正在保存为本地默认素材（两页共用）…";
    try {
      const dataUrl = await fileToDataUrl(file);
      const v = state.source;
      const r = await fetch("/api/media/upload", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: file.name, kind: "video", data_url: dataUrl,
          poster: v && v.videoWidth ? videoPoster(v) : null,
          width: v?.videoWidth || 0, height: v?.videoHeight || 0, duration: v?.duration || 0,
        }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || ("HTTP " + r.status));
      if (state.media) {
        state.media.sampleId = d.sample.id;
        state.media.title = d.sample.title;
        state.media.poster = d.sample.poster;
      }
      $("#hint").textContent = "已保存为本地默认素材：对话分析默认也用它，事件跳转可定位到这段视频";
      loadSamples(true);
    } catch (e) {
      $("#hint").textContent = "保存默认素材失败：" + e.message;
    } finally {
      state.uploading = false;
    }
  }

  document.getElementById("file").addEventListener("change", (e) => {
    const f = e.target.files[0]; if (!f) return;
    stopGuard();
    const url = URL.createObjectURL(f);
    state.media = { kind: f.type.startsWith("video/") ? "video" : "image", title: f.name, sampleId: null, width: 0, height: 0, file: url };
    state.objects = [];
    state.episode = null;
    if (state.media.kind === "video") {
      const v = document.createElement("video"); v.src = url; v.muted = true; v.loop = true; v.playsInline = true;
      v.onloadeddata = () => {
        state.media.width = v.videoWidth; state.media.height = v.videoHeight;
        state.source = v; drawScene();
        $("#hint").textContent = "视频已加载；可开始塔吊值守";
        persistVideo(f);
      };
    } else {
      const im = new Image();
      im.onload = () => {
        state.media.width = im.naturalWidth; state.media.height = im.naturalHeight;
        state.source = im; drawScene(); $("#hint").textContent = "图片已加载";
      };
      im.src = url;
    }
  });

  /* ------------------------------------------------------------ 画面渲染 */
  function sourceSize() {
    if (!state.source) return [960, 540];
    const w = state.source.videoWidth || state.source.naturalWidth || 960;
    const h = state.source.videoHeight || state.source.naturalHeight || 540;
    return [w, h];
  }

  /* ------------------------------------------------ 危险区域（可拖拽多边形） */
  function editing() { return state.editKind ? state.zones[state.editKind] : null; }

  function drawZoneHandles() {
    const z = editing();
    if (!z) return;
    ctx.save();
    z.forEach((p, i) => {
      const x = p[0] * canvas.width, y = p[1] * canvas.height;
      ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.fillStyle = state.dragIdx === i ? "#ffffff" : (state.editKind === "duty" ? "#2ee6a8" : "#ffd84d");
      ctx.fill(); ctx.strokeStyle = "#08131e"; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = "#08131e"; ctx.font = "bold 10px system-ui"; ctx.textAlign = "center";
      ctx.fillText(String(i + 1), x, y + 3.5); ctx.textAlign = "left";
    });
    ctx.restore();
  }

  function drawPolygon(zone, color, dash) {
    if (!zone || zone.length < 3) return;
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = 3;
    if (dash) ctx.setLineDash([9, 6]);
    ctx.beginPath();
    zone.forEach((p, i) => { const x = p[0] * canvas.width, y = p[1] * canvas.height; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.closePath(); ctx.stroke();
    ctx.fillStyle = color.replace("rgb", "rgba").replace(")", ", .08)");
    ctx.fill();
    ctx.restore();
  }

  function canvasPoint(ev) {
    const rect = canvas.getBoundingClientRect();
    const x = (ev.clientX - rect.left) / Math.max(1, rect.width);
    const y = (ev.clientY - rect.top) / Math.max(1, rect.height);
    return [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))];
  }

  function hitHandle(pt) {
    const z = editing();
    if (!z) return -1;
    const dx = canvas.width || 1, dy = canvas.height || 1;
    const tx = 14 / dx, ty = 14 / dy;
    return z.findIndex((p) => Math.abs(p[0] - pt[0]) <= tx && Math.abs(p[1] - pt[1]) <= ty);
  }

  function nearestEdge(pt) {
    const z = editing();
    if (!z) return 0;
    let best = 1, idx = z.length - 1;
    z.forEach((p, i) => {
      const q = z[(i + 1) % z.length];
      const ax = p[0], ay = p[1], bx = q[0], by = q[1];
      const vx = bx - ax, vy = by - ay;
      const t = Math.max(0, Math.min(1, ((pt[0] - ax) * vx + (pt[1] - ay) * vy) / (vx * vx + vy * vy || 1)));
      const cx = ax + vx * t, cy = ay + vy * t;
      const d = Math.hypot(pt[0] - cx, pt[1] - cy);
      if (d < best) { best = d; idx = i; }
    });
    return idx;
  }

  function saveZones() {
    clearTimeout(state.zoneTimer);
    state.zoneTimer = setTimeout(async () => {
      try { localStorage.setItem("insight_zones", JSON.stringify(state.zones)); } catch (e) {}
      try {
        await fetch("/api/guard/zone", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ danger: state.zones.danger, duty: state.zones.duty }),
        });
        $("#hint").textContent = "区域已保存（危险区有人=告警；值守区无人=告警）";
      } catch (e) { $("#hint").textContent = "区域保存失败：" + e.message; }
    }, 400);
  }

  async function loadZones() {
    let z = null;
    try { z = JSON.parse(localStorage.getItem("insight_zones") || "null"); } catch (e) {}
    try {
      const r = await fetch("/api/guard/zone", { cache: "no-store" });
      const d = await r.json();
      if (d && (d.danger || d.duty)) z = { danger: d.danger || d.zone, duty: d.duty };
      else if (d && Array.isArray(d.zone)) z = { danger: d.zone, duty: null };
    } catch (e) {}
    if (z && Array.isArray(z.danger) && z.danger.length >= 3) {
      state.zones.danger = z.danger.map((p) => [p[0], p[1]]);
      state.zones.duty = Array.isArray(z.duty) && z.duty.length >= 3 ? z.duty.map((p) => [p[0], p[1]]) : null;
    }
    drawScene();
  }

  function toggleZoneEdit(kind) {
    const next = state.editKind === kind ? null : kind;
    state.editKind = next;
    const dbtn = $("#zoneDangerBtn"), ubtn = $("#zoneDutyBtn");
    if (dbtn) { dbtn.classList.toggle("primary", next === "danger"); dbtn.textContent = next === "danger" ? "完成框选" : "框选危险区"; }
    if (ubtn) { ubtn.classList.toggle("primary", next === "duty"); ubtn.textContent = next === "duty" ? "完成框选" : "框选值守区"; }
    if (next === "duty" && !state.zones.duty) state.zones.duty = DEFAULT_DUTY.map((p) => p.slice());
    $("#hint").textContent = next === "danger"
      ? "危险区：人员进去要报警。拖动控制点调整，点空白加点，右键删点"
      : next === "duty"
        ? "值守区：必须一直有人，没人要报警。拖动控制点调整，点空白加点，右键删点"
        : "区域已锁定";
    updateGuardMeta();
    drawScene();
  }

  function resetZone() {
    state.zones.danger = DEFAULT_ZONE.map((p) => p.slice());
    state.zones.duty = null;
    state.editKind = null;
    saveZones();
    const dbtn = $("#zoneDangerBtn"), ubtn = $("#zoneDutyBtn");
    if (dbtn) { dbtn.classList.remove("primary"); dbtn.textContent = "框选危险区"; }
    if (ubtn) { ubtn.classList.remove("primary"); ubtn.textContent = "框选值守区"; }
    drawScene();
    $("#hint").textContent = "已重置：危险区恢复默认，值守区清空";
  }

  let dragging = false;
  canvas.addEventListener("pointerdown", (ev) => {
    const z = editing();
    if (!z) return;
    const pt = canvasPoint(ev);
    const idx = hitHandle(pt);
    if (idx >= 0) {
      state.dragIdx = idx; dragging = true;
      canvas.setPointerCapture?.(ev.pointerId);
      drawScene();
    } else {
      const e = nearestEdge(pt);
      z.splice(e + 1, 0, [Math.round(pt[0] * 1000) / 1000, Math.round(pt[1] * 1000) / 1000]);
      if (z.length > 12) z.splice(e + 1, 1);
      saveZones(); drawScene();
    }
  });
  canvas.addEventListener("pointermove", (ev) => {
    const z = editing();
    if (!dragging || !z || state.dragIdx < 0) return;
    const pt = canvasPoint(ev);
    z[state.dragIdx] = [Math.round(pt[0] * 1000) / 1000, Math.round(pt[1] * 1000) / 1000];
    drawScene();
  });
  canvas.addEventListener("pointerup", (ev) => {
    if (!dragging) return;
    dragging = false; state.dragIdx = -1;
    canvas.releasePointerCapture?.(ev.pointerId);
    saveZones(); drawScene();
  });
  canvas.addEventListener("contextmenu", (ev) => {
    const z = editing();
    if (!z) return;
    ev.preventDefault();
    const idx = hitHandle(canvasPoint(ev));
    if (idx >= 0 && z.length > 3) {
      z.splice(idx, 1);
      saveZones(); drawScene();
      $("#hint").textContent = "已删除 1 个点";
    }
  });

  function drawScene() {
    const [sw, sh] = sourceSize();
    const maxW = 1280, scale = Math.min(1, maxW / sw);
    canvas.width = Math.round(sw * scale); canvas.height = Math.round(sh * scale);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // 值守模式：画"被检测的那一帧"（冻结），保证框与画面严格同帧
    const useFrozen = state.guard && state.frozenImage && state.frozenImage.complete;
    const src = useFrozen ? state.frozenImage : state.source;
    // 先画封面，视频尚未解码/被浏览器暂停时不会留下黑屏
    if (state.poster) {
      try { ctx.drawImage(state.poster, 0, 0, canvas.width, canvas.height); } catch (e) {}
    }
    if (src) {
      try { ctx.drawImage(src, 0, 0, canvas.width, canvas.height); } catch (e) {}
    }
    // 危险区：常态黄色，区内有人 → 红色；值守区：有人 → 绿色，无人 → 红色
    drawPolygon(state.zones.danger, state.alarm.danger ? "rgb(255,77,79)" : "rgb(255,216,77)", false);
    if (state.zones.duty) {
      drawPolygon(state.zones.duty, state.alarm.duty ? "rgb(255,77,79)" : "rgb(46,230,168)", true);
    }
    if (state.editKind) drawZoneHandles();
    drawBoxes(state.objects);
  }

  function drawBoxes(objects) {
    (objects || []).forEach((o, i) => {
      const b = o.box_norm || o.box; if (!b) return;
      const x1 = b[0] * canvas.width, y1 = b[1] * canvas.height, x2 = b[2] * canvas.width, y2 = b[3] * canvas.height;
      const bad = o.in_danger === true || o.status === "not_worn" || o.inside === true;
      const onDuty = o.in_duty === true;
      const col = bad ? "#ff4d4f" : "#2ee6a8";
      ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillStyle = col; ctx.font = "bold 14px system-ui";
      const tag = bad ? "DANGER " : onDuty ? "ON DUTY " : "";
      ctx.fillText(tag + (o.label || "target") + (o.where ? " · " + o.where : ""), x1 + 3, Math.max(15, y1 - 5));
    });
  }

  function setMode(m) {
    state.mode = m;
    document.querySelectorAll("#modes button").forEach((b) => b.classList.toggle("active", b.dataset.mode === m));
    if ($("#query")) $("#query").value = m === "ppe" ? "找出没戴安全帽的人" : m === "guard" ? "现在塔吊下面有人吗？" : "画面里有哪些人？";
  }
  function quick(q) { if ($("#query")) $("#query").value = q; runAgent(); }

  function addTrace(items) {
    const el = $("#trace"); if (!el) return;
    el.innerHTML = (items || []).map((t) => {
      if (t.type === "thought") return `<div>✦ ${t.text || ""}</div>`;
      if (t.type === "plan") return `<div>计划：${(t.steps || []).map((s) => s.tool).join(" → ")}</div>`;
      if (t.type === "tool") return `<div>${t.tool || ""} · ${t.observation || t.text || ""}</div>`;
      if (t.type === "recovery") return `<div style="color:#ffd84d">! ${t.text || t.observation || ""}</div>`;
      if (t.type === "cache") return `<div>缓存命中</div>`;
      return "";
    }).join("") || "—";
  }

  function setStatus(text, hot = false) {
    const el = $("#status"); if (!el) return;
    el.className = "status" + (hot ? " hot" : "");
    el.innerHTML = `<b>${text}</b><span>${hot ? "本地告警 / 证据已生成" : "本地推理中"}</span>`;
  }

  async function runAgent() {
    if (!state.source || state.busy) return;
    stopGuard();
    state.busy = true;
    const q = ($("#query")?.value || "").trim();
    if (!q) { state.busy = false; return; }
    setStatus("Agent 执行中");
    if ($("#answer")) $("#answer").textContent = "执行中…";
    if ($("#findings")) $("#findings").innerHTML = "";
    if ($("#evidence")) $("#evidence").innerHTML = "";
    drawScene();
    const image = canvas.toDataURL("image/jpeg", 0.82);
    const media = {
      sample_id: state.media?.sampleId, kind: state.media?.kind || "image",
      name: state.media?.title || "upload",
      width: state.media?.width || canvas.width, height: state.media?.height || canvas.height,
      data_url: image,
    };
    try {
      const r = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.session, messages: [{ role: "user", content: q }], media }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || ("HTTP " + r.status));
      if ($("#answer")) $("#answer").textContent = d.reply || "—";
      addTrace(d.trace || []);
      state.objects = (d.objects || []).map((o) => ({ ...o, box_norm: o.box }));
      drawScene();
      const bad = (d.objects || []).some((o) => o.status === "not_worn" || o.status === "intrusion" || o.inside === true) ||
        /危险区内有|未佩戴/.test(d.reply || "");
      setStatus(bad ? "发现隐患" : "任务完成", bad);
      if ($("#findings")) {
        $("#findings").innerHTML = (d.objects || []).map((o) => {
          const bad2 = o.status === "not_worn" || o.status === "intrusion" || o.inside === true;
          return `<div class="finding"><span class="dot ${bad2 ? "" : "ok"}"></span>${o.label || "目标"} ${o.where ? "· " + o.where : ""}${o.status ? " · " + o.status : ""}</div>`;
        }).join("") || "<span class='muted'>没有定位结果</span>";
      }
      if (d.evidence && $("#evidence")) $("#evidence").innerHTML = `<img src="${d.evidence}?t=${Date.now()}"><div class="muted" style="font-size:11px">${d.evidence}</div>`;
      renderHistory(d.history || []);
      loadEvents();
    } catch (e) {
      if ($("#answer")) $("#answer").textContent = "失败：" + e.message;
      setStatus("执行失败", true);
    } finally { state.busy = false; }
  }

  function renderHistory(items) {
    const el = $("#history"); if (!el) return;
    el.innerHTML = (items || []).slice().reverse().map((h) =>
      `<div class="evt"><b>${h.ts || ""}</b> · ${(h.tools || []).join(" → ")}<br><span class="muted">${h.query || ""}</span><br>${h.answer || ""}${h.evidence ? ` · <a href="${h.evidence}" target="_blank">证据</a>` : ""}</div>`
    ).join("") || "<span class='muted'>暂无 Agent 历史</span>";
  }

  function autoMode(q) {
    if (/安全帽|helmet/i.test(q)) return "ppe";
    if (/塔吊|禁区|下面|值守|违规|报警|几次|刚才/.test(q)) return "guard";
    return "visual";
  }
  function normBox(b) {
    if (!b) return [0, 0, 0, 0];
    if (b[2] <= 1 && b[3] <= 1) return b;
    const [w, h] = sourceSize();
    return [b[0] / w, b[1] / h, b[2] / w, b[3] / h];
  }

  /* ------------------------------------------------------------ 值守 */
  async function toggleGuard() {
    if (state.guard) { stopGuard(); return; }
    if (!state.source) { $("#hint").textContent = "请先选择场景"; return; }
    $("#guardBtn").textContent = "停止值守";
    if (state.source.play) {
      state.source.playbackRate = state.rate;
      state.source.play().catch(() => {});
    }
    state.guard = setInterval(guardTick, state.tickMs);
    guardTick();
  }
  function stopGuard() {
    if (state.guard) clearInterval(state.guard);
    state.guard = null; state.frozenImage = null;
    if (state.source && state.source.playbackRate !== undefined) {
      try { state.source.playbackRate = 1; } catch (e) {}
    }
    const b = $("#guardBtn"); if (b) b.textContent = "开始值守";
    updateGuardMeta(); drawScene();
  }
  async function guardTick() {
    if (state.busy || !state.source) return;
    state.busy = true;
    state.tickNo += 1;
    state.frozenImage = null;              // 抓帧前先回到实时画面
    drawScene();
    const t0 = performance.now();
    try {
      const image = canvas.toDataURL("image/jpeg", 0.8);
      // 保存这一帧：后面所有框都画在它上面
      const fi = new Image();
      await new Promise((res) => { fi.onload = res; fi.onerror = res; fi.src = image; });
      state.frozenImage = fi;
      state.frameTs = Date.now();
      const videoT = state.source && typeof state.source.currentTime === "number" ? state.source.currentTime : null;
      const r = await fetch("/api/guard/frame", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image, zones: state.zones, session_id: state.session, ts: Date.now() / 1000,
          sample: state.media?.sampleId || null, video_t: videoT,
        }),
      });
      const d = await r.json(); if (!r.ok) throw new Error(d.error || r.status);
      state.objects = (d.persons || []).map((p) => ({
        box_norm: normBox(p.box), label: "person", in_danger: !!p.in_danger, in_duty: !!p.in_duty,
      }));
      state.alarm = { danger: !!(d.danger && d.danger.alert), duty: !!(d.duty && d.duty.alert) };
      state.detectMs = Math.round(performance.now() - t0);
      drawScene();
      updateGuardMeta();
      if (state.alarm.danger) setStatus("危险区入侵", true);
      else if (state.alarm.duty) setStatus("值守区无人", true);
      else setStatus("值守中", false);
      if ($("#findings")) {
        $("#findings").innerHTML = (d.persons || []).map((p, i) => {
          const where = p.in_danger ? "危险区内" : p.in_duty ? "值守区内" : "画面上";
          const bad = p.in_danger;
          return `<div class="finding"><span class="dot ${bad ? "" : "ok"}"></span>人员 #${i + 1} · ${where}</div>`;
        }).join("") || "<span class='muted'>当前无人员</span>";
      }
      const eps = [d.event, d.duty_event].filter((e) => e && e.id);
      const key = eps.map((e) => `${e.id}:${e.status}`).join(",");
      if (key !== state.episodeKey) {
        state.episodeKey = key;
        if (eps.length) loadEvents();
      }
      if (eps.length) {
        const ep = eps[eps.length - 1];
        state.episode = ep;
        if (ep.evidence && $("#evidence")) {
          $("#evidence").innerHTML = `<img src="${ep.evidence}?t=${Date.now()}"><div class="muted" style="font-size:11px">事件 #${ep.id} · ${ep.kind_label || ""} · ${ep.answer || ""}</div>`;
        }
        if ($("#answer")) $("#answer").textContent = ep.answer || "";
      }
    } catch (e) {
      state.frozenImage = null;
      $("#hint").textContent = "值守帧失败：" + e.message;
      drawScene();
    } finally { state.busy = false; }
  }

  /* ------------------------------------------------------------ 事件台账 */
  function eventAskUrl(x) {
    return "/studio.html?event=" + encodeURIComponent(x.id)
      + "&ask=" + encodeURIComponent("这起事件几点开始、几点结束？期间一共几个人？有没有戴安全帽？");
  }

  function renderEvents(ev) {
    const box = $("#events"); if (!box) return;
    if (!ev.length) { box.innerHTML = "<span class='muted'>暂无事件</span>"; return; }
    const rows = ev.slice(0, 50).map((x) => {
      const open = x.status !== "closed";
      const duty = x.kind === "duty" || x.mission === "duty_zone";
      const people = duty ? "—" : (x.persons_total
        ? `${x.persons_max ?? "-"} <span class="muted">/ 共 ${x.persons_total}</span>`
        : `${x.persons_max ?? "-"}`);
      return `<tr class="row" data-ev="${x.id}">
        <td>${open ? '<span class="tag hot">进行中</span>' : '<span class="tag">已结束</span>'}</td>
        <td><span class="tag ${duty ? "" : "hot"}">${duty ? "值守区" : "危险区"}</span></td>
        <td>${x.enter_hms || "-"}</td>
        <td>${x.exit_hms || "—"}</td>
        <td>${open ? "进行中" : (x.duration_text || "—")}</td>
        <td>${people}</td>
        <td class="muted">${x.session_id ? String(x.session_id).replace("insight_", "").slice(-7) : "—"}</td>
        <td>${x.n_frames ? `<span class="muted">${x.n_frames} 帧</span>` : "—"}</td>
        <td>${x.evidence ? `<a href="${x.evidence}" target="_blank">查看</a>` : "—"}</td>
        <td><a href="${eventAskUrl(x)}" style="color:#7ef0dc">追问 →</a></td>
      </tr>`;
    }).join("");
    box.innerHTML = `<table class="evt">
      <thead><tr><th>状态</th><th>类型</th><th>进入</th><th>离开</th><th>持续</th><th>人数</th><th>会话</th><th>片段</th><th>证据</th><th>对话</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
    box.querySelectorAll("tr.row").forEach((tr) => {
      tr.onclick = (e) => {
        if (e.target.closest("a")) return;
        const x = ev.find((y) => String(y.id) === tr.dataset.ev);
        if (x) location.href = eventAskUrl(x);
      };
    });
  }

  async function loadEvents() {
    try {
      const r = await fetch("/api/guard/events", { cache: "no-store" });
      const d = await r.json();
      const ev = d.events || [];
      state.events = ev;
      state.total = d.total ?? ev.length;
      renderEvents(ev);
      const t = document.getElementById("evtTotal");
      if (t) t.textContent = `共 ${state.total} 起 · 一次进入记一条`;
    } catch (e) {}
  }

  function askLedgerPreset(q) {
    const input = document.getElementById("guardQ");
    if (input) input.value = q;
    askLedger();
  }

  async function askLedger() {
    const input = document.getElementById("guardQ");
    const q = (input?.value || "").trim();
    if (!q) return;
    const box = document.getElementById("ledgerAnswer");
    if (box) { box.style.display = "block"; box.textContent = "查询中…"; }
    try {
      const r = await fetch("/api/guard/ask", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, session_id: state.session }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || ("HTTP " + r.status));
      if (box) box.textContent = d.answer || "—";
      if (d.events) { renderEvents(d.events); if (typeof d.total === "number") { state.total = d.total; const t = document.getElementById("evtTotal"); if (t) t.textContent = `共 ${state.total} 起 · 一次进入记一条`; } }
    } catch (e) {
      if (box) box.textContent = "查询失败：" + e.message;
    }
  }

  function setGuardParam(key, val) {
    if (key === "tickMs") state.tickMs = Math.max(500, parseInt(val, 10) || 2000);
    if (key === "rate") state.rate = parseFloat(val) || 1;
    if (state.source && state.source.playbackRate !== undefined && state.guard) {
      try { state.source.playbackRate = state.rate; } catch (e) {}
    }
    if (state.guard) {                 // 重启定时器，立即生效
      clearInterval(state.guard);
      state.guard = setInterval(guardTick, state.tickMs);
    }
    updateGuardMeta();
  }
  window.setGuardParam = setGuardParam;

  function updateGuardMeta() {
    const el = document.getElementById("guardMeta");
    if (!el) return;
    if (!state.guard) {
      const z = editing();
      el.textContent = z ? `${state.editKind === "duty" ? "值守区" : "危险区"} ${z.length} 个顶点 · 拖动调整，右键删点` : "";
      return;
    }
    const t = state.frameTs ? new Date(state.frameTs).toLocaleTimeString("zh-CN", { hour12: false }) : "—";
    const ms = state.frameTs ? (state.frameTs % 1000).toString().padStart(3, "0") : "000";
    const vt = state.source && typeof state.source.currentTime === "number" ? ` · 视频 ${state.source.currentTime.toFixed(1)}s` : "";
    el.innerHTML = `采样帧 <b>${t}.${ms}</b> · 每 <b>${(state.tickMs / 1000).toFixed(1)}</b> s 一帧（<b>${state.rate}×</b>） · 检测 <b>${state.detectMs}</b> ms · 第 <b>${state.tickNo}</b> 帧${vt}`;
  }

  function clearScene() { state.objects = []; drawScene(); }
  function newSession() {
    stopGuard();
    state.session = "insight_" + Math.random().toString(36).slice(2, 10);
    localStorage.setItem("insight_session_v2", state.session);
    state.objects = []; state.episode = null; state.episodeKey = "";
    if ($("#history")) $("#history").innerHTML = "<span class='muted'>暂无 Agent 历史</span>";
    if ($("#answer")) $("#answer").textContent = "已开启新会话";
    if ($("#findings")) $("#findings").innerHTML = "";
    if ($("#evidence")) $("#evidence").innerHTML = "";
    if ($("#trace")) $("#trace").textContent = "Agent trace 会显示在这里";
    drawScene();
  }

  const guardInput = document.getElementById("guardQ");
  if (guardInput) {
    guardInput.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && !ev.isComposing) { ev.preventDefault(); askLedger(); }
    });
  }

  window.askLedgerPreset = askLedgerPreset;
  window.newSession = newSession;
  window.setMode = setMode; window.runAgent = runAgent; window.quick = quick;
  window.toggleGuard = toggleGuard; window.clearScene = clearScene;
  async function loadStatus() {
    const el = document.getElementById("sysChip");
    if (!el) return;
    try {
      const r = await fetch("/api/status", { cache: "no-store" });
      const d = await r.json();
      const bad = Object.entries(d.services || {}).filter(([, v]) => !v.ok).map(([k]) => k);
      el.textContent = d.ok ? "系统在线 · 全离线" : "服务异常：" + bad.join(" / ");
      el.classList.toggle("bad", !d.ok);
      el.classList.toggle("g", !!d.ok);
      el.title = Object.entries(d.services || {})
        .map(([k, v]) => `${k}: ${v.ok ? (v.ms || 0) + "ms" : (v.error || "down")}`).join("\n");
    } catch (e) { el.textContent = "网关异常"; el.classList.add("bad"); }
  }

  window.askLedger = askLedger; window.loadEvents = loadEvents;
  window.toggleZoneEdit = toggleZoneEdit; window.resetZone = resetZone;
  loadSamples(); loadEvents(); loadZones(); loadStatus(); drawScene();
  setInterval(loadStatus, 15000);

  // 深链接：?zoneedit=1 直接进框选模式；?guard=1 加载后自动开始值守（展会/演示用）
  const params = new URLSearchParams(location.search);
  if (params.get("zoneedit") === "1") setTimeout(() => toggleZoneEdit("danger"), 1200);
  if (params.get("guard") === "1") {
    const tryGuard = () => {
      if (state.source) toggleGuard();
      else setTimeout(tryGuard, 400);
    };
    setTimeout(tryGuard, 1600);
  }
})();
