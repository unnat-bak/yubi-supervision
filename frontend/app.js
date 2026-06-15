const stream = document.getElementById("stream");
const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const statusPill = document.getElementById("status-pill");
const statusLabel = document.getElementById("status-label");
const errorMsg = document.getElementById("error-msg");
const errorText = document.getElementById("error-text");
const btnRetry = document.getElementById("btn-retry");
const toast = document.getElementById("toast");
const idleOverlay = document.getElementById("idle-overlay");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingText = document.getElementById("loading-text");
const statsBar = document.getElementById("stats-bar");
const detectionsPanel = document.getElementById("detections-panel");
const detectionsList = document.getElementById("detections-list");
const detectionsEmpty = document.getElementById("detections-empty");
const objectTotal = document.getElementById("object-total");
const confidenceSlider = document.getElementById("confidence-slider");
const confidenceValue = document.getElementById("confidence-value");
const layerToggles = document.getElementById("layer-toggles");
const sourceInput = document.getElementById("source-input");
const telemetrySource = document.getElementById("telemetry-source");

const statObjects = document.getElementById("stat-objects");
const statPose = document.getElementById("stat-pose");
const statFace = document.getElementById("stat-face");
const statHands = document.getElementById("stat-hands");
const statFps = document.getElementById("stat-fps");
const statLatency = document.getElementById("stat-latency");
const degradedBadge = document.getElementById("degraded-badge");
const btnSnapshot = document.getElementById("btn-snapshot");
const btnRecord = document.getElementById("btn-record");
const alertBanner = document.getElementById("alert-banner");
const analysisPanel = document.getElementById("analysis-panel");
const analysisState = document.getElementById("analysis-state");
const analysisSummary = document.getElementById("analysis-summary");
const analysisList = document.getElementById("analysis-list");
const analysisEmpty = document.getElementById("analysis-empty");
const analysisError = document.getElementById("analysis-error");
const expressionPanel = document.getElementById("expression-panel");
const expressionState = document.getElementById("expression-state");
const expressionNotes = document.getElementById("expression-notes");
const expressionList = document.getElementById("expression-list");
const expressionEmpty = document.getElementById("expression-empty");
const expressionError = document.getElementById("expression-error");
const expressionPulse = document.getElementById("expression-pulse");
const expressionLiveLabel = document.getElementById("expression-live-label");
const expressionSignalCount = document.getElementById("expression-signal-count");
const exprBarBrow = document.getElementById("expr-bar-brow");
const exprBarEye = document.getElementById("expr-bar-eye");
const exprRegionBrowVal = document.getElementById("expr-region-brow-val");
const exprRegionEyeVal = document.getElementById("expr-region-eye-val");
const exprRegionEls = document.querySelectorAll(".expr-region");
const hudOverlay = document.getElementById("hud-overlay");
const pipelineRail = document.getElementById("pipeline-rail");
const telemetryUtc = document.getElementById("telemetry-utc");
const telemetrySession = document.getElementById("telemetry-session");
const telemetryFrame = document.getElementById("telemetry-frame");
const telemetryUptime = document.getElementById("telemetry-uptime");
const feedList = document.getElementById("feed-list");
const feedCount = document.getElementById("feed-count");
const fpsSparkline = document.getElementById("fps-sparkline");
const sessionLogActions = document.getElementById("session-log-actions");
const sessionLogSummary = document.getElementById("session-log-summary");
const btnDownloadLog = document.getElementById("btn-download-log");
const btnDismissLog = document.getElementById("btn-dismiss-log");
const reportOverlay = document.getElementById("report-overlay");
const reportStatus = document.getElementById("report-status");
const reportPassItems = [
  document.getElementById("report-pass-1"),
  document.getElementById("report-pass-2"),
  document.getElementById("report-pass-3"),
];

let pollTimer = null;
let isLive = false;
let configDebounce = null;
let isRecording = false;
let lastAlertTs = 0;
let alertHideTimer = null;
let lastTracks = [];
let lastGrouped = [];
let activeThreshold = 0.35;
let analysisEnabled = false;
let toastTimer = null;
let snapshotBusy = false;
let logDownloadBusy = false;
let clockTimer = null;
let lastPipelineFrame = 0;
let lastPipelineFrameAt = 0;
let lastStreamReloadAt = 0;
let feedEntries = [];
const seenFeedKeys = new Set();
let lastGeminiState = null;
let lastObjectCount = null;
const fpsHistory = [];
const MAX_FPS_HISTORY = 48;

const LOG_TAG_LABELS = {
  sys: "System",
  v3: "YUBI v3.0",
  alert: "Alert",
  expr: "Expression",
  obs: "Observation",
  config: "Config",
  capture: "Capture",
  record: "Record",
};

let activeSession = null;
let completedSession = null;

function escapeMd(text) {
  return String(text ?? "").replace(/\|/g, "\\|").replace(/\r?\n/g, " ").trim();
}

function formatLogTimestamp(iso) {
  const d = new Date(iso);
  return d.toISOString().replace("T", " ").slice(0, 19);
}

function formatDurationMs(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) {
    return `${h}h ${m}m ${s}s`;
  }
  if (m > 0) {
    return `${m}m ${s}s`;
  }
  return `${s}s`;
}

function beginSession(sessionId) {
  completedSession = null;
  hideSessionLogActions();
  activeSession = {
    id: sessionId || "PENDING",
    startedAt: new Date().toISOString(),
    endedAt: null,
    events: [],
    config: {},
    peakObjectCount: 0,
    lastV3Summary: "",
    snapshotCount: 0,
    recorded: false,
    finalStats: null,
    finalTracks: [],
    finalAlerts: [],
    fpsSamples: [],
    degraded: [],
  };
}

function logSessionEvent(tag, message, extra = {}) {
  if (!activeSession) return;
  activeSession.events.push({
    at: new Date().toISOString(),
    tag,
    message,
    ...extra,
  });
}

function finalizeSession(status) {
  if (!activeSession) return;
  activeSession.endedAt = new Date().toISOString();
  if (status) {
    if (status.session_id) {
      activeSession.id = status.session_id;
    }
    activeSession.finalStats = {
      frame_index: status.frame_index ?? 0,
      uptime_sec: status.uptime_sec ?? 0,
      object_count: status.object_count ?? 0,
      pose_count: status.pose_count ?? 0,
      face_count: status.face_count ?? 0,
      hand_count: status.hand_count ?? 0,
      fps: status.fps ?? 0,
      latency_ms: status.latency_ms ?? 0,
    };
    activeSession.config = status.config || {};
    activeSession.sourceLabel = status.source_label || "";
    activeSession.lastV3Summary = status.gemini?.scene_summary || "";
    activeSession.finalTracks = status.tracks || [];
    activeSession.finalAlerts = status.alerts || [];
    activeSession.degraded = status.degraded || [];
    activeSession.recorded = Boolean(status.recording);
  }
  logSessionEvent("sys", "Session terminated");
  completedSession = activeSession;
  activeSession = null;
}

function hideSessionLogActions() {
  sessionLogActions.hidden = true;
}

function showSessionLogActions() {
  if (!completedSession || completedSession.events.length < 2) return;
  const duration = completedSession.endedAt && completedSession.startedAt
    ? formatDurationMs(
        new Date(completedSession.endedAt) - new Date(completedSession.startedAt)
      )
    : "—";
  const events = completedSession.events.length;
  sessionLogSummary.textContent = `${completedSession.id} · ${duration} · ${events} events logged`;
  sessionLogActions.hidden = false;
}

function buildSessionMarkdown(session) {
  const started = formatLogTimestamp(session.startedAt);
  const ended = session.endedAt ? formatLogTimestamp(session.endedAt) : "—";
  const duration =
    session.endedAt && session.startedAt
      ? formatDurationMs(
          new Date(session.endedAt) - new Date(session.startedAt)
        )
      : "—";
  const stats = session.finalStats || {};
  const fpsSamples = session.fpsSamples || [];
  const avgFps =
    fpsSamples.length
      ? (fpsSamples.reduce((a, b) => a + b, 0) / fpsSamples.length).toFixed(1)
      : stats.fps
        ? Number(stats.fps).toFixed(1)
        : "—";

  const config = session.config || {};
  const layers = [
    ["Objects", config.show_objects],
    ["Pose / skeleton", config.show_pose],
    ["Joint tags", config.show_pose_labels],
    ["Face", config.show_face],
    ["Hands", config.show_hands],
    ["Expressions", config.show_expressions],
    ["Segmentation masks", config.show_masks],
    ["Identity", config.show_identity],
    ["YUBI v3.0", config.show_gemini],
  ];

  let md = `# YUBI Supervision — Session Log\n\n`;
  md += `> Exported ${formatLogTimestamp(new Date().toISOString())} UTC\n\n`;
  md += `## Session overview\n\n`;
  md += `| Field | Value |\n| --- | --- |\n`;
  md += `| Session ID | \`${session.id}\` |\n`;
  if (session.sourceLabel) {
    md += `| Input source | ${escapeMd(session.sourceLabel)} |\n`;
  }
  md += `| Started (UTC) | ${started} |\n`;
  md += `| Ended (UTC) | ${ended} |\n`;
  md += `| Duration | ${duration} |\n`;
  md += `| Frames processed | ${stats.frame_index ?? "—"} |\n`;
  md += `| Peak tracked objects | ${session.peakObjectCount} |\n`;
  md += `| Snapshots captured | ${session.snapshotCount} |\n`;
  md += `| Video recorded | ${session.recorded ? "Yes" : "No"} |\n`;
  if (session.degraded?.length) {
    md += `| Degraded modules | ${session.degraded.join(", ")} |\n`;
  }
  md += `\n`;

  md += `## Pipeline configuration\n\n`;
  md += `| Layer | Status |\n| --- | --- |\n`;
  for (const [name, on] of layers) {
    md += `| ${name} | ${on ? "Enabled" : "Disabled"} |\n`;
  }
  const confPct =
    config.confidence != null ? `${Math.round(config.confidence * 100)}%` : "—";
  md += `| Confidence gate | ${confPct} |\n\n`;

  md += `## Performance summary\n\n`;
  md += `| Metric | Value |\n| --- | --- |\n`;
  md += `| Average FPS (sampled) | ${avgFps} |\n`;
  md += `| Final FPS | ${stats.fps ? Number(stats.fps).toFixed(1) : "—"} |\n`;
  md += `| Final latency | ${stats.latency_ms ? `${Math.round(stats.latency_ms)} ms` : "—"} |\n`;
  md += `| Final object count | ${stats.object_count ?? "—"} |\n`;
  md += `| Pose / face / hands | ${stats.pose_count ?? 0} / ${stats.face_count ?? 0} / ${stats.hand_count ?? 0} |\n\n`;

  if (session.lastV3Summary) {
    md += `## YUBI v3.0 — final scene analysis\n\n`;
    md += `${session.lastV3Summary}\n\n`;
  }

  if (session.finalTracks?.length) {
    md += `## Final object registry\n\n`;
    md += `| Label | Confidence | Tracker |\n| --- | --- | --- |\n`;
    for (const track of session.finalTracks) {
      const conf =
        track.confidence != null ? `${Math.round(track.confidence * 100)}%` : "—";
      const tid = track.tracker_id != null ? `#${track.tracker_id}` : "—";
      md += `| ${escapeMd(track.label)} | ${conf} | ${tid} |\n`;
    }
    md += `\n`;
  }

  if (session.finalAlerts?.length) {
    md += `## Watchlist alerts\n\n`;
    md += `| Time | Label |\n| --- | --- |\n`;
    for (const alert of session.finalAlerts) {
      md += `| ${escapeMd(alert.time || "—")} | ${escapeMd(alert.label)} |\n`;
    }
    md += `\n`;
  }

  md += `## Event chronology\n\n`;
  md += `| Time (UTC) | Type | Event |\n| --- | --- | --- |\n`;
  for (const entry of session.events) {
    const type = LOG_TAG_LABELS[entry.tag] || entry.tag;
    let message = escapeMd(entry.message);
    if (entry.summary) {
      message += ` — ${escapeMd(entry.summary)}`;
    }
    md += `| ${formatLogTimestamp(entry.at)} | ${type} | ${message} |\n`;
  }
  md += `\n`;

  md += `---\n\n`;
  md += `*Generated by YUBI Supervision Node · YUBI v3.0*\n`;
  return md;
}

function serializeSession(session) {
  return {
    id: session.id,
    startedAt: session.startedAt,
    endedAt: session.endedAt,
    events: session.events,
    peakObjectCount: session.peakObjectCount,
    snapshotCount: session.snapshotCount,
    recorded: session.recorded,
    finalStats: session.finalStats,
    finalTracks: session.finalTracks,
    finalAlerts: session.finalAlerts,
    lastV3Summary: session.lastV3Summary,
    config: session.config,
    sourceLabel: session.sourceLabel,
    degraded: session.degraded,
    fpsSamples: session.fpsSamples,
  };
}

function setReportPassState(activePass, doneThrough) {
  for (let i = 0; i < reportPassItems.length; i += 1) {
    const el = reportPassItems[i];
    if (!el) continue;
    const passNum = i + 1;
    el.classList.toggle("active", activePass > 0 && passNum === activePass);
    el.classList.toggle("done", passNum <= doneThrough);
  }
}

function showReportOverlay() {
  if (!reportOverlay) return;
  reportOverlay.hidden = false;
  if (reportStatus) reportStatus.textContent = "Compiling session data…";
  setReportPassState(0, 0);
}

function hideReportOverlay() {
  if (reportOverlay) reportOverlay.hidden = true;
}

function saveMarkdownFile(markdown, session) {
  const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
  const stamp = session.endedAt
    ? session.endedAt.replace(/[:.]/g, "-").slice(0, 19)
    : new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `yubi-session-${session.id}-${stamp}.md`;
  a.click();
  URL.revokeObjectURL(a.href);
}

async function downloadSessionLog() {
  if (!completedSession || logDownloadBusy) return;
  logDownloadBusy = true;
  btnDownloadLog.disabled = true;

  const draft = buildSessionMarkdown(completedSession);
  let finalMd = draft;
  let enriched = false;

  showReportOverlay();

  try {
    const res = await fetch("/api/session-report/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        draft_markdown: draft,
        session: serializeSession(completedSession),
      }),
    });
    if (!res.ok || !res.body) {
      throw new Error("Report stream failed");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.message && reportStatus) {
          reportStatus.textContent = event.message;
        }
        if (event.phase === "pass") {
          setReportPassState(event.pass, Math.max(0, event.pass - 1));
        }
        if (event.phase === "done") {
          finalMd = event.markdown || finalMd;
          enriched = Boolean(event.enriched);
          setReportPassState(0, event.passes_completed || 0);
        }
        if (event.phase === "error" && reportStatus) {
          reportStatus.textContent =
            event.message || "YUBI v3.0 pass failed — exporting best draft.";
        }
      }
    }
  } catch {
    if (reportStatus) {
      reportStatus.textContent = "YUBI v3.0 unavailable — exporting raw session log.";
    }
    finalMd = draft;
  } finally {
    hideReportOverlay();
    saveMarkdownFile(finalMd, completedSession);
    showToast(
      enriched ? "YUBI v3.0 session report downloaded" : "Session report downloaded"
    );
    logDownloadBusy = false;
    btnDownloadLog.disabled = false;
  }
}

function formatUptime(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) {
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function startTelemetryClock() {
  if (clockTimer) return;
  const tick = () => {
    telemetryUtc.textContent = new Date().toISOString().slice(11, 19);
  };
  tick();
  clockTimer = setInterval(tick, 1000);
}

function reconnectLiveStream(reason) {
  if (!isLive) return;
  const now = Date.now();
  if (now - lastStreamReloadAt < 5000) return;
  lastStreamReloadAt = now;
  stream.src = `/api/stream?${now}`;
  pushFeed("sys", reason || "Refreshing live feed", new Date().toTimeString().slice(0, 8));
}

function maybeRecoverLiveStream(frameIndex, fps) {
  if (!isLive || frameIndex == null) return;
  const now = Date.now();
  if (frameIndex !== lastPipelineFrame) {
    lastPipelineFrame = frameIndex;
    lastPipelineFrameAt = now;
    return;
  }
  // Only reload MJPEG when the pipeline is truly stalled (not just slow).
  if (now - lastPipelineFrameAt < 6000) return;
  if (fps != null && fps > 2) return;
  reconnectLiveStream("Pipeline stalled — refreshing live feed");
  lastPipelineFrameAt = now;
}

function updateTelemetry(data) {
  if (data.session_id) {
    telemetrySession.textContent = data.session_id;
    if (activeSession && activeSession.id === "PENDING") {
      activeSession.id = data.session_id;
    }
  }
  telemetryFrame.textContent = String(data.frame_index ?? 0);
  telemetryUptime.textContent = formatUptime(data.uptime_sec ?? 0);
  if (telemetrySource) {
    telemetrySource.textContent = (data.source_label || "WEBCAM").toUpperCase();
  }
  maybeRecoverLiveStream(data.frame_index, data.fps);
}

function updatePipelineRail(config, gemini, expressions) {
  if (!config) return;
  const map = {
    objects: config.show_objects,
    pose: config.show_pose,
    face: config.show_face,
    hands: config.show_hands,
    expressions: config.show_expressions && Boolean(expressions?.enabled),
    masks: config.show_masks,
    identity: config.show_identity && gemini?.enabled,
    v3: config.show_gemini && gemini?.enabled,
  };
  for (const el of pipelineRail.querySelectorAll(".pipe-item")) {
    const key = el.dataset.pipe;
    const on = Boolean(map[key]);
    el.classList.toggle("active", on);
    const thinking =
      key === "v3" && gemini?.state === "thinking" ||
      key === "expressions" && expressions?.state === "thinking";
    el.classList.toggle("thinking", thinking && on);
  }
}

function pushFeed(tag, message, time) {
  feedEntries.unshift({ tag, message, time });
  if (feedEntries.length > 40) {
    feedEntries.length = 40;
  }
  feedCount.textContent = String(feedEntries.length);
  renderFeed();
  logSessionEvent(tag, message);
}

function renderFeed() {
  feedList.replaceChildren();
  for (const entry of feedEntries) {
    const li = document.createElement("li");
    li.className = "feed-item";
    const time = document.createElement("span");
    time.className = "feed-time";
    time.textContent = entry.time;
    const tag = document.createElement("span");
    tag.className = `feed-tag ${entry.tag}`;
    tag.textContent = entry.tag;
    const msg = document.createElement("span");
    msg.className = "feed-msg";
    msg.textContent = entry.message;
    li.append(time, tag, msg);
    feedList.appendChild(li);
  }
}

function clearIntelFeed() {
  feedEntries = [];
  seenFeedKeys.clear();
  lastGeminiState = null;
  lastObjectCount = null;
  fpsHistory.length = 0;
  feedCount.textContent = "0";
  renderFeed();
  drawSparkline();
}

function ingestIntelFeed(data) {
  const time = new Date().toTimeString().slice(0, 8);
  const gemini = data.gemini;
  const expressions = data.expressions;

  if (gemini?.state && lastGeminiState !== null && gemini.state !== lastGeminiState) {
    pushFeed("v3", `Pipeline ${gemini.state.toUpperCase()}`, time);
  }
  if (gemini?.state) {
    lastGeminiState = gemini.state;
  }

  if (lastObjectCount !== null && data.object_count !== lastObjectCount) {
    if (lastObjectCount === 0 || data.object_count === 0) {
      const msg =
        data.object_count === 0
          ? "Track field cleared"
          : `Tracks active (${data.object_count})`;
      pushFeed("obs", msg, time);
    }
  }
  lastObjectCount = data.object_count;

  for (const alert of data.alerts || []) {
    const key = `alert-${alert.ts}`;
    if (seenFeedKeys.has(key)) continue;
    seenFeedKeys.add(key);
    pushFeed("alert", `${String(alert.label).toUpperCase()} in frame`, alert.time || time);
  }

  for (const event of expressions?.events || []) {
    const key = `expr-${event.label}-${event.ts}`;
    if (seenFeedKeys.has(key)) continue;
    seenFeedKeys.add(key);
    pushFeed("expr", event.label, time);
  }
}

function drawSparkline() {
  if (!fpsSparkline) return;
  const ctx = fpsSparkline.getContext("2d");
  const w = fpsSparkline.width;
  const h = fpsSparkline.height;
  ctx.clearRect(0, 0, w, h);
  if (fpsHistory.length < 2) return;

  const max = Math.max(...fpsHistory, 1);
  ctx.strokeStyle = "rgba(142, 184, 176, 0.75)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  fpsHistory.forEach((fps, i) => {
    const x = (i / (MAX_FPS_HISTORY - 1)) * (w - 4) + 2;
    const y = h - 4 - (fps / max) * (h - 8);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function showToast(message, durationMs = 2200) {
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.hidden = true;
  }, durationMs);
}

function setLiveControls(enabled) {
  btnStop.disabled = !enabled;
  btnSnapshot.disabled = !enabled || snapshotBusy;
  btnRecord.disabled = !enabled;
}

function applyLiveUI() {
  const becomingLive = !isLive;
  isLive = true;
  if (becomingLive) {
    beginSession(telemetrySession.textContent);
  }
  stream.src = `/api/stream?${Date.now()}`;
  stream.classList.add("live");
  lastPipelineFrame = 0;
  lastPipelineFrameAt = Date.now();
  lastStreamReloadAt = 0;
  btnStart.disabled = true;
  setLiveControls(true);
  statsBar.hidden = false;
  detectionsPanel.hidden = false;
  hudOverlay.hidden = false;
  pipelineRail.hidden = false;
  syncAnalysisPanelVisibility();
  idleOverlay.classList.add("hidden");
  statusPill.classList.remove("idle", "starting", "error");
  statusPill.classList.add("live");
  statusLabel.textContent = "LIVE";
  setError(null);
  if (becomingLive) {
    pushFeed("sys", "Vision pipeline online", new Date().toTimeString().slice(0, 8));
  }
}

function setError(message) {
  if (message) {
    errorText.textContent = message;
    errorMsg.hidden = false;
    btnRetry.hidden = false;
    statusPill.classList.remove("idle", "live");
    statusPill.classList.add("error");
    statusLabel.textContent = "FAULT";
  } else {
    errorMsg.hidden = true;
    errorText.textContent = "";
    btnRetry.hidden = true;
  }
}

function syncLayerChips(config) {
  if (!config || !layerToggles) return;
  for (const chip of layerToggles.querySelectorAll(".layer-chip")) {
    const key = chip.dataset.layer;
    chip.classList.toggle("active", Boolean(config[key]));
  }
}

function getThreshold() {
  return Number(confidenceSlider.value) / 100;
}

function syncThresholdDisplay(confidence) {
  if (confidence == null) return;
  const pct = Math.round(confidence * 100);
  confidenceSlider.value = String(pct);
  confidenceValue.textContent = `${pct}%`;
  activeThreshold = confidence;
}

function filterByThreshold(items) {
  return items.filter((item) => item.confidence >= activeThreshold);
}

function renderDetections(tracks, grouped) {
  if (tracks !== undefined) lastTracks = tracks || [];
  if (grouped !== undefined) lastGrouped = grouped || [];

  const source = lastTracks.length ? lastTracks : lastGrouped;
  const items = filterByThreshold(source);

  detectionsList.replaceChildren();
  objectTotal.textContent = String(items.length);
  detectionsEmpty.hidden = items.length > 0;
  detectionsEmpty.textContent =
    items.length === 0 && source.length > 0
      ? "No objects above threshold"
      : "No objects in frame";

  for (const item of items) {
    const li = document.createElement("li");
    li.className = "detection-item";

    const row = document.createElement("div");
    row.className = "detection-row";

    const label = document.createElement("span");
    label.className = "detection-label";
    label.textContent = item.label;

    const meta = document.createElement("div");
    meta.className = "detection-meta";

    if (item.count > 1) {
      const count = document.createElement("span");
      count.className = "detection-count";
      count.textContent = `×${item.count}`;
      meta.appendChild(count);
    }

    if (item.tracker_id != null) {
      const id = document.createElement("span");
      id.className = "detection-id";
      id.textContent = `#${item.tracker_id}`;
      meta.appendChild(id);
    }

    const confidence = document.createElement("span");
    confidence.className = "detection-confidence";
    confidence.textContent = `${Math.round(item.confidence * 100)}%`;
    meta.appendChild(confidence);

    row.append(label, meta);

    const bar = document.createElement("div");
    bar.className = "confidence-bar";
    const fill = document.createElement("div");
    fill.className = "confidence-fill";
    fill.style.width = `${Math.round(item.confidence * 100)}%`;
    bar.appendChild(fill);

    li.append(row, bar);
    detectionsList.appendChild(li);
  }
}

function updateStats(data) {
  statObjects.textContent = String(data.object_count ?? 0);
  statPose.textContent = String(data.pose_count ?? 0);
  statFace.textContent = String(data.face_count ?? 0);
  statHands.textContent = String(data.hand_count ?? 0);
  statFps.textContent = data.fps ? data.fps.toFixed(1) : "—";
  statLatency.textContent = data.latency_ms ? String(Math.round(data.latency_ms)) : "—";
  if (data.fps) {
    fpsHistory.push(data.fps);
    if (fpsHistory.length > MAX_FPS_HISTORY) fpsHistory.shift();
    drawSparkline();
  }
  if (activeSession) {
    if (data.fps) {
      activeSession.fpsSamples.push(data.fps);
      if (activeSession.fpsSamples.length > 180) {
        activeSession.fpsSamples.shift();
      }
    }
    if ((data.object_count ?? 0) > activeSession.peakObjectCount) {
      activeSession.peakObjectCount = data.object_count;
    }
  }
  updateTelemetry(data);
}

function beep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
    osc.start();
    osc.stop(ctx.currentTime + 0.35);
  } catch {
    /* audio unavailable */
  }
}

function handleAlerts(alerts) {
  if (!alerts?.length) return;
  const latest = alerts[alerts.length - 1];
  if (latest.ts <= lastAlertTs) return;
  lastAlertTs = latest.ts;
  alertBanner.textContent = `${latest.label} detected · ${latest.time}`;
  alertBanner.hidden = false;
  beep();
  clearTimeout(alertHideTimer);
  alertHideTimer = setTimeout(() => {
    alertBanner.hidden = true;
  }, 4000);
}

function setRecordingUI(recording) {
  isRecording = recording;
  btnRecord.classList.toggle("recording", recording);
  for (const node of btnRecord.childNodes) {
    if (node.nodeType === Node.TEXT_NODE) {
      node.textContent = recording ? "Stop" : "Record";
    }
  }
}

function updateDegraded(degraded) {
  if (degraded?.length) {
    degradedBadge.textContent = `${degraded.join(", ")} unavailable`;
    degradedBadge.hidden = false;
  } else {
    degradedBadge.hidden = true;
  }
}

function syncAnalysisPanelVisibility() {
  analysisPanel.hidden = !isLive;
}

function clearPanelContent() {
  analysisState.textContent = "OFF";
  analysisState.className = "panel-badge analysis-badge";
  analysisSummary.textContent = "Semantic scene analysis when vision is live.";
  analysisList.replaceChildren();
  analysisEmpty.hidden = true;
  analysisError.hidden = true;
  expressionState.textContent = "OFF";
  expressionState.className = "panel-badge expression-badge";
  expressionNotes.textContent = "High-precision facial mesh for micro-movement detection.";
  expressionList.replaceChildren();
  expressionEmpty.hidden = false;
  expressionEmpty.textContent = "No micro-signals";
  expressionError.hidden = true;
  if (expressionPulse) {
    expressionPulse.className = "expression-pulse";
  }
  if (expressionLiveLabel) {
    expressionLiveLabel.textContent = "Awaiting face lock";
  }
  if (expressionSignalCount) {
    expressionSignalCount.textContent = "0";
  }
  setExpressionRegionActivity("eyebrow", 0);
  setExpressionRegionActivity("under-eye", 0);
}

function renderAnalysis(analysis) {
  if (!analysis) return;

  analysisEnabled = Boolean(analysis.enabled);
  syncAnalysisPanelVisibility();

  const stateLabels = {
    disabled: "OFF",
    idle: "IDLE",
    thinking: "RUN",
    ready: "LIVE",
    error: "FAULT",
  };

  analysisState.textContent = stateLabels[analysis.state] || analysis.state;
  analysisState.className = "panel-badge analysis-badge";
  if (analysis.state === "thinking") analysisState.classList.add("thinking");
  if (analysis.state === "ready") analysisState.classList.add("ready");
  if (analysis.state === "error") analysisState.classList.add("error");

  if (!analysis.enabled) {
    analysisSummary.textContent = "YUBI v3.0 is not configured on this server.";
    analysisList.replaceChildren();
    analysisEmpty.hidden = true;
    analysisError.hidden = true;
    if (!isLive) analysisPanel.hidden = true;
    return;
  }

  if (analysis.error) {
    analysisError.textContent = analysis.error;
    analysisError.hidden = false;
  } else {
    analysisError.hidden = true;
    analysisError.textContent = "";
  }

  analysisSummary.textContent =
    analysis.scene_summary ||
    (analysis.state === "thinking"
      ? "Analyzing scene…"
      : isLive
        ? "Waiting for first YUBI v3.0 analysis…"
        : "Start vision for live analysis.");

  analysisList.replaceChildren();
  const items = analysis.objects || [];
  analysisEmpty.hidden = items.length > 0 || analysis.state === "thinking";
  analysisEmpty.textContent =
    analysis.state === "thinking" ? "Analyzing…" : "No objects identified yet";

  for (const item of items) {
    const li = document.createElement("li");
    li.className = "analysis-item";
    const label = document.createElement("span");
    label.className = "analysis-item-label";
    label.textContent = item.label;
    const meta = document.createElement("span");
    meta.className = "analysis-item-meta";
    meta.textContent = `${Math.round(item.confidence * 100)}%`;
    li.append(label, meta);
    analysisList.appendChild(li);
  }

  syncAnalysisPanelVisibility();
}

function setExpressionRegionActivity(region, intensity) {
  const pct = Math.round(Math.max(0, Math.min(1, intensity)) * 100);
  const label = pct > 0 ? `${pct}%` : "—";
  if (region === "eyebrow" && exprBarBrow && exprRegionBrowVal) {
    exprBarBrow.style.width = `${pct}%`;
    exprRegionBrowVal.textContent = label;
  } else if (region === "under-eye" && exprBarEye && exprRegionEyeVal) {
    exprBarEye.style.width = `${pct}%`;
    exprRegionEyeVal.textContent = label;
  }
  for (const el of exprRegionEls) {
    const regionLabel = el.querySelector(".expr-region-label")?.textContent;
    const match =
      (region === "eyebrow" && regionLabel === "Brow") ||
      (region === "under-eye" && regionLabel === "Under-eye");
    if (match) {
      el.classList.toggle("active", pct > 12);
    }
  }
}

function renderExpressions(expr) {
  if (!expr) return;
  const videoWrap = document.getElementById("video-wrap");
  const showPanel = isLive && expr.enabled;
  expressionPanel.hidden = !showPanel;
  videoWrap.classList.toggle("expressions-live", showPanel);

  const stateLabels = {
    disabled: "OFF",
    idle: "IDLE",
    thinking: "RUN",
    ready: "LIVE",
    error: "FAULT",
  };

  expressionState.textContent = stateLabels[expr.state] || expr.state;
  expressionState.className = "panel-badge expression-badge";
  if (expr.state === "thinking") expressionState.classList.add("thinking");
  if (expr.state === "ready") expressionState.classList.add("ready");
  if (expr.state === "error") expressionState.classList.add("error");

  if (expr.error) {
    expressionError.textContent = expr.error;
    expressionError.hidden = false;
  } else {
    expressionError.hidden = true;
    expressionError.textContent = "";
  }

  const cues = expr.micro_cues || [];
  const events = expr.events || [];
  const browIntensity = events
    .filter((e) => e.region === "eyebrow")
    .reduce((max, e) => Math.max(max, e.intensity || 0), 0);
  const eyeIntensity = events
    .filter((e) => e.region === "under-eye")
    .reduce((max, e) => Math.max(max, e.intensity || 0), 0);

  setExpressionRegionActivity("eyebrow", browIntensity);
  setExpressionRegionActivity("under-eye", eyeIntensity);

  if (expressionPulse) {
    expressionPulse.className = "expression-pulse";
    if (expr.state === "thinking") {
      expressionPulse.classList.add("thinking");
    } else if (events.length > 0) {
      expressionPulse.classList.add("active");
    }
  }
  if (expressionLiveLabel) {
    if (expr.state === "thinking") {
      expressionLiveLabel.textContent = "Mapping facial structure…";
    } else if (events.length > 0) {
      expressionLiveLabel.textContent = `${events.length} micro-signal${events.length === 1 ? "" : "s"} active`;
    } else if (expr.state === "ready") {
      expressionLiveLabel.textContent = "Tracking — mesh locked";
    } else {
      expressionLiveLabel.textContent = "Awaiting face lock";
    }
  }

  expressionNotes.textContent =
    expr.structure_notes ||
    (cues.length ? cues.join(" · ") : "Monitoring brow and under-eye micro-movements.");

  expressionList.replaceChildren();
  const items = events.length
    ? events
    : cues.map((c) => ({ label: c, intensity: null, region: "" }));

  if (expressionSignalCount) {
    expressionSignalCount.textContent = String(events.length);
  }
  expressionEmpty.hidden = items.length > 0 || expr.state === "thinking";
  expressionEmpty.textContent =
    expr.state === "thinking" ? "Analyzing face structure…" : "No micro-movements detected";

  for (const item of items) {
    const li = document.createElement("li");
    li.className = "expression-item";
    if (item.intensity != null && item.intensity > 0.1) {
      li.classList.add("active");
    }
    const label = document.createElement("span");
    label.className = "expression-item-label";
    label.textContent = item.label;
    const meta = document.createElement("span");
    meta.className = "expression-item-meta";
    meta.textContent =
      item.intensity != null ? `${Math.round(item.intensity * 100)}%` : "cue";
    const bar = document.createElement("div");
    bar.className = "expression-item-bar";
    const fill = document.createElement("div");
    fill.className = "expression-item-fill";
    fill.style.width =
      item.intensity != null ? `${Math.round(item.intensity * 100)}%` : "18%";
    bar.append(fill);
    li.append(label, meta, bar);
    expressionList.appendChild(li);
  }
}

function setIdle() {
  isLive = false;
  snapshotBusy = false;
  stream.removeAttribute("src");
  stream.classList.remove("live");
  btnStart.disabled = false;
  setLiveControls(false);
  setRecordingUI(false);
  alertBanner.hidden = true;
  statusPill.classList.remove("live", "error");
  statusPill.classList.add("idle");
  statusLabel.textContent = "STANDBY";
  idleOverlay.classList.remove("hidden");
  statsBar.hidden = true;
  detectionsPanel.hidden = true;
  hudOverlay.hidden = true;
  pipelineRail.hidden = true;
  expressionPanel.hidden = true;
  analysisPanel.hidden = true;
  document.getElementById("video-wrap").classList.remove("expressions-live");
  clearPanelContent();
  syncAnalysisPanelVisibility();
  loadingOverlay.hidden = true;
  renderDetections([], []);
  updateStats({});
  updateDegraded([]);
  clearIntelFeed();
  setError(null);
  stopPolling();
}

async function pushConfig(partial) {
  try {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(partial),
    });
    if (!res.ok) return null;
    const config = await res.json();
    syncLayerChips(config);
    if (isLive) {
      for (const [key, value] of Object.entries(partial)) {
        if (key.startsWith("show_")) {
          const layer = key.replace("show_", "").replace("_", " ");
          logSessionEvent(
            "config",
            `${layer} ${value ? "enabled" : "disabled"}`
          );
        }
        if (key === "confidence" && value != null) {
          logSessionEvent(
            "config",
            `Confidence gate set to ${Math.round(Number(value) * 100)}%`
          );
        }
      }
    }
    if (partial.show_expressions != null && !isLive) {
      renderExpressions({
        enabled: config.show_expressions,
        state: config.show_expressions ? "idle" : "disabled",
        events: [],
        micro_cues: [],
        structure_notes: config.show_expressions
          ? "Expressions will activate when vision is live."
          : "",
        error: null,
      });
    }
    if (isLive) {
      refreshStatus();
    }
    return config;
  } catch {
    return null;
  }
}

async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    renderAnalysis(data.gemini);
    renderExpressions(data.expressions);

    if (data.state === "idle" && !isLive) {
      analysisPanel.hidden = true;
      expressionPanel.hidden = true;
    }

    if (data.state === "error" && data.error) {
      if (isLive) {
        finalizeSession(data);
        setIdle();
        showSessionLogActions();
      }
      setError(data.error);
      return;
    }

    if (data.state === "live") {
      if (!isLive) {
        applyLiveUI();
        startPolling();
      }
      setError(null);
      updateStats(data);
      updateDegraded(data.degraded);
      updatePipelineRail(data.config, data.gemini, data.expressions);
      ingestIntelFeed(data);
      handleAlerts(data.alerts);
      if (data.recording !== isRecording) setRecordingUI(data.recording);
      if (data.config?.confidence != null) {
        syncThresholdDisplay(data.config.confidence);
      }
      renderDetections(data.tracks, data.objects);
      syncLayerChips(data.config);
      if (activeSession && data.gemini?.scene_summary) {
        const summary = data.gemini.scene_summary;
        if (summary !== activeSession.lastV3Summary) {
          if (activeSession.lastV3Summary) {
            logSessionEvent("v3", "Scene analysis updated", { summary });
          }
          activeSession.lastV3Summary = summary;
        }
      }
    } else if (data.state === "starting") {
      statusPill.classList.remove("idle", "live", "error");
      statusPill.classList.add("starting");
      statusLabel.textContent = "BOOTING";
      if (data.startup_message) {
        loadingText.textContent = data.startup_message;
      }
    }
  } catch {
    /* ignore transient network errors */
  }
}

function startPolling() {
  stopPolling();
  pollTimer = setInterval(refreshStatus, 400);
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function waitUntilLive(timeoutMs = 120000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const res = await fetch("/api/status");
    const data = await res.json();
    renderAnalysis(data.gemini);
    renderExpressions(data.expressions);
    if (data.startup_message) {
      loadingText.textContent = data.startup_message;
    }
    if (data.state === "live") {
      return data;
    }
    if (data.state === "error") {
      throw new Error(data.error || "Failed to start vision");
    }
    await new Promise((resolve) => setTimeout(resolve, 400));
  }
  throw new Error(
    "Startup timed out. Check camera permissions in System Settings and try again."
  );
}

async function startVision() {
  setError(null);
  btnStart.disabled = true;
  loadingOverlay.hidden = false;
  loadingText.textContent = "Starting vision pipeline…";
  idleOverlay.classList.add("hidden");

  const source = sourceInput ? sourceInput.value.trim() : "";
  try {
    localStorage.setItem("yubi-source", source);
  } catch {
    /* storage unavailable */
  }

  try {
    const res = await fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || "Failed to start vision");
    }

    await waitUntilLive();

    applyLiveUI();
    startPolling();
    refreshStatus();
  } catch (err) {
    setIdle();
    setError(err.message);
  } finally {
    loadingOverlay.hidden = true;
  }
}

async function stopVision() {
  if (!isLive) return;
  let finalStatus = null;
  try {
    const statusRes = await fetch("/api/status");
    if (statusRes.ok) {
      finalStatus = await statusRes.json();
    }
    await fetch("/api/stop", { method: "POST" });
  } catch {
    /* still reset UI */
  }
  finalizeSession(finalStatus);
  setIdle();
  showSessionLogActions();
  refreshStatus();
}

async function takeSnapshot() {
  if (!isLive || snapshotBusy) return;
  snapshotBusy = true;
  btnSnapshot.disabled = true;

  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  let saved = 0;

  for (const [path, name] of [
    ["/api/snapshot", `snapshot-${stamp}.png`],
    ["/api/snapshot/json", `snapshot-${stamp}.json`],
  ]) {
    try {
      const res = await fetch(path);
      if (!res.ok) continue;
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = name;
      a.click();
      URL.revokeObjectURL(a.href);
      saved += 1;
    } catch {
      /* skip */
    }
  }

  snapshotBusy = false;
  if (isLive) {
    btnSnapshot.disabled = false;
  }
  if (saved && activeSession) {
    activeSession.snapshotCount += 1;
    logSessionEvent(
      "capture",
      `Snapshot exported (${saved} file${saved > 1 ? "s" : ""})`
    );
  }
  showToast(saved ? "Capture exported" : "Capture failed — retry");
}

async function retryVision() {
  setError(null);
  btnRetry.disabled = true;
  try {
    await fetch("/api/stop", { method: "POST" });
  } catch {
    /* still retry */
  }
  setIdle();
  btnRetry.disabled = false;
  await startVision();
}

async function toggleRecording() {
  if (!isLive) return;
  const action = isRecording ? "stop" : "start";
  try {
    const res = await fetch(`/api/record/${action}`, { method: "POST" });
    if (res.ok) {
      const data = await res.json();
      setRecordingUI(data.recording);
      logSessionEvent(
        "record",
        data.recording ? "Recording started" : "Recording stopped"
      );
    }
  } catch {
    /* keep current state */
  }
}

btnStart.addEventListener("click", startVision);
btnStop.addEventListener("click", stopVision);
btnSnapshot.addEventListener("click", takeSnapshot);
btnRecord.addEventListener("click", toggleRecording);
btnRetry.addEventListener("click", retryVision);
btnDownloadLog.addEventListener("click", downloadSessionLog);
btnDismissLog.addEventListener("click", hideSessionLogActions);

if (layerToggles) {
  layerToggles.addEventListener("click", (e) => {
    const chip = e.target.closest(".layer-chip");
    if (!chip) return;
    const layer = chip.dataset.layer;
    const wasActive = chip.classList.contains("active");
    const next = !wasActive;
    chip.classList.toggle("active", next);
    pushConfig({ [layer]: next }).then((config) => {
      if (!config) {
        chip.classList.toggle("active", wasActive);
      }
    });
  });
}

function applyThreshold() {
  activeThreshold = getThreshold();
  const pct = Math.round(activeThreshold * 100);
  confidenceValue.textContent = `${pct}%`;
  if (isLive) {
    renderDetections();
    statObjects.textContent = String(
      filterByThreshold(lastTracks.length ? lastTracks : lastGrouped).length
    );
    clearTimeout(configDebounce);
    configDebounce = setTimeout(() => {
      pushConfig({ confidence: activeThreshold });
    }, 80);
  }
}

confidenceSlider.addEventListener("input", applyThreshold);
confidenceSlider.addEventListener("change", applyThreshold);

document.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
    return;
  }
  if (e.code === "Space" && !isLive && !btnStart.disabled) {
    e.preventDefault();
    startVision();
    return;
  }
  if (!isLive) return;
  if (e.key === "Escape" || e.key === "q" || e.key === "Q") {
    stopVision();
    return;
  }
  if ((e.key === "s" || e.key === "S") && !btnSnapshot.disabled) {
    e.preventDefault();
    takeSnapshot();
  }
});

async function hydrateFromServer() {
  try {
    const [configRes, statusRes] = await Promise.all([
      fetch("/api/config"),
      fetch("/api/status"),
    ]);
    const config = await configRes.json();
    const status = await statusRes.json();

    syncLayerChips(config);
    if (config.confidence != null) {
      syncThresholdDisplay(config.confidence);
    }
    if (status.session_id) {
      telemetrySession.textContent = status.session_id;
    }
    renderAnalysis(status.gemini);
    renderExpressions(status.expressions);

    if (status.state === "live") {
      applyLiveUI();
      startPolling();
      refreshStatus();
      return;
    }

    if (status.state === "starting") {
      btnStart.disabled = true;
      loadingOverlay.hidden = false;
      idleOverlay.classList.add("hidden");
      statusPill.classList.remove("idle", "live", "error");
      statusPill.classList.add("starting");
      statusLabel.textContent = "BOOTING";
      if (status.startup_message) {
        loadingText.textContent = status.startup_message;
      }
      try {
        await waitUntilLive();
        applyLiveUI();
        startPolling();
        refreshStatus();
      } catch (err) {
        setIdle();
        setError(err.message);
      } finally {
        loadingOverlay.hidden = true;
      }
      return;
    }

    if (status.state === "error" && status.error) {
      setError(status.error);
    }
  } catch {
    /* server may be offline on first paint */
  }
}

function restoreSavedSource() {
  if (!sourceInput) return;
  try {
    const saved = localStorage.getItem("yubi-source");
    if (saved) sourceInput.value = saved;
  } catch {
    /* storage unavailable */
  }
}

restoreSavedSource();
hydrateFromServer();
startTelemetryClock();
