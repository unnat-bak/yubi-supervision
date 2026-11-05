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
const expressionSubjects = document.getElementById("expression-subjects");
const expressionAuCount = document.getElementById("expression-au-count");
const exprValence = document.getElementById("expr-valence");
const exprEmotionBars = document.getElementById("expr-emotion-bars");
const exprTimeline = document.getElementById("expr-timeline");
const exprTimelineSpan = document.getElementById("expr-timeline-span");
const exprAuChips = document.getElementById("expr-au-chips");
const exprRegionEls = document.querySelectorAll(".expr-region[data-region]");
const exprVitalEls = document.querySelectorAll(".expr-vital[data-vital]");
const aiSpend = document.getElementById("ai-spend");
const aiSpendValue = document.getElementById("ai-spend-value");
const aiSpendFill = document.getElementById("ai-spend-fill");
const aiSpendNote = document.getElementById("ai-spend-note");
const insightSection = document.getElementById("insight-section");
const insightState = document.getElementById("insight-state");
const insightSummary = document.getElementById("insight-summary");
const insightList = document.getElementById("insight-list");
const profileOptions = document.getElementById("profile-options");
const profileHint = document.getElementById("profile-hint");
const telemetryMediaItem = document.getElementById("telemetry-media-item");
const telemetryMedia = document.getElementById("telemetry-media");
const telemetryProfile = document.getElementById("telemetry-profile");
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
const btnDownloadRaw = document.getElementById("btn-download-raw");
const btnDownloadEnriched = document.getElementById("btn-download-enriched");
const btnDismissLog = document.getElementById("btn-dismiss-log");
const btnExportAnalysis = document.getElementById("btn-export-analysis");
const reportOverlay = document.getElementById("report-overlay");
const reportStatus = document.getElementById("report-status");
const reportPassList = document.getElementById("report-pass-list");

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
let lastStreamFrameSeq = 0;
let lastStreamFrameAt = 0;
let lastStreamReloadAt = 0;
let feedEntries = [];
const seenFeedKeys = new Set();
const seenEpisodeIds = new Set();
const seenInsightIds = new Set();
let lastGeminiState = null;
let lastObjectCount = null;
let lastAiPausedReason = null;
let lastInsightKey = null;
let lastEpisodeKey = null;
const fpsHistory = [];
const MAX_FPS_HISTORY = 48;

const EMOTION_ROWS = [
  ["happiness", "Joy"],
  ["surprise", "Surprise"],
  ["fear", "Fear"],
  ["anger", "Anger"],
  ["disgust", "Disgust"],
  ["sadness", "Sadness"],
  ["contempt", "Contempt"],
];
const EPISODE_KIND_LABELS = { micro: "MICRO", macro: "MACRO", behavior: "CUE" };
const EXPRESSION_STATE_LABELS = {
  off: "OFF",
  idle: "IDLE",
  no_face: "NO FACE",
  calibrating: "CAL",
  live: "LIVE",
};
const INSIGHT_STATE_LABELS = {
  off: "OFF",
  idle: "IDLE",
  watching: "WATCH",
  thinking: "RUN",
  paused: "PAUSED",
};
const DEFAULT_REPORT_STEPS = ["Analyzing behavioral evidence", "Composing intelligence brief"];

let profileChoices = [];
let selectedProfile = "general";
let reportSteps = DEFAULT_REPORT_STEPS;

const videoWrap = document.getElementById("video-wrap");
const btnFocus = document.getElementById("btn-focus");
const btnClean = document.getElementById("btn-clean");
const leftRail = document.getElementById("left-rail");
const rightRail = document.getElementById("right-rail");
const toggleLeftPanel = document.getElementById("toggle-left-panel");
const toggleRightPanel = document.getElementById("toggle-right-panel");
const sourceCacheHint = document.getElementById("source-cache-hint");

const PIPE_LAYER_MAP = {
  objects: "show_objects",
  pose: "show_pose",
  face: "show_face",
  hands: "show_hands",
  expressions: "show_expressions",
  masks: "show_masks",
  jtags: "show_pose_labels",
  identity: "show_identity",
  v3: "show_gemini",
};

const OVERLAY_LAYER_KEYS = Object.values(PIPE_LAYER_MAP);

let lastLiveConfig = null;
let focusMode = false;
let leftPanelCollapsed = false;
let rightPanelCollapsed = false;
let cleanFeedMode = false;
let savedOverlayConfig = null;
let youtubeCacheDir = "";
let youtubeCacheAnnounced = false;

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

function formatMediaClock(seconds) {
  const total = Math.max(0, Number(seconds) || 0);
  const m = Math.floor(total / 60);
  const s = total - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
}

function formatTokens(count) {
  const n = Number(count) || 0;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
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
    profile: selectedProfile,
    mediaDurationSec: 0,
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
    activeSession.v3Available = Boolean(status.gemini?.enabled);
    activeSession.profile = status.profile || activeSession.profile;
    activeSession.mediaDurationSec = status.media_duration_sec || 0;
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
  if (btnDownloadEnriched) {
    btnDownloadEnriched.disabled = !completedSession.v3Available;
    btnDownloadEnriched.title = completedSession.v3Available
      ? "Run multi-pass YUBI v3.0 enrichment"
      : "YUBI v3.0 is not configured for this session";
  }
  sessionLogActions.hidden = false;
}

function condenseSessionEvents(events) {
  const filtered = [];
  for (const entry of events) {
    if (entry.tag === "v3" && entry.message === "Scene analysis updated") {
      continue;
    }
    filtered.push(entry);
  }

  const condensed = [];
  for (const entry of filtered) {
    const key = `${entry.tag}\0${entry.message}`;
    const prev = condensed[condensed.length - 1];
    if (prev && prev._key === key) {
      prev._count += 1;
      prev._lastAt = entry.at;
      continue;
    }
    condensed.push({
      ...entry,
      _key: key,
      _count: 1,
      _firstAt: entry.at,
      _lastAt: entry.at,
    });
  }

  return condensed.map(({ _key, _count, _firstAt, _lastAt, ...entry }) => {
    if (_count <= 1) {
      return entry;
    }
    const start = formatLogTimestamp(_firstAt);
    const end = formatLogTimestamp(_lastAt);
    const timeSpan =
      _firstAt === _lastAt ? start : `${start}–${end.slice(11)}`;
    return {
      ...entry,
      at: _firstAt,
      message: `${entry.message} (×${_count}, ${timeSpan})`,
    };
  });
}

function buildSessionMarkdown(session, { condensed = false } = {}) {
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
  if (session.profile) {
    md += `| Analysis profile | ${escapeMd(profileLabel(session.profile))} |\n`;
  }
  if (session.mediaDurationSec > 0) {
    md += `| Media length | ${formatDurationMs(session.mediaDurationSec * 1000)} |\n`;
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
  const chronology = condensed
    ? condenseSessionEvents(session.events)
    : session.events;
  for (const entry of chronology) {
    const type = LOG_TAG_LABELS[entry.tag] || entry.tag;
    let message = escapeMd(entry.message);
    if (!condensed && entry.summary) {
      message += ` — ${escapeMd(entry.summary)}`;
    }
    md += `| ${formatLogTimestamp(entry.at)} | ${type} | ${message} |\n`;
  }
  if (condensed && session.events.length > chronology.length) {
    md += `\n*Chronology condensed: ${session.events.length} raw events → ${chronology.length} rows.*\n`;
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
    profile: session.profile,
    mediaDurationSec: session.mediaDurationSec,
  };
}

function renderReportSteps(steps) {
  if (!reportPassList) return;
  const labels = steps?.length ? steps : reportSteps;
  reportPassList.replaceChildren(
    ...labels.map((label) => {
      const li = document.createElement("li");
      li.className = "report-pass-item";
      li.textContent = label;
      return li;
    })
  );
}

function setReportPassState(activePass, doneThrough) {
  if (!reportPassList) return;
  Array.from(reportPassList.children).forEach((el, i) => {
    const passNum = i + 1;
    el.classList.toggle("active", activePass > 0 && passNum === activePass);
    el.classList.toggle("done", passNum <= doneThrough);
  });
}

function showReportOverlay() {
  if (!reportOverlay) return;
  reportOverlay.hidden = false;
  if (reportStatus) reportStatus.textContent = "Compiling session data…";
  renderReportSteps(reportSteps);
  setReportPassState(0, 0);
}

function withAnalysisSection(markdown, analysisMd) {
  if (!analysisMd || markdown.includes("## Behavioral analysis")) return markdown;
  const section = `${analysisMd.trim()}\n\n`;
  const index = markdown.indexOf("## Event chronology");
  if (index >= 0) {
    return markdown.slice(0, index) + section + markdown.slice(index);
  }
  return `${markdown.trimEnd()}\n\n${section}`;
}

async function fetchAnalysisMarkdown() {
  try {
    const res = await fetch("/api/analysis/report.md");
    if (!res.ok) return "";
    return await res.text();
  } catch {
    return "";
  }
}

function downloadBlob(blob, filename) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function hideReportOverlay() {
  if (reportOverlay) reportOverlay.hidden = true;
}

function saveMarkdownFile(markdown, session, { raw = false } = {}) {
  const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
  const stamp = session.endedAt
    ? session.endedAt.replace(/[:.]/g, "-").slice(0, 19)
    : new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const suffix = raw ? "raw" : "report";
  downloadBlob(blob, `yubi-session-${session.id}-${suffix}-${stamp}.md`);
}

function setSessionLogButtonsBusy(busy) {
  logDownloadBusy = busy;
  if (btnDownloadRaw) btnDownloadRaw.disabled = busy;
  if (btnExportAnalysis) btnExportAnalysis.disabled = busy;
  if (btnDownloadEnriched) {
    btnDownloadEnriched.disabled =
      busy || !completedSession?.v3Available;
  }
}

async function downloadRawSessionLog() {
  if (!completedSession || logDownloadBusy) return;
  setSessionLogButtonsBusy(true);
  try {
    const draft = buildSessionMarkdown(completedSession, { condensed: true });
    const markdown = withAnalysisSection(draft, await fetchAnalysisMarkdown());
    saveMarkdownFile(markdown, completedSession, { raw: true });
    showToast("Raw session log downloaded");
  } finally {
    setSessionLogButtonsBusy(false);
  }
}

async function exportAnalysisTimeline() {
  if (logDownloadBusy) return;
  setSessionLogButtonsBusy(true);
  try {
    const res = await fetch("/api/analysis/export");
    if (!res.ok) throw new Error("Export failed");
    const blob = await res.blob();
    const id = completedSession?.id || telemetrySession.textContent || "session";
    downloadBlob(blob, `yubi-analysis-${id}.json`);
    showToast("Episode timeline exported");
  } catch {
    showToast("Timeline export failed — retry");
  } finally {
    setSessionLogButtonsBusy(false);
  }
}

async function downloadEnrichedSessionLog() {
  if (!completedSession || logDownloadBusy || !completedSession.v3Available) return;
  setSessionLogButtonsBusy(true);

  const draft = buildSessionMarkdown(completedSession, { condensed: true });
  let finalMd = draft;
  let enriched = false;
  let failed = false;

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
        if (event.phase === "compile" && event.steps?.length) {
          renderReportSteps(event.steps);
        }
        if (event.phase === "pass") {
          setReportPassState(event.pass, Math.max(0, event.pass - 1));
        }
        if (event.phase === "done") {
          finalMd = event.markdown || finalMd;
          enriched = Boolean(event.enriched);
          setReportPassState(0, event.passes_completed || 0);
        }
        if (event.phase === "error") {
          failed = true;
          if (reportStatus) {
            reportStatus.textContent =
              event.message || "YUBI v3.0 pass failed — exporting best draft.";
          }
        }
      }
    }
  } catch {
    failed = true;
    if (reportStatus) {
      reportStatus.textContent = "YUBI v3.0 unavailable — exporting raw session log.";
    }
    finalMd = withAnalysisSection(draft, await fetchAnalysisMarkdown());
  } finally {
    hideReportOverlay();
    saveMarkdownFile(finalMd, completedSession);
    let toast = "Session report downloaded";
    if (enriched) toast = "YUBI v3.0 session report downloaded";
    else if (failed) toast = "YUBI v3.0 unavailable — downloaded measured report";
    showToast(toast);
    setSessionLogButtonsBusy(false);
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

function maybeRecoverLiveStream(data) {
  if (!isLive) return;
  const now = Date.now();
  const remoteLive =
    data.source_is_live &&
    (data.source_kind === "youtube" || data.source_kind === "stream");

  if (remoteLive && data.stream_frame_seq != null) {
    if (data.stream_frame_seq !== lastStreamFrameSeq) {
      lastStreamFrameSeq = data.stream_frame_seq;
      lastStreamFrameAt = now;
      return;
    }
    // Remote live MJPEG is decoupled from inference — use stream seq, not frame_index.
    if (now - lastStreamFrameAt < 8000) return;
    const streamFps = data.stream_fps ?? 0;
    if (streamFps > 2) return;
    reconnectLiveStream("Live stream stalled — refreshing feed");
    lastStreamFrameAt = now;
    return;
  }

  const frameIndex = data.frame_index;
  if (frameIndex == null) return;
  if (frameIndex !== lastPipelineFrame) {
    lastPipelineFrame = frameIndex;
    lastPipelineFrameAt = now;
    return;
  }
  if (now - lastPipelineFrameAt < 6000) return;
  if (data.fps != null && data.fps > 2) return;
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
  if (telemetryProfile) {
    telemetryProfile.textContent = profileLabel(data.profile || selectedProfile).toUpperCase();
  }
  const duration = data.media_duration_sec || 0;
  if (telemetryMediaItem) {
    telemetryMediaItem.hidden = !(duration > 0);
    if (duration > 0 && telemetryMedia) {
      const speed = data.processing_speed ? ` · ${data.processing_speed.toFixed(1)}×` : "";
      telemetryMedia.textContent =
        `${formatUptime(data.media_position_sec || 0)} / ${formatUptime(duration)}${speed}`;
    }
  }
  maybeRecoverLiveStream(data);
}

function updatePipelineRail(config, gemini, expressions, insights) {
  if (!config) return;
  const map = {
    objects: config.show_objects,
    pose: config.show_pose,
    face: config.show_face,
    hands: config.show_hands,
    expressions: config.show_expressions && Boolean(expressions?.enabled),
    masks: config.show_masks,
    jtags: config.show_pose_labels,
    identity: config.show_identity && gemini?.enabled,
    v3: config.show_gemini && gemini?.enabled,
  };
  for (const el of pipelineRail.querySelectorAll(".pipe-item")) {
    const key = el.dataset.pipe;
    const on = Boolean(map[key]);
    el.classList.toggle("active", on);
    const thinking =
      key === "v3" && (gemini?.state === "thinking" || insights?.state === "thinking") ||
      key === "expressions" && expressions?.state === "calibrating";
    el.classList.toggle("thinking", thinking && on);
  }
}

function pushFeed(tag, message, time, extra = {}) {
  feedEntries.unshift({ tag, message, time });
  if (feedEntries.length > 40) {
    feedEntries.length = 40;
  }
  feedCount.textContent = String(feedEntries.length);
  renderFeed();
  logSessionEvent(tag, message, extra);
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
  seenEpisodeIds.clear();
  seenInsightIds.clear();
  lastGeminiState = null;
  lastObjectCount = null;
  lastAiPausedReason = null;
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

  for (const episode of [...(expressions?.episodes || [])].reverse()) {
    if (seenEpisodeIds.has(episode.id)) continue;
    seenEpisodeIds.add(episode.id);
    if (!episode.notable) continue;
    const kind = EPISODE_KIND_LABELS[episode.kind] || episode.kind.toUpperCase();
    const aus = episode.aus?.length ? ` (${episode.aus.join("+")})` : "";
    pushFeed("expr", `${kind} ${episode.label}${aus} @ ${formatMediaClock(episode.at)}`, time, {
      summary: [episode.au_label, ...(episode.cue_labels || [])].filter(Boolean).join(" · "),
    });
  }

  for (const insight of [...(data.insights?.insights || [])].reverse()) {
    if (seenInsightIds.has(insight.id)) continue;
    seenInsightIds.add(insight.id);
    pushFeed("v3", `Insight · ${insight.title} @ ${insight.at_clock}`, time, {
      summary: insight.observation,
    });
  }

  const pausedReason = data.ai_usage?.paused ? data.ai_usage.paused_reason || "paused" : null;
  if (pausedReason !== lastAiPausedReason) {
    if (pausedReason) {
      pushFeed("alert", pausedReason, time);
    } else if (lastAiPausedReason) {
      pushFeed("v3", "YUBI v3.0 resumed", time);
    }
    lastAiPausedReason = pausedReason;
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
  if (btnFocus) btnFocus.disabled = !enabled;
  if (btnClean) btnClean.disabled = !enabled;
}

function loadPanelCollapsePrefs() {
  try {
    leftPanelCollapsed = localStorage.getItem("yubi-panel-left") === "1";
    rightPanelCollapsed = localStorage.getItem("yubi-panel-right") === "1";
  } catch {
    leftPanelCollapsed = false;
    rightPanelCollapsed = false;
  }
}

function savePanelCollapsePrefs() {
  try {
    localStorage.setItem("yubi-panel-left", leftPanelCollapsed ? "1" : "0");
    localStorage.setItem("yubi-panel-right", rightPanelCollapsed ? "1" : "0");
  } catch {
    /* storage unavailable */
  }
}

function applyPanelCollapse() {
  if (leftRail) {
    leftRail.classList.toggle("collapsed", leftPanelCollapsed);
  }
  if (rightRail) {
    rightRail.classList.toggle("collapsed", rightPanelCollapsed);
  }
  if (toggleLeftPanel) {
    const expanded = !leftPanelCollapsed;
    toggleLeftPanel.setAttribute("aria-expanded", String(expanded));
    toggleLeftPanel.title = expanded
      ? "Collapse object registry"
      : "Expand object registry";
  }
  if (toggleRightPanel) {
    const expanded = !rightPanelCollapsed;
    toggleRightPanel.setAttribute("aria-expanded", String(expanded));
    toggleRightPanel.title = expanded
      ? "Collapse intelligence panel"
      : "Expand intelligence panel";
  }
}

function toggleLeftPanelCollapse() {
  if (!isLive || focusMode) return;
  leftPanelCollapsed = !leftPanelCollapsed;
  savePanelCollapsePrefs();
  applyPanelCollapse();
}

function toggleRightPanelCollapse() {
  if (!isLive || focusMode) return;
  rightPanelCollapsed = !rightPanelCollapsed;
  savePanelCollapsePrefs();
  applyPanelCollapse();
}

function applyFocusMode() {
  if (videoWrap) {
    videoWrap.classList.toggle("cinema-mode", focusMode);
  }
  if (btnFocus) {
    btnFocus.classList.toggle("active", focusMode);
    btnFocus.textContent = focusMode ? "Panels" : "Focus";
  }
  if (!focusMode) {
    applyPanelCollapse();
  }
}

function toggleFocusMode() {
  if (!isLive) return;
  focusMode = !focusMode;
  applyFocusMode();
  showToast(focusMode ? "Side panels hidden — use top rail to toggle layers" : "Side panels restored");
}

function overlayConfigFrom(config) {
  const partial = {};
  for (const key of OVERLAY_LAYER_KEYS) {
    partial[key] = Boolean(config?.[key]);
  }
  return partial;
}

function allOverlaysOffPartial() {
  const partial = {};
  for (const key of OVERLAY_LAYER_KEYS) {
    partial[key] = false;
  }
  return partial;
}

async function toggleLayer(layerKey, next) {
  const partial = { [layerKey]: next };
  const chip = layerToggles?.querySelector(`[data-layer="${layerKey}"]`);
  const wasActive = chip?.classList.contains("active");
  if (chip) chip.classList.toggle("active", next);
  const config = await pushConfig(partial);
  if (!config) {
    if (chip && wasActive != null) chip.classList.toggle("active", wasActive);
    return null;
  }
  lastLiveConfig = config;
  syncLayerChips(config);
  if (cleanFeedMode && Object.values(overlayConfigFrom(config)).some(Boolean)) {
    cleanFeedMode = false;
    savedOverlayConfig = null;
    if (btnClean) btnClean.textContent = "Clean feed";
  }
  return config;
}

async function toggleCleanFeed() {
  if (!isLive) return;
  if (cleanFeedMode && savedOverlayConfig) {
    const config = await pushConfig(savedOverlayConfig);
    if (!config) return;
    lastLiveConfig = config;
    syncLayerChips(config);
    cleanFeedMode = false;
    savedOverlayConfig = null;
    if (btnClean) {
      btnClean.classList.remove("active");
      btnClean.textContent = "Clean feed";
    }
    showToast("Vision overlays restored");
    return;
  }
  if (!lastLiveConfig) return;
  savedOverlayConfig = overlayConfigFrom(lastLiveConfig);
  const config = await pushConfig(allOverlaysOffPartial());
  if (!config) {
    savedOverlayConfig = null;
    return;
  }
  lastLiveConfig = config;
  syncLayerChips(config);
  cleanFeedMode = true;
  if (btnClean) {
    btnClean.classList.add("active");
    btnClean.textContent = "Overlays";
  }
  showToast("All vision overlays hidden");
}

function resetViewModes() {
  focusMode = false;
  cleanFeedMode = false;
  savedOverlayConfig = null;
  applyFocusMode();
  if (btnClean) {
    btnClean.classList.remove("active");
    btnClean.textContent = "Clean feed";
  }
}

function updateSourceCacheHint() {
  if (!sourceCacheHint || !youtubeCacheDir) return;
  sourceCacheHint.textContent = `YouTube clips cache: ${youtubeCacheDir} (removed on Terminate; stale files pruned after 7 days)`;
  sourceCacheHint.hidden = false;
}

function maybeAnnounceYoutubeCache(sourceKind, sourceLabel) {
  if (sourceKind !== "youtube" || !youtubeCacheDir || youtubeCacheAnnounced) return;
  if (sourceLabel?.includes("LIVE")) return;
  youtubeCacheAnnounced = true;
  showToast(`YouTube cache: ${youtubeCacheDir}`, 4500);
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
  lastStreamFrameSeq = 0;
  lastStreamFrameAt = Date.now();
  lastStreamReloadAt = 0;
  btnStart.disabled = true;
  setLiveControls(true);
  statsBar.hidden = false;
  if (leftRail) leftRail.hidden = false;
  if (rightRail) rightRail.hidden = false;
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
  if (btnFocus) btnFocus.disabled = false;
  if (btnClean) btnClean.disabled = false;
  applyPanelCollapse();
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
  const label = document.getElementById("btn-record-label");
  if (label) label.textContent = recording ? "Stop" : "Record";
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
  lastInsightKey = null;
  lastEpisodeKey = null;
  insightList?.replaceChildren();
  renderExpressions({ enabled: false, state: "off" });
  renderInsights(null);
  renderAiUsage(null, false);
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
    paused: "PAUSED",
    error: "FAULT",
  };

  analysisState.textContent = stateLabels[analysis.state] || analysis.state;
  analysisState.className = "panel-badge analysis-badge";
  if (analysis.state === "thinking") analysisState.classList.add("thinking");
  if (analysis.state === "ready") analysisState.classList.add("ready");
  if (analysis.state === "error" || analysis.state === "paused") {
    analysisState.classList.add("error");
  }

  if (!analysis.enabled) {
    analysisSummary.textContent = "YUBI v3.0 is not configured on this server.";
    analysisList.replaceChildren();
    analysisEmpty.hidden = true;
    analysisError.hidden = true;
    if (!isLive) analysisPanel.hidden = true;
    return;
  }

  if (analysis.error && analysis.state !== "paused") {
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
      : analysis.state === "paused"
        ? "Scene analysis paused."
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

function pct(value) {
  return Math.round(Math.max(0, Math.min(1, Number(value) || 0)) * 100);
}

function makeEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text != null) el.textContent = text;
  return el;
}

function ensureEmotionRows() {
  if (!exprEmotionBars || exprEmotionBars.childElementCount) return;
  for (const [key, label] of EMOTION_ROWS) {
    const row = makeEl("div", "expr-emotion-row");
    row.dataset.emotion = key;
    const bar = makeEl("div", "expr-region-bar");
    bar.append(makeEl("div", "expr-region-fill"));
    row.append(makeEl("span", "expr-emotion-label", label), bar, makeEl("span", "expr-region-val", "—"));
    exprEmotionBars.append(row);
  }
}

function renderEmotionBars(subject) {
  ensureEmotionRows();
  const emotions = subject?.emotions || {};
  const dominant = subject?.dominant?.emotion;
  for (const row of exprEmotionBars?.children || []) {
    const level = pct(emotions[row.dataset.emotion]);
    row.querySelector(".expr-region-fill").style.width = `${level}%`;
    row.querySelector(".expr-region-val").textContent = level > 0 ? `${level}%` : "—";
    row.classList.toggle("dominant", row.dataset.emotion === dominant && level >= 25);
  }
  if (exprValence) {
    if (subject && subject.valence != null) {
      const v = Number(subject.valence);
      exprValence.textContent =
        `V ${v >= 0 ? "+" : ""}${v.toFixed(2)} · A ${Number(subject.activation || 0).toFixed(2)}`;
    } else {
      exprValence.textContent = "—";
    }
  }
}

function renderRegions(regions) {
  for (const el of exprRegionEls) {
    const level = pct(regions?.[el.dataset.region]);
    el.querySelector(".expr-region-fill").style.width = `${level}%`;
    el.querySelector(".expr-region-val").textContent = level > 0 ? `${level}%` : "—";
    el.classList.toggle("active", level > 12);
  }
}

function setVital(key, value, { alert = false, title = "" } = {}) {
  for (const el of exprVitalEls) {
    if (el.dataset.vital !== key) continue;
    el.querySelector(".expr-vital-v").textContent = value;
    el.classList.toggle("alert", alert);
    if (title) el.title = title;
  }
}

function renderVitals(subject) {
  if (!subject) {
    for (const key of ["blink", "gaze", "head", "pulse", "speech", "touch"]) setVital(key, "—");
    return;
  }
  const rate = Number(subject.blink_rate) || 0;
  const baseline = Number(subject.blink_baseline) || 0;
  setVital("blink", `${Math.round(rate)}/min`, {
    alert: baseline > 0 && rate > baseline * 1.5 && rate >= 20,
    title: `Session average ${Math.round(baseline)}/min · ${subject.blink_total ?? 0} blinks`,
  });
  setVital("gaze", subject.gaze_away ? "AWAY" : `ON · ${Math.round(subject.gaze_away_pct || 0)}%`, {
    alert: Boolean(subject.gaze_away),
    title: `${subject.gaze_away_pct ?? 0}% of observed time looking away`,
  });
  const head = subject.head;
  setVital("head", head ? `${Math.round(head.pitch)}° / ${Math.round(head.yaw)}°` : "—", {
    alert: Boolean(head && head.speed > 60),
    title: "Pitch / yaw (degrees)",
  });
  const pulse = subject.pulse;
  setVital("pulse", pulse ? `${Math.round(pulse.bpm)} bpm${pulse.confidence < 0.5 ? "?" : ""}` : "—");
  setVital("speech", subject.speaking ? "TALKING" : "QUIET");
  setVital("touch", subject.self_touch ? "FACE" : "—", { alert: Boolean(subject.self_touch) });
}

function renderAuChips(aus) {
  if (!exprAuChips) return;
  exprAuChips.replaceChildren(
    ...aus.map((au) => {
      const level = pct(au.level);
      const chip = makeEl("span", "expr-au-chip", `${au.code} ${level}%`);
      chip.title = au.name;
      chip.style.setProperty("--level", `${level}%`);
      return chip;
    })
  );
  if (expressionAuCount) expressionAuCount.textContent = String(aus.length);
}

function renderEpisodes(episodes, counts) {
  const total = (counts?.micro || 0) + (counts?.macro || 0) + (counts?.behavior || 0);
  if (expressionSignalCount) {
    expressionSignalCount.textContent = total ? `${counts.micro || 0} µ · ${total}` : "0";
    expressionSignalCount.title =
      `${counts?.micro || 0} micro · ${counts?.macro || 0} macro · ${counts?.behavior || 0} behavioral`;
  }
  expressionEmpty.hidden = episodes.length > 0;
  const key = episodes.map((e) => e.id).join(",");
  if (key === lastEpisodeKey) return;
  lastEpisodeKey = key;
  expressionList.replaceChildren(
    ...episodes.map((episode) => {
      const li = makeEl("li", `expression-item episode-${episode.kind}`);
      if (episode.notable) li.classList.add("active");
      const head = makeEl("div", "episode-head");
      head.append(
        makeEl("span", "episode-kind", EPISODE_KIND_LABELS[episode.kind] || episode.kind),
        makeEl("span", "expression-item-label", episode.label),
        makeEl("span", "expression-item-meta", formatMediaClock(episode.at))
      );
      const side = episode.side === "bilateral" ? "bilateral" : episode.side ? `side ${episode.side}` : "";
      const details = [
        episode.aus?.length ? episode.aus.join("+") : "",
        episode.duration_ms ? `${episode.duration_ms} ms` : "",
        side,
        episode.speaking ? "speaking" : "",
        ...(episode.cue_labels || []).filter((cue) => !episode.label.startsWith(cue)),
      ].filter(Boolean);
      const meta = makeEl("span", "episode-meta", details.join(" · "));
      meta.title = episode.au_label || "";
      li.append(head, meta);
      return li;
    })
  );
}

function drawExpressionTimeline(points, episodes, subjectId) {
  if (!exprTimeline) return;
  const dpr = window.devicePixelRatio || 1;
  const w = exprTimeline.clientWidth || 384;
  const h = exprTimeline.clientHeight || 64;
  if (exprTimeline.width !== Math.round(w * dpr) || exprTimeline.height !== Math.round(h * dpr)) {
    exprTimeline.width = Math.round(w * dpr);
    exprTimeline.height = Math.round(h * dpr);
  }
  const ctx = exprTimeline.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const mid = h / 2;
  ctx.strokeStyle = "rgba(232, 234, 237, 0.08)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(w, mid);
  ctx.stroke();
  if (points.length < 2) {
    if (exprTimelineSpan) exprTimelineSpan.textContent = "—";
    return;
  }
  const end = points[points.length - 1].t;
  const start = Math.min(points[0].t, end - 10);
  const span = Math.max(end - start, 1);
  const x = (t) => ((t - start) / span) * (w - 2) + 1;
  if (exprTimelineSpan) exprTimelineSpan.textContent = `last ${Math.round(span)}s`;

  ctx.fillStyle = "rgba(142, 184, 176, 0.12)";
  ctx.beginPath();
  ctx.moveTo(x(points[0].t), h);
  for (const p of points) ctx.lineTo(x(p.t), h - Math.max(0, Math.min(1, p.a)) * (h - 4));
  ctx.lineTo(x(end), h);
  ctx.closePath();
  ctx.fill();

  for (const episode of episodes) {
    if (subjectId && episode.subject !== subjectId) continue;
    if (episode.at < start || episode.at > end) continue;
    const ex = x(episode.at);
    if (episode.kind === "micro") {
      ctx.strokeStyle = "rgba(201, 168, 106, 0.9)";
      ctx.lineWidth = 1.5;
    } else if (episode.kind === "macro") {
      ctx.strokeStyle = "rgba(232, 234, 237, 0.45)";
      ctx.lineWidth = 1;
    } else {
      ctx.strokeStyle = "rgba(142, 184, 176, 0.35)";
      ctx.lineWidth = 1;
    }
    ctx.beginPath();
    ctx.moveTo(ex, episode.kind === "behavior" ? h - 8 : 2);
    ctx.lineTo(ex, h);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(142, 184, 176, 0.9)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  points.forEach((p, i) => {
    const y = mid - Math.max(-1, Math.min(1, p.v)) * (mid - 3);
    if (i === 0) ctx.moveTo(x(p.t), y);
    else ctx.lineTo(x(p.t), y);
  });
  ctx.stroke();
}

function expressionLiveText(expr, subject) {
  if (!expr.enabled) return "Expression tracking off";
  if (!subject) return expr.state === "no_face" ? "Awaiting face lock" : "Expression tracking idle";
  if (!subject.calibrated) {
    return `Calibrating ${subject.id} baseline — keep a relaxed face (${subject.calibration_pct}%)`;
  }
  const dominant = subject.dominant || {};
  const speech = subject.speaking ? " · speaking" : "";
  if (!dominant.emotion || dominant.emotion === "neutral" || dominant.score < 0.25) {
    return `Neutral — at baseline${speech}`;
  }
  return `${dominant.label} ${pct(dominant.score)}%${speech}`;
}

function renderExpressions(expr) {
  if (!expr) return;
  const showPanel = isLive && Boolean(expr.enabled);
  expressionPanel.hidden = !showPanel;
  videoWrap.classList.toggle("expressions-live", showPanel);

  const subjects = expr.subjects || [];
  const subject = subjects[0] || null;
  let badge = EXPRESSION_STATE_LABELS[expr.state] || String(expr.state || "off").toUpperCase();
  if (expr.state === "calibrating" && subject) badge = `CAL ${subject.calibration_pct}%`;
  expressionState.textContent = badge;
  expressionState.className = "panel-badge expression-badge";
  if (expr.state === "live") expressionState.classList.add("ready");
  if (expr.state === "calibrating") expressionState.classList.add("thinking");

  if (expr.error) {
    expressionError.textContent = expr.error;
    expressionError.hidden = false;
  } else {
    expressionError.hidden = true;
    expressionError.textContent = "";
  }

  if (expressionPulse) {
    expressionPulse.className = "expression-pulse";
    if (expr.state === "calibrating") {
      expressionPulse.classList.add("thinking");
    } else if (subject?.aus?.some((au) => au.level >= 0.3)) {
      expressionPulse.classList.add("active");
    }
  }
  if (expressionLiveLabel) expressionLiveLabel.textContent = expressionLiveText(expr, subject);
  if (expressionSubjects) {
    expressionSubjects.textContent = subjects.length > 1
      ? subjects.map((s) => s.id).join(" · ")
      : subject ? `SUBJ ${subject.id}` : "";
  }

  renderEmotionBars(subject);
  drawExpressionTimeline(expr.timeline || [], expr.episodes || [], subject?.id);
  renderRegions(subject?.regions);
  renderVitals(subject);
  renderAuChips(subject?.aus || []);
  if (expressionNotes) {
    expressionNotes.hidden = Boolean(subject?.aus?.length);
  }
  renderEpisodes(expr.episodes || [], expr.counts || {});
}

function renderInsights(insights) {
  if (!insightSection) return;
  const items = insights?.insights || [];
  const visible = isLive && Boolean(insights?.enabled) &&
    (insights.state !== "idle" || items.length > 0);
  insightSection.hidden = !visible;
  if (!visible) return;

  insightState.textContent = INSIGHT_STATE_LABELS[insights.state] || insights.state.toUpperCase();
  insightState.className = "panel-badge analysis-badge";
  if (insights.state === "thinking") insightState.classList.add("thinking");
  if (insights.state === "paused") insightState.classList.add("error");

  let summary = insights.running_summary || "";
  if (!summary) {
    if (insights.state === "paused") {
      summary = "Interpretation paused — measured analysis continues and is kept for the report.";
    } else if (insights.pending) {
      summary = `${insights.pending} notable moment${insights.pending === 1 ? "" : "s"} queued for interpretation.`;
    } else {
      summary = "Watching for notable moments — interpretation runs only on new evidence.";
    }
  }
  insightSummary.textContent = summary;

  const key = items.map((i) => i.id).join(",");
  if (key === lastInsightKey) return;
  lastInsightKey = key;
  insightList.replaceChildren(
    ...items.map((insight) => {
      const li = makeEl("li", `insight-item insight-${insight.category}`);
      const head = makeEl("div", "insight-head");
      head.append(
        makeEl("span", "insight-time", insight.at_clock),
        makeEl("span", "insight-title", insight.title),
        makeEl("span", "insight-confidence", `${pct(insight.confidence)}%`)
      );
      li.append(head, makeEl("p", "insight-observation", insight.observation));
      if (insight.interpretation) {
        li.append(makeEl("p", "insight-interpretation", insight.interpretation));
      }
      if (insight.evidence?.length) {
        li.title = `Evidence: ${insight.evidence.join(", ")}`;
      }
      return li;
    })
  );
}

function renderAiUsage(usage, enabled) {
  if (!aiSpend) return;
  const visible = isLive && enabled && usage && Object.keys(usage).length > 0;
  aiSpend.hidden = !visible;
  if (!visible) return;
  const cost = Number(usage.cost_usd) || 0;
  const cap = Number(usage.budget_usd) || 0;
  aiSpendValue.textContent = cap > 0 ? `$${cost.toFixed(3)} / $${cap.toFixed(2)}` : `$${cost.toFixed(3)}`;
  aiSpendFill.style.width = `${Math.min(100, Number(usage.budget_used_pct) || 0)}%`;
  aiSpend.classList.toggle("paused", Boolean(usage.paused));
  if (usage.paused) {
    const resume = usage.resume_in_sec ? ` Retrying in ${usage.resume_in_sec}s.` : "";
    aiSpendNote.textContent = `${usage.paused_reason || "Paused."}${resume}`;
  } else {
    const tokens = (usage.input_tokens || 0) + (usage.output_tokens || 0) + (usage.thinking_tokens || 0);
    const skipped = usage.skipped ? ` · ${usage.skipped} avoided by change gating` : "";
    aiSpendNote.textContent =
      `${usage.calls || 0} call${usage.calls === 1 ? "" : "s"} · ${formatTokens(tokens)} tokens${skipped}`;
  }
}

function setIdle() {
  isLive = false;
  snapshotBusy = false;
  resetViewModes();
  youtubeCacheAnnounced = false;
  lastLiveConfig = null;
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
  if (leftRail) leftRail.hidden = true;
  if (rightRail) rightRail.hidden = true;
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
        state: config.show_expressions ? "idle" : "off",
      });
    }
    if (isLive) {
      lastLiveConfig = config;
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
    renderInsights(data.insights);
    renderAiUsage(data.ai_usage, Boolean(data.gemini?.enabled));

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

    if (data.state === "idle" && isLive) {
      finalizeSession(data);
      setIdle();
      showSessionLogActions();
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
      updatePipelineRail(data.config, data.gemini, data.expressions, data.insights);
      ingestIntelFeed(data);
      handleAlerts(data.alerts);
      if (data.recording !== isRecording) setRecordingUI(data.recording);
      if (data.config?.confidence != null) {
        syncThresholdDisplay(data.config.confidence);
      }
      renderDetections(data.tracks, data.objects);
      syncLayerChips(data.config);
      lastLiveConfig = data.config;
      maybeAnnounceYoutubeCache(data.source_kind, data.source_label);
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
      body: JSON.stringify({ source, profile: selectedProfile }),
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
  let saved = false;

  try {
    const res = await fetch("/api/snapshot");
    if (res.ok) {
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `snapshot-${stamp}.png`;
      a.click();
      URL.revokeObjectURL(a.href);
      saved = true;
    }
  } catch {
    /* skip */
  }

  snapshotBusy = false;
  if (isLive) {
    btnSnapshot.disabled = false;
  }
  if (saved && activeSession) {
    activeSession.snapshotCount += 1;
    logSessionEvent("capture", "Snapshot exported");
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
if (toggleLeftPanel) {
  toggleLeftPanel.addEventListener("click", toggleLeftPanelCollapse);
}
if (toggleRightPanel) {
  toggleRightPanel.addEventListener("click", toggleRightPanelCollapse);
}
btnSnapshot.addEventListener("click", takeSnapshot);
btnRecord.addEventListener("click", toggleRecording);
btnRetry.addEventListener("click", retryVision);
btnDownloadRaw.addEventListener("click", downloadRawSessionLog);
btnDownloadEnriched.addEventListener("click", downloadEnrichedSessionLog);
btnDismissLog.addEventListener("click", hideSessionLogActions);
if (btnExportAnalysis) {
  btnExportAnalysis.addEventListener("click", exportAnalysisTimeline);
}

if (profileOptions) {
  profileOptions.addEventListener("click", (e) => {
    const chip = e.target.closest(".profile-chip");
    if (!chip || isLive) return;
    selectProfile(chip.dataset.profile);
  });
}

if (layerToggles) {
  layerToggles.addEventListener("click", (e) => {
    const chip = e.target.closest(".layer-chip");
    if (!chip) return;
    const layer = chip.dataset.layer;
    const next = !chip.classList.contains("active");
    toggleLayer(layer, next);
  });
}

if (pipelineRail) {
  pipelineRail.addEventListener("click", (e) => {
    const item = e.target.closest(".pipe-item");
    if (!item || !isLive) return;
    const layerKey = PIPE_LAYER_MAP[item.dataset.pipe];
    if (!layerKey || !lastLiveConfig) return;
    toggleLayer(layerKey, !lastLiveConfig[layerKey]);
  });
}

if (btnFocus) {
  btnFocus.addEventListener("click", toggleFocusMode);
}
if (btnClean) {
  btnClean.addEventListener("click", toggleCleanFeed);
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
    return;
  }
  if ((e.key === "f" || e.key === "F") && !btnFocus?.disabled) {
    e.preventDefault();
    toggleFocusMode();
    return;
  }
  if ((e.key === "c" || e.key === "C") && !btnClean?.disabled) {
    e.preventDefault();
    toggleCleanFeed();
  }
});

async function hydrateFromServer() {
  try {
    const [configRes, statusRes, healthRes] = await Promise.all([
      fetch("/api/config"),
      fetch("/api/status"),
      fetch("/api/health"),
    ]);
    const config = await configRes.json();
    const status = await statusRes.json();
    const health = await healthRes.json().catch(() => ({}));

    if (health.youtube_cache_dir) {
      youtubeCacheDir = health.youtube_cache_dir;
      updateSourceCacheHint();
    }
    setProfileChoices(health.profiles, health.default_profile);
    if (health.report_steps?.length) {
      reportSteps = health.report_steps;
    }

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

function profileLabel(key) {
  const match = profileChoices.find((p) => p.key === key);
  if (match) return match.label;
  return key ? key.charAt(0).toUpperCase() + key.slice(1) : "General";
}

function renderProfileOptions() {
  if (!profileOptions) return;
  for (const chip of profileOptions.querySelectorAll(".profile-chip")) {
    const on = chip.dataset.profile === selectedProfile;
    chip.classList.toggle("active", on);
    chip.setAttribute("aria-checked", String(on));
  }
  const profile = profileChoices.find((p) => p.key === selectedProfile);
  if (profileHint && profile?.description) profileHint.textContent = profile.description;
}

function setProfileChoices(list, defaultKey) {
  if (!Array.isArray(list) || !list.length || !profileOptions) return;
  profileChoices = list;
  profileOptions.replaceChildren(
    ...list.map((profile) => {
      const chip = makeEl("button", "profile-chip", profile.label);
      chip.type = "button";
      chip.dataset.profile = profile.key;
      chip.setAttribute("role", "radio");
      chip.title = profile.description || "";
      return chip;
    })
  );
  if (!list.some((p) => p.key === selectedProfile)) {
    selectedProfile = list.some((p) => p.key === defaultKey) ? defaultKey : list[0].key;
  }
  renderProfileOptions();
}

function selectProfile(key) {
  if (!key) return;
  selectedProfile = key;
  try {
    localStorage.setItem("yubi-profile", key);
  } catch {
    /* storage unavailable */
  }
  renderProfileOptions();
}

function restoreSavedSource() {
  try {
    const savedProfile = localStorage.getItem("yubi-profile");
    if (savedProfile) selectedProfile = savedProfile;
  } catch {
    /* storage unavailable */
  }
  renderProfileOptions();
  if (!sourceInput) return;
  try {
    const saved = localStorage.getItem("yubi-source");
    if (saved) sourceInput.value = saved;
  } catch {
    /* storage unavailable */
  }
}

restoreSavedSource();
loadPanelCollapsePrefs();
applyPanelCollapse();
hydrateFromServer();
startTelemetryClock();
