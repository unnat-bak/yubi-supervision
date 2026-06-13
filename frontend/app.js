const stream = document.getElementById("stream");
const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const statusPill = document.getElementById("status-pill");
const statusLabel = document.getElementById("status-label");
const errorMsg = document.getElementById("error-msg");
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
const geminiPanel = document.getElementById("gemini-panel");
const geminiState = document.getElementById("gemini-state");
const geminiSummary = document.getElementById("gemini-summary");
const geminiList = document.getElementById("gemini-list");
const geminiEmpty = document.getElementById("gemini-empty");
const geminiError = document.getElementById("gemini-error");

let pollTimer = null;
let isLive = false;
let configDebounce = null;
let isRecording = false;
let lastAlertTs = 0;
let alertHideTimer = null;
let lastTracks = [];
let lastGrouped = [];
let activeThreshold = 0.35;

function setError(message) {
  if (message) {
    errorMsg.textContent = message;
    errorMsg.hidden = false;
    statusPill.classList.remove("idle", "live");
    statusPill.classList.add("error");
    statusLabel.textContent = "Error";
  } else {
    errorMsg.hidden = true;
    errorMsg.textContent = "";
  }
}

function syncLayerChips(config) {
  if (!config) return;
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
  alertBanner.textContent = `⚠ ${latest.label} detected · ${latest.time}`;
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
  btnRecord.lastChild.textContent = recording ? "Stop Rec" : "Rec";
}

function updateDegraded(degraded) {
  if (degraded?.length) {
    degradedBadge.textContent = `⚠ ${degraded.join(", ")} unavailable`;
    degradedBadge.hidden = false;
  } else {
    degradedBadge.hidden = true;
  }
}

function renderGemini(gemini) {
  if (!gemini) return;

  const stateLabels = {
    disabled: "Off",
    idle: "Idle",
    thinking: "Thinking",
    ready: "Live",
    error: "Error",
  };

  geminiState.textContent = stateLabels[gemini.state] || gemini.state;
  geminiState.className = "panel-badge gemini-badge";
  if (gemini.state === "thinking") geminiState.classList.add("thinking");
  if (gemini.state === "ready") geminiState.classList.add("ready");

  if (!gemini.enabled) {
    geminiSummary.textContent =
      "Add GEMINI_API_KEY to .env for semantic scene analysis.";
    geminiList.replaceChildren();
    geminiEmpty.hidden = true;
    geminiError.hidden = true;
    return;
  }

  if (gemini.error) {
    geminiError.textContent = gemini.error;
    geminiError.hidden = false;
  } else {
    geminiError.hidden = true;
    geminiError.textContent = "";
  }

  geminiSummary.textContent =
    gemini.scene_summary ||
    (gemini.state === "thinking"
      ? "Analyzing scene with Gemini…"
      : "Waiting for first Gemini analysis…");

  geminiList.replaceChildren();
  const items = gemini.objects || [];
  geminiEmpty.hidden = items.length > 0 || gemini.state === "thinking";
  geminiEmpty.textContent =
    gemini.state === "thinking" ? "Analyzing…" : "No Gemini objects yet";

  for (const item of items) {
    const li = document.createElement("li");
    li.className = "gemini-item";
    const label = document.createElement("span");
    label.className = "gemini-item-label";
    label.textContent = item.label;
    const meta = document.createElement("span");
    meta.className = "gemini-item-meta";
    meta.textContent = `${Math.round(item.confidence * 100)}% confidence`;
    li.append(label, meta);
    geminiList.appendChild(li);
  }
}

function setIdle() {
  isLive = false;
  stream.removeAttribute("src");
  stream.classList.remove("live");
  btnStart.disabled = false;
  btnStop.disabled = true;
  btnSnapshot.disabled = true;
  btnRecord.disabled = true;
  setRecordingUI(false);
  alertBanner.hidden = true;
  statusPill.classList.remove("live", "error");
  statusPill.classList.add("idle");
  statusLabel.textContent = "Idle";
  idleOverlay.classList.remove("hidden");
  statsBar.hidden = true;
  detectionsPanel.hidden = true;
  geminiPanel.hidden = true;
  loadingOverlay.hidden = true;
  renderDetections([], []);
  updateStats({});
  updateDegraded([]);
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
    const config = await res.json();
    syncLayerChips(config);
  } catch {
    /* ignore */
  }
}

async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.error) {
      setError(data.error);
      return;
    }
    if (data.state === "live") {
      updateStats(data);
      updateDegraded(data.degraded);
      handleAlerts(data.alerts);
      if (data.recording !== isRecording) setRecordingUI(data.recording);
      if (data.config?.confidence != null) {
        syncThresholdDisplay(data.config.confidence);
      }
      renderDetections(data.tracks, data.objects);
      renderGemini(data.gemini);
      syncLayerChips(data.config);
    } else if (data.state === "starting") {
      statusPill.classList.remove("idle", "live", "error");
      statusPill.classList.add("starting");
      statusLabel.textContent = "Starting";
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

  try {
    const res = await fetch("/api/start", { method: "POST" });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || "Failed to start vision");
    }

    await waitUntilLive();

    isLive = true;
    stream.src = `/api/stream?${Date.now()}`;
    stream.classList.add("live");
    btnStop.disabled = false;
    btnSnapshot.disabled = false;
    btnRecord.disabled = false;
    statsBar.hidden = false;
    detectionsPanel.hidden = false;
    geminiPanel.hidden = false;
    statusPill.classList.remove("idle", "error");
    statusPill.classList.add("live");
    statusLabel.textContent = "Live";
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
  try {
    await fetch("/api/stop", { method: "POST" });
  } catch {
    /* still reset UI */
  }
  setIdle();
}

async function takeSnapshot() {
  if (!isLive) return;
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
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
    } catch {
      /* skip */
    }
  }
}

async function toggleRecording() {
  if (!isLive) return;
  const action = isRecording ? "stop" : "start";
  try {
    const res = await fetch(`/api/record/${action}`, { method: "POST" });
    if (res.ok) {
      const data = await res.json();
      setRecordingUI(data.recording);
    }
  } catch {
    /* keep current state */
  }
}

btnStart.addEventListener("click", startVision);
btnStop.addEventListener("click", stopVision);
btnSnapshot.addEventListener("click", takeSnapshot);
btnRecord.addEventListener("click", toggleRecording);

layerToggles.addEventListener("click", (e) => {
  const chip = e.target.closest(".layer-chip");
  if (!chip || !isLive) return;
  const layer = chip.dataset.layer;
  const next = !chip.classList.contains("active");
  chip.classList.toggle("active", next);
  pushConfig({ [layer]: next });
});

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
  if (e.code === "Space" && !isLive && !btnStart.disabled) {
    e.preventDefault();
    startVision();
    return;
  }
  if (!isLive) return;
  if (e.key === "Escape" || e.key === "q" || e.key === "Q") {
    stopVision();
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

    if (status.state === "live") {
      isLive = true;
      stream.src = `/api/stream?${Date.now()}`;
      stream.classList.add("live");
      btnStart.disabled = true;
      btnStop.disabled = false;
      statsBar.hidden = false;
      detectionsPanel.hidden = false;
    geminiPanel.hidden = false;
      idleOverlay.classList.add("hidden");
      statusPill.classList.remove("idle", "error");
      statusPill.classList.add("live");
      statusLabel.textContent = "Live";
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
      statusLabel.textContent = "Starting";
      if (status.startup_message) {
        loadingText.textContent = status.startup_message;
      }
      try {
        await waitUntilLive();
        isLive = true;
        stream.src = `/api/stream?${Date.now()}`;
        stream.classList.add("live");
        btnStop.disabled = false;
        statsBar.hidden = false;
        detectionsPanel.hidden = false;
    geminiPanel.hidden = false;
        statusPill.classList.remove("starting", "error");
        statusPill.classList.add("live");
        statusLabel.textContent = "Live";
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

hydrateFromServer();
