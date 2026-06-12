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

let pollTimer = null;
let isLive = false;
let configDebounce = null;

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

function renderDetections(tracks, grouped, objectCount) {
  detectionsList.replaceChildren();
  objectTotal.textContent = String(objectCount);
  const items = tracks?.length ? tracks : grouped || [];
  detectionsEmpty.hidden = items.length > 0;

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
}

function setIdle() {
  isLive = false;
  stream.removeAttribute("src");
  stream.classList.remove("live");
  btnStart.disabled = false;
  btnStop.disabled = true;
  statusPill.classList.remove("live", "error");
  statusPill.classList.add("idle");
  statusLabel.textContent = "Idle";
  idleOverlay.classList.remove("hidden");
  statsBar.hidden = true;
  detectionsPanel.hidden = true;
  loadingOverlay.hidden = true;
  renderDetections([], [], 0);
  updateStats({});
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
      renderDetections(data.tracks, data.objects, data.object_count || 0);
      syncLayerChips(data.config);
      if (data.config?.confidence != null) {
        const pct = Math.round(data.config.confidence * 100);
        confidenceSlider.value = String(pct);
        confidenceValue.textContent = `${pct}%`;
      }
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
    statsBar.hidden = false;
    detectionsPanel.hidden = false;
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

btnStart.addEventListener("click", startVision);
btnStop.addEventListener("click", stopVision);

layerToggles.addEventListener("click", (e) => {
  const chip = e.target.closest(".layer-chip");
  if (!chip || !isLive) return;
  const layer = chip.dataset.layer;
  const next = !chip.classList.contains("active");
  chip.classList.toggle("active", next);
  pushConfig({ [layer]: next });
});

confidenceSlider.addEventListener("input", () => {
  const pct = Number(confidenceSlider.value);
  confidenceValue.textContent = `${pct}%`;
  clearTimeout(configDebounce);
  configDebounce = setTimeout(() => {
    if (isLive) pushConfig({ confidence: pct / 100 });
  }, 150);
});

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

fetch("/api/config")
  .then((r) => r.json())
  .then((config) => {
    syncLayerChips(config);
    if (config.confidence != null) {
      const pct = Math.round(config.confidence * 100);
      confidenceSlider.value = String(pct);
      confidenceValue.textContent = `${pct}%`;
    }
  })
  .catch(() => {});
