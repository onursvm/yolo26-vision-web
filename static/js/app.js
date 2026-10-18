/**
 * YOLO26 Vision AI - Client Application Logic
 * Features:
 * - Direct Webcam / Mobile Camera access (getUserMedia)
 * - Real-time HUD canvas rendering with high-tech bounding boxes
 * - English / Turkish dual language support
 * - Dark / Light theme switching
 * - Front / Back Camera selection (mobile & desktop)
 * - Snapshot capture and download
 */

// --- Localization Translations ---
const I18N = {
  tr: {
    appTitle: "YOLO26 Vision AI",
    appSubtitle: "Canlı Nesne Algılama",
    live: "CANLI",
    idleTitle: "Kamera Beklemede",
    idleSubtitle: "Canlı nesne algılamayı başlatmak için kamerayı açın.",
    btnStartCamera: "Kamerayı Başlat",
    btnStopCamera: "Durdur",
    btnSnapshot: "Fotoğraf Çek",
    confidenceLabel: "Güven Eşiği",
    cameraSelectDefault: "Varsayılan Kamera",
    cardDetectionsTitle: "Algılanan Nesneler",
    emptyDetections: "Şu anda nesne algılanmadı",
    metricInference: "Model Gecikmesi",
    metricFps: "FPS",
    metricCount: "Toplam Nesne",
    metricModel: "Model",
    snapshotSaved: "Ekran görüntüsü kaydedildi!"
  },
  en: {
    appTitle: "YOLO26 Vision AI",
    appSubtitle: "Real-time Object Detection",
    live: "LIVE",
    idleTitle: "Camera Idle",
    idleSubtitle: "Start your camera to begin real-time object detection.",
    btnStartCamera: "Start Camera",
    btnStopCamera: "Stop",
    btnSnapshot: "Snapshot",
    confidenceLabel: "Confidence",
    cameraSelectDefault: "Default Camera",
    cardDetectionsTitle: "Detected Objects",
    emptyDetections: "No objects detected yet",
    metricInference: "Inference Latency",
    metricFps: "FPS",
    metricCount: "Total Objects",
    metricModel: "Model",
    snapshotSaved: "Snapshot saved!"
  }
};

// --- App State ---
const state = {
  lang: localStorage.getItem('yolo_lang') || 'tr',
  theme: localStorage.getItem('yolo_theme') || 'dark',
  isStreaming: false,
  confidence: 0.40,
  selectedDeviceId: null,
  currentStream: null,
  isProcessingFrame: false,
  detectionLoopId: null,
  fpsCounter: 0,
  lastFpsTime: performance.now(),
  currentFps: 0,
  activeDetections: [],
  isMobileDevice: /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
};

// --- DOM Elements ---
const videoEl = document.getElementById('webcam-video');
const canvasEl = document.getElementById('detection-canvas');
const ctx = canvasEl.getContext('2d');
const idleScreenEl = document.getElementById('idle-screen');
const btnStartCam = document.getElementById('btn-start-camera');
const btnStopCam = document.getElementById('btn-stop-camera');
const btnSnapshot = document.getElementById('btn-snapshot');
const confidenceSlider = document.getElementById('confidence-slider');
const confidenceValEl = document.getElementById('confidence-val');
const cameraSelectEl = document.getElementById('camera-select');
const detectionsListEl = document.getElementById('detections-list');
const countBadgeEl = document.getElementById('count-badge');
const metricInferenceEl = document.getElementById('metric-inference');
const metricFpsEl = document.getElementById('metric-fps');
const metricCountEl = document.getElementById('metric-count');
const hudFpsEl = document.getElementById('hud-fps');
const hudLatencyEl = document.getElementById('hud-latency');

// Hidden canvas for optimized frame extraction
const hiddenCanvas = document.createElement('canvas');
const hiddenCtx = hiddenCanvas.getContext('2d');

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
  applyTheme(state.theme);
  applyLanguage(state.lang);
  initEventListeners();
  loadVideoDevices();
});

// --- Theme Management ---
function applyTheme(theme) {
  state.theme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('yolo_theme', theme);

  document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.theme === theme);
  });
}

// --- Language Management ---
function applyLanguage(lang) {
  state.lang = lang;
  localStorage.setItem('yolo_lang', lang);

  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.lang === lang);
  });

  const t = I18N[lang] || I18N.tr;

  // Update elements with data-i18n attribute
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (t[key]) {
      el.textContent = t[key];
    }
  });

  // Refresh detections list to match language
  renderDetectionsList(state.activeDetections);
}

// --- Device Cameras Enumeration ---
async function loadVideoDevices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === 'videoinput');

    cameraSelectEl.innerHTML = '';
    const defaultOpt = document.createElement('option');
    defaultOpt.value = '';
    defaultOpt.textContent = I18N[state.lang].cameraSelectDefault;
    cameraSelectEl.appendChild(defaultOpt);

    videoDevices.forEach((dev, idx) => {
      const opt = document.createElement('option');
      opt.value = dev.deviceId;
      opt.textContent = dev.label || `Kamera ${idx + 1}`;
      cameraSelectEl.appendChild(opt);
    });

    if (videoDevices.length > 1) {
      cameraSelectEl.style.display = 'block';
    }
  } catch (err) {
    console.warn("Kamera listesi alınamadı:", err);
  }
}

// --- Event Listeners ---
function initEventListeners() {
  // Theme Buttons
  document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.addEventListener('click', () => applyTheme(btn.dataset.theme));
  });

  // Language Buttons
  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.addEventListener('click', () => applyLanguage(btn.dataset.lang));
  });

  // Start & Stop Controls
  btnStartCam.addEventListener('click', startCamera);
  btnStopCam.addEventListener('click', stopCamera);
  btnSnapshot.addEventListener('click', captureSnapshot);

  // Confidence Slider
  confidenceSlider.addEventListener('input', (e) => {
    state.confidence = parseFloat(e.target.value) / 100;
    confidenceValEl.textContent = `%${e.target.value}`;
  });

  // Camera Selector
  cameraSelectEl.addEventListener('change', (e) => {
    state.selectedDeviceId = e.target.value;
    if (state.isStreaming) {
      startCamera();
    }
  });
}

// --- Local Camera Streaming ---
async function startCamera() {
  stopCamera();

  const constraints = {
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: state.isMobileDevice ? { ideal: 'environment' } : 'user'
    },
    audio: false
  };

  if (state.selectedDeviceId) {
    constraints.video.deviceId = { exact: state.selectedDeviceId };
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    state.currentStream = stream;
    videoEl.srcObject = stream;
    await videoEl.play();

    state.isStreaming = true;
    idleScreenEl.style.display = 'none';
    videoEl.style.display = 'block';
    canvasEl.style.display = 'block';

    btnStartCam.style.display = 'none';
    btnStopCam.style.display = 'inline-flex';
    btnSnapshot.style.display = 'inline-flex';

    // Set canvas internal resolution to match video
    videoEl.addEventListener('loadedmetadata', resizeCanvas);
    resizeCanvas();

    // Begin real-time detection loop
    startDetectionLoop();
  } catch (err) {
    console.error("Kamera açılamadı:", err);
    alert("Kameraya erişilemedi: " + (err.message || err.name));
  }
}

function resizeCanvas() {
  if (videoEl.videoWidth && videoEl.videoHeight) {
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
    hiddenCanvas.width = 640; // optimized size for inference
    hiddenCanvas.height = Math.round((videoEl.videoHeight / videoEl.videoWidth) * 640);
  }
}

// --- Detection Loop (REST POST /api/detect) ---
async function startDetectionLoop() {
  if (!state.isStreaming) return;

  async function loop() {
    if (!state.isStreaming) return;

    if (!state.isProcessingFrame && videoEl.readyState === videoEl.HAVE_ENOUGH_DATA) {
      state.isProcessingFrame = true;
      try {
        await captureAndDetectFrame();
      } catch (e) {
        console.error("Frame detection error:", e);
      } finally {
        state.isProcessingFrame = false;
      }
    }

    state.detectionLoopId = requestAnimationFrame(loop);
  }

  state.detectionLoopId = requestAnimationFrame(loop);
}

async function captureAndDetectFrame() {
  hiddenCtx.drawImage(videoEl, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
  const base64Data = hiddenCanvas.toDataURL('image/jpeg', 0.65);

  const reqStart = performance.now();
  const response = await fetch('/api/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image: base64Data,
      confidence: state.confidence
    })
  });

  if (!response.ok) return;

  const data = await response.json();
  const totalLatency = Math.round(performance.now() - reqStart);

  // Update FPS Counter
  state.fpsCounter++;
  const now = performance.now();
  if (now - state.lastFpsTime >= 1000) {
    state.currentFps = Math.round((state.fpsCounter * 1000) / (now - state.lastFpsTime));
    state.fpsCounter = 0;
    state.lastFpsTime = now;
  }

  // Draw detections onto canvas overlay
  renderBoundingBoxes(data.detections || []);
  state.activeDetections = data.detections || [];
  renderDetectionsList(state.activeDetections);

  // Update Metrics
  metricInferenceEl.textContent = `${data.inference_time_ms || totalLatency} ms`;
  metricFpsEl.textContent = `${state.currentFps}`;
  metricCountEl.textContent = `${data.count || 0}`;
  countBadgeEl.textContent = `${data.count || 0}`;
  hudFpsEl.textContent = `${state.currentFps} FPS`;
  hudLatencyEl.textContent = `${data.inference_time_ms || totalLatency}ms`;
}

// --- High-Tech HUD Bounding Box Rendering ---
function renderBoundingBoxes(detections) {
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  const scaleX = canvasEl.width / (hiddenCanvas.width || canvasEl.width);
  const scaleY = canvasEl.height / (hiddenCanvas.height || canvasEl.height);

  detections.forEach(det => {
    let x1, y1, x2, y2;
    if (det.box_normalized) {
      x1 = det.box_normalized[0] * canvasEl.width;
      y1 = det.box_normalized[1] * canvasEl.height;
      x2 = det.box_normalized[2] * canvasEl.width;
      y2 = det.box_normalized[3] * canvasEl.height;
    } else {
      x1 = det.box[0] * scaleX;
      y1 = det.box[1] * scaleY;
      x2 = det.box[2] * scaleX;
      y2 = det.box[3] * scaleY;
    }

    const boxW = x2 - x1;
    const boxH = y2 - y1;

    // Pick vibrant color based on class ID
    const hue = (det.class_id * 47) % 360;
    const boxColor = `hsl(${hue}, 90%, 55%)`;
    const boxBg = `hsla(${hue}, 90%, 55%, 0.12)`;

    // Draw translucent highlight box
    ctx.fillStyle = boxBg;
    ctx.fillRect(x1, y1, boxW, boxH);

    // Draw border
    ctx.strokeStyle = boxColor;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, boxW, boxH);

    // Draw futuristic HUD corner brackets
    const cornerSize = Math.min(16, boxW / 4, boxH / 4);
    ctx.lineWidth = 4;

    // Top-left corner
    ctx.beginPath();
    ctx.moveTo(x1, y1 + cornerSize);
    ctx.lineTo(x1, y1);
    ctx.lineTo(x1 + cornerSize, y1);
    ctx.stroke();

    // Top-right corner
    ctx.beginPath();
    ctx.moveTo(x2 - cornerSize, y1);
    ctx.lineTo(x2, y1);
    ctx.lineTo(x2, y1 + cornerSize);
    ctx.stroke();

    // Bottom-left corner
    ctx.beginPath();
    ctx.moveTo(x1, y2 - cornerSize);
    ctx.lineTo(x1, y2);
    ctx.lineTo(x1 + cornerSize, y2);
    ctx.stroke();

    // Bottom-right corner
    ctx.beginPath();
    ctx.moveTo(x2 - cornerSize, y2);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x2, y2 - cornerSize);
    ctx.stroke();

    // Label Chip
    const name = state.lang === 'tr' ? (det.name_tr || det.name) : det.name;
    const confPct = Math.round(det.confidence * 100);
    const labelText = `${name} %${confPct}`;

    ctx.font = 'bold 13px "JetBrains Mono", monospace';
    const textW = ctx.measureText(labelText).width;
    const padX = 8;
    const padY = 5;
    const chipH = 22;

    const chipY = Math.max(0, y1 - chipH - 3);

    // Chip Background
    ctx.fillStyle = boxColor;
    ctx.beginPath();
    ctx.roundRect(x1, chipY, textW + (padX * 2), chipH, [4, 4, 4, 4]);
    ctx.fill();

    // Chip Text
    ctx.fillStyle = '#000000';
    ctx.fillText(labelText, x1 + padX, chipY + chipH - padY);
  });
}

// --- Render Sidebar Detections List ---
function renderDetectionsList(detections) {
  if (!detections || detections.length === 0) {
    const t = I18N[state.lang] || I18N.tr;
    detectionsListEl.innerHTML = `
      <div class="empty-detections">
        <span style="font-size: 1.75rem;">🔍</span>
        <span>${t.emptyDetections}</span>
      </div>
    `;
    return;
  }

  let html = '';
  detections.forEach(det => {
    const name = state.lang === 'tr' ? (det.name_tr || det.name) : det.name;
    const subName = state.lang === 'tr' ? det.name : (det.name_tr || '');
    const conf = Math.round(det.confidence * 100);
    const hue = (det.class_id * 47) % 360;

    html += `
      <div class="detection-item">
        <div class="detection-info">
          <div class="detection-name">
            <span style="width: 8px; height: 8px; border-radius: 50%; background: hsl(${hue}, 90%, 55%); display: inline-block;"></span>
            ${escapeHtml(name)}
          </div>
          <div class="detection-sub">${escapeHtml(subName)}</div>
        </div>
        <div class="detection-confidence">
          <span class="confidence-val">%${conf}</span>
          <div class="confidence-bar-bg">
            <div class="confidence-bar-fill" style="width: ${conf}%;"></div>
          </div>
        </div>
      </div>
    `;
  });

  detectionsListEl.innerHTML = html;
}

// --- Stop Camera ---
function stopCamera() {
  state.isStreaming = false;

  if (state.detectionLoopId) {
    cancelAnimationFrame(state.detectionLoopId);
    state.detectionLoopId = null;
  }

  if (state.currentStream) {
    state.currentStream.getTracks().forEach(track => track.stop());
    state.currentStream = null;
  }

  videoEl.srcObject = null;

  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  idleScreenEl.style.display = 'flex';
  videoEl.style.display = 'none';
  canvasEl.style.display = 'none';

  btnStartCam.style.display = 'inline-flex';
  btnStopCam.style.display = 'none';
  btnSnapshot.style.display = 'none';

  state.activeDetections = [];
  renderDetectionsList([]);
  metricCountEl.textContent = '0';
  countBadgeEl.textContent = '0';
  hudFpsEl.textContent = '0 FPS';
}

// --- Snapshot Capture & Download ---
function captureSnapshot() {
  if (!state.isStreaming) return;

  const snapCanvas = document.createElement('canvas');
  snapCanvas.width = canvasEl.width || videoEl.videoWidth || 1280;
  snapCanvas.height = canvasEl.height || videoEl.videoHeight || 720;
  const sCtx = snapCanvas.getContext('2d');

  sCtx.drawImage(videoEl, 0, 0, snapCanvas.width, snapCanvas.height);
  sCtx.drawImage(canvasEl, 0, 0, snapCanvas.width, snapCanvas.height);

  const link = document.createElement('a');
  link.download = `yolo26_snapshot_${Date.now()}.jpg`;
  link.href = snapCanvas.toDataURL('image/jpeg', 0.92);
  link.click();
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
