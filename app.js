import { HandLandmarker, FilesetResolver } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const select = document.getElementById("camera-select");
const startBtn = document.getElementById("start-btn");
const mirrorToggle = document.getElementById("mirror-toggle");
const statusEl = document.getElementById("status");
const hint = document.getElementById("hint");

// Parámetros sintonizables vía URL:
// ?debug=1&thumbMCP=145&thumbIP=145&thumbDist=0.5
const params = new URLSearchParams(location.search);
const DEBUG = params.has("debug");
const THUMB_MCP_MIN = parseFloat(params.get("thumbMCP") || "145");
const THUMB_IP_MIN  = parseFloat(params.get("thumbIP")  || "145");
const THUMB_DIST_MIN = parseFloat(params.get("thumbDist") || "0.5");

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

// Los 4 dedos "normales": [tip, articulación PIP]
const FINGERS_DIST = [[8, 6], [12, 10], [16, 14], [20, 18]];

let handLandmarker = null;
let stream = null;
let running = false;
let rafId = null;
let lastVideoTime = -1;

// ---------- Utilidades ----------

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// Ángulo (en grados) en el punto b, formado por a-b-c
function angleDeg(a, b, c) {
  const v1x = a.x - b.x, v1y = a.y - b.y;
  const v2x = c.x - b.x, v2y = c.y - b.y;
  const dot = v1x * v2x + v1y * v2y;
  const m1 = Math.hypot(v1x, v1y);
  const m2 = Math.hypot(v2x, v2y);
  if (m1 === 0 || m2 === 0) return 0;
  const cos = Math.max(-1, Math.min(1, dot / (m1 * m2)));
  return Math.acos(cos) * 180 / Math.PI;
}

// ---------- Conteo de dedos ----------

function isThumbExtended(lm, debugInfo) {
  const palmSize = dist(lm[0], lm[9]);         // muñeca → MCP del medio
  const tipToIndex = dist(lm[4], lm[5]) / palmSize;
  const mcpAngle = angleDeg(lm[1], lm[2], lm[4]); // ángulo en la MCP del pulgar
  const ipAngle  = angleDeg(lm[2], lm[3], lm[4]); // ángulo en la IP del pulgar

  const extended =
    mcpAngle > THUMB_MCP_MIN &&
    ipAngle  > THUMB_IP_MIN  &&
    tipToIndex > THUMB_DIST_MIN;

  if (debugInfo) {
    debugInfo.mcp = Math.round(mcpAngle);
    debugInfo.ip  = Math.round(ipAngle);
    debugInfo.dst = tipToIndex.toFixed(2);
    debugInfo.ext = extended;
  }
  return extended;
}

function countFingers(lm, debugInfo) {
  const wrist = lm[0];
  let count = 0;

  // 4 dedos normales por distancia tip vs PIP a la muñeca
  for (const [tip, ref] of FINGERS_DIST) {
    if (dist(lm[tip], wrist) > dist(lm[ref], wrist) * 1.1) count++;
  }

  // Pulgar por ángulos + distancia a la MCP del índice
  if (isThumbExtended(lm, debugInfo)) count++;

  return count;
}

function drawCartoonGlove(lm) {
  const point = (index) => ({
    x: lm[index].x * canvas.width,
    y: lm[index].y * canvas.height,
  });
  const wrist = point(0);
  const middleBase = point(9);
  const palmWidth = Math.hypot(point(5).x - point(17).x, point(5).y - point(17).y);
  const outlineWidth = palmWidth * 0.065;
  const fingerWidth = palmWidth * 0.17;
  const thumbWidth = palmWidth * 0.19;

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  const forearmX = middleBase.x - wrist.x;
  const forearmY = middleBase.y - wrist.y;
  const forearmLength = Math.hypot(forearmX, forearmY);
  if (forearmLength > 0) {
    ctx.beginPath();
    ctx.moveTo(wrist.x, wrist.y);
    ctx.lineTo(
      wrist.x - forearmX / forearmLength * Math.max(canvas.height * 0.55, palmWidth * 1.8),
      wrist.y - forearmY / forearmLength * Math.max(canvas.height * 0.55, palmWidth * 1.8),
    );
    ctx.strokeStyle = "#080808";
    ctx.lineWidth = palmWidth * 0.34;
    ctx.stroke();
  }

  const palmPoints = [0, 1, 5, 9, 13, 17].map(point);
  ctx.beginPath();
  ctx.moveTo(palmPoints[0].x, palmPoints[0].y);
  for (const p of palmPoints.slice(1)) ctx.lineTo(p.x, p.y);
  ctx.closePath();
  ctx.fillStyle = "#fff";
  ctx.fill();
  ctx.strokeStyle = "#111";
  ctx.lineWidth = outlineWidth * 2;
  ctx.stroke();

  const fingerChains = [
    { points: [1, 2, 3, 4], width: thumbWidth },
    { points: [5, 6, 7, 8], width: fingerWidth },
    { points: [9, 10, 11, 12], width: fingerWidth },
    { points: [13, 14, 15, 16], width: fingerWidth },
    { points: [17, 18, 19, 20], width: fingerWidth },
  ];

  for (const chain of fingerChains) {
    ctx.beginPath();
    chain.points.forEach((index, i) => {
      const p = point(index);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.strokeStyle = "#111";
    ctx.lineWidth = chain.width + outlineWidth * 2;
    ctx.stroke();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = chain.width;
    ctx.stroke();
  }

  ctx.restore();
}

// ---------- Modelo ----------

async function initModel() {
  setStatus("Cargando modelo de MediaPipe…");
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 2,
  });
  setStatus("Modelo listo. Elige cámara e inicia.");
  startBtn.disabled = false;
}

async function refreshCameras() {
  try {
    const s = await navigator.mediaDevices.getUserMedia({ video: true });
    s.getTracks().forEach(t => t.stop());
  } catch {}
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter(d => d.kind === "videoinput");
  select.innerHTML = "";
  if (cams.length === 0) {
    select.innerHTML = '<option value="">Sin cámaras</option>';
    return;
  }
  cams.forEach((cam, i) => {
    const opt = document.createElement("option");
    opt.value = cam.deviceId;
    opt.textContent = cam.label || `Cámara ${i + 1}`;
    select.appendChild(opt);
  });
}

// ---------- Cámara ----------

async function startCamera(deviceId) {
  stopCamera();
  const constraints = {
    audio: false,
    video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "user" },
  };
  stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;
  await video.play();
  const track = stream.getVideoTracks()[0];
  setStatus(`Cámara activa: ${track.label || "desconocida"}`);
  hint.classList.add("hidden");
  if (video.videoWidth) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  applyMirror();
  running = true;
  startBtn.textContent = "Detener";
  lastVideoTime = -1;
  rafId = requestAnimationFrame(detectLoop);
}

function stopCamera() {
  running = false;
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  startBtn.textContent = "Iniciar";
  hint.classList.remove("hidden");
}

function applyMirror() {
  const on = mirrorToggle.checked;
  video.classList.toggle("mirror", on);
  canvas.classList.toggle("mirror", on);
}

// ---------- Detección y dibujo ----------

function detectLoop() {
  if (!running) return;

  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const results = handLandmarker.detectForVideo(video, performance.now());
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const hands = results.landmarks || [];
    const scale = Math.max(canvas.width, canvas.height) / 600;
    let totalFingers = 0;

    for (const lm of hands) {
      const debugInfo = DEBUG ? {} : null;
      const fingers = countFingers(lm, debugInfo);
      totalFingers += fingers;

      drawCartoonGlove(lm);

      // Número grande sobre la palma (landmark 9)
      const palm = lm[9];
      const px = palm.x * canvas.width;
      const py = palm.y * canvas.height;
      const fontSize = Math.max(40, 90 * scale);

      ctx.font = `bold ${fontSize}px system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.lineWidth = Math.max(4, 8 * scale);
      ctx.strokeStyle = "rgba(0,0,0,0.75)";
      ctx.strokeText(fingers, px, py);
      ctx.fillStyle = "#fbbf24";
      ctx.fillText(fingers, px, py);

      // Debug: mostrar valores del pulgar junto a la punta (landmark 4)
      if (DEBUG && debugInfo) {
        const tip = lm[4];
        const tx = tip.x * canvas.width;
        const ty = tip.y * canvas.height;
        const fsz = Math.max(12, 18 * scale);

        ctx.font = `bold ${fsz}px monospace`;
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.lineWidth = Math.max(2, 3 * scale);
        ctx.strokeStyle = "rgba(0,0,0,0.85)";

        const label = `MCP ${debugInfo.mcp}° IP ${debugInfo.ip}° D ${debugInfo.dst}`;
        ctx.strokeText(label, tx + 10 * scale, ty);
        ctx.fillStyle = debugInfo.ext ? "#22c55e" : "#ef4444";
        ctx.fillText(label, tx + 10 * scale, ty);
      }
    }

    // Estado inferior
    const cfg = DEBUG
      ? ` · MCP>${THUMB_MCP_MIN}° IP>${THUMB_IP_MIN}° D>${THUMB_DIST_MIN}`
      : "";
    if (hands.length === 0) {
      setStatus("Acerca tu mano a la cámara…");
    } else if (hands.length === 1) {
      setStatus(`Manos: 1 · Dedos extendidos: ${totalFingers}${cfg}`);
    } else {
      const detalle = hands.map(lm => countFingers(lm, null)).join(" + ");
      setStatus(`Manos: ${hands.length} · Dedos: ${totalFingers} (${detalle})${cfg}`);
    }
  }

  rafId = requestAnimationFrame(detectLoop);
}

// ---------- UI ----------

function setStatus(msg) { statusEl.textContent = msg; }

startBtn.addEventListener("click", async () => {
  if (running) {
    stopCamera();
    setStatus("Detenido.");
    return;
  }
  startBtn.disabled = true;
  try {
    await startCamera(select.value || undefined);
  } catch (e) {
    setStatus(`Error: ${e.message}`);
    console.error(e);
  } finally {
    startBtn.disabled = false;
  }
});

select.addEventListener("change", async () => {
  if (running) {
    try { await startCamera(select.value); }
    catch (e) { setStatus(`Error: ${e.message}`); }
  }
});

mirrorToggle.addEventListener("change", applyMirror);

// ---------- Arranque ----------

(async () => {
  try {
    await refreshCameras();
    await initModel();
  } catch (e) {
    setStatus(`Error inicial: ${e.message}`);
    console.error(e);
  }
})();
