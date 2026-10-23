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

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17]
];

// [tip, articulación de referencia] para cada dedo
const FINGERS = [
  [4, 2],   // pulgar
  [8, 6],   // índice
  [12, 10], // medio
  [16, 14], // anular
  [20, 18], // meñique
];

let handLandmarker = null;
let stream = null;
let running = false;
let rafId = null;
let lastVideoTime = -1;

// ---------- Conteo de dedos ----------

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function countFingers(lm) {
  const wrist = lm[0];
  let count = 0;
  for (const [tip, ref] of FINGERS) {
    if (dist(lm[tip], wrist) > dist(lm[ref], wrist) * 1.1) count++;
  }
  return count;
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
      const fingers = countFingers(lm);
      totalFingers += fingers;

      // Huesos
      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = Math.max(2, 3 * scale);
      for (const [a, b] of HAND_CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(lm[a].x * canvas.width, lm[a].y * canvas.height);
        ctx.lineTo(lm[b].x * canvas.width, lm[b].y * canvas.height);
        ctx.stroke();
      }

      // Puntos
      for (let i = 0; i < lm.length; i++) {
        const p = lm[i];
        ctx.beginPath();
        ctx.arc(p.x * canvas.width, p.y * canvas.height, Math.max(3, 5 * scale), 0, Math.PI * 2);
        ctx.fillStyle = [4, 8, 12, 16, 20].includes(i) ? "#f472b6" : "#fbbf24";
        ctx.fill();
      }

      // Número grande sobre la palma (landmark 9 = MCP del medio)
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
    }

    // Estado inferior
    if (hands.length === 0) {
      setStatus("Acerca tu mano a la cámara…");
    } else if (hands.length === 1) {
      setStatus(`Manos: 1 · Dedos extendidos: ${totalFingers}`);
    } else {
      const detalle = hands.map(lm => countFingers(lm)).join(" + ");
      setStatus(`Manos: ${hands.length} · Dedos: ${totalFingers} (${detalle})`);
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
