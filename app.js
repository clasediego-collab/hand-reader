import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const select = document.getElementById("camera-select");
const statusEl = document.getElementById("status");
const scoreEl = document.getElementById("score");
const gestureLabelEl = document.getElementById("gesture-label");
const hint = document.getElementById("hint");
const introImage = document.getElementById("intro-image");
const gestureImage = document.getElementById("gesture-image");
const countdownEl = document.getElementById("countdown");
const resultBadge = document.getElementById("result-badge");

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

const state = {
  handLandmarker: null,
  stream: null,
  rafId: null,
  running: false,
  ready: false,
  lastVideoTime: -1,
  playerScore: 0,
  appScore: 0,
  roundLocked: false,
};

const choices = ["rock", "paper", "scissors"];

function setStatus(message) {
  statusEl.textContent = message;
}

function setScore() {
  scoreEl.textContent = `Tú ${state.playerScore} · App ${state.appScore}`;
}

function setGestureLabel(label) {
  gestureLabelEl.textContent = label;
}

function setResultBadge(label, type = "neutral") {
  if (!resultBadge) return;

  resultBadge.textContent = label;
  resultBadge.classList.remove("success", "danger", "warning", "hidden");

  if (type === "success") resultBadge.classList.add("success");
  if (type === "danger") resultBadge.classList.add("danger");
  if (type === "warning") resultBadge.classList.add("warning");

  resultBadge.classList.remove("hidden");
}

function hideResultBadge() {
  if (resultBadge) resultBadge.classList.add("hidden");
}

function hideIntro() {
  if (introImage) introImage.classList.add("hidden");
}

function showIntro() {
  if (introImage) introImage.classList.remove("hidden");
}

function showCountdown(value) {
  if (!countdownEl) return;
  countdownEl.textContent = String(value);
  countdownEl.classList.remove("hidden");
}

function hideCountdown() {
  if (!countdownEl) return;
  countdownEl.classList.add("hidden");
}

function hideGesture() {
  if (gestureImage) {
    gestureImage.classList.add("hidden");
    gestureImage.src = "";
  }
}

function buildGestureSvg(kind) {
  const svgMap = {
    paper: `
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 220">
        <rect x="42" y="28" width="138" height="160" rx="16" fill="#f8fafc"/>
        <path d="M68 72h84M68 102h84M68 132h72" stroke="#94a3b8" stroke-width="7" stroke-linecap="round"/>
        <path d="M120 52l28 28v74H64V52z" fill="none" stroke="#dbeafe" stroke-width="4"/>
      </svg>
    `,
    rock: `
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 220">
        <path d="M58 38c24-20 72-20 104 4 30 22 34 64 28 94-8 36-36 58-74 60-48 4-84-20-90-62-5-28 5-73 32-96z" fill="#e2e8f0"/>
        <circle cx="84" cy="88" r="8" fill="#64748b"/>
        <circle cx="128" cy="96" r="8" fill="#64748b"/>
        <path d="M84 132c16 16 36 18 52 0" stroke="#64748b" stroke-width="8" fill="none" stroke-linecap="round"/>
      </svg>
    `,
    scissors: `
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 220">
        <g stroke="#f8fafc" stroke-width="8" stroke-linecap="round" stroke-linejoin="round" fill="none">
          <circle cx="68" cy="62" r="28"/>
          <circle cx="68" cy="150" r="28"/>
          <path d="M96 82L168 112M96 130L168 92"/>
          <path d="M170 54L148 110M170 160L148 110"/>
        </g>
      </svg>
    `,
  };

  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgMap[kind])}`;
}

function setGestureImage(kind) {
  if (!gestureImage) return;

  if (!kind) {
    hideGesture();
    return;
  }

  gestureImage.src = buildGestureSvg(kind);
  gestureImage.classList.remove("hidden");
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function angleDeg(a, b, c) {
  const v1x = a.x - b.x;
  const v1y = a.y - b.y;
  const v2x = c.x - b.x;
  const v2y = c.y - b.y;

  const dot = v1x * v2x + v1y * v2y;
  const m1 = Math.hypot(v1x, v1y);
  const m2 = Math.hypot(v2x, v2y);

  if (m1 === 0 || m2 === 0) return 0;

  const cos = Math.max(-1, Math.min(1, dot / (m1 * m2)));
  return Math.acos(cos) * 180 / Math.PI;
}

function isThumbExtended(lm) {
  const palmSize = dist(lm[0], lm[9]);
  const tipToIndex = dist(lm[4], lm[5]) / palmSize;
  const mcpAngle = angleDeg(lm[1], lm[2], lm[4]);
  const ipAngle = angleDeg(lm[2], lm[3], lm[4]);

  return mcpAngle > 145 && ipAngle > 145 && tipToIndex > 0.5;
}

function countFingers(lm) {
  const wrist = lm[0];
  let count = 0;

  const fingerTuples = [
    [8, 6],
    [12, 10],
    [16, 14],
    [20, 18],
  ];

  for (const [tip, ref] of fingerTuples) {
    if (dist(lm[tip], wrist) > dist(lm[ref], wrist) * 1.1) count++;
  }

  if (isThumbExtended(lm)) count++;

  return count;
}

function classifyGesture(lm) {
  const fingers = countFingers(lm);

  if (fingers >= 5) return "paper";
  if (fingers === 0) return "rock";
  if (fingers === 2) return "scissors";

  return null;
}

function randomChoice() {
  return choices[Math.floor(Math.random() * choices.length)];
}

function compareChoices(player, app) {
  if (player === app) return "Empate";

  if (
    (player === "rock" && app === "scissors") ||
    (player === "scissors" && app === "paper") ||
    (player === "paper" && app === "rock")
  ) {
    return "Ganaste";
  }

  return "Perdiste";
}

function resolveRound(playerChoice) {
  if (state.roundLocked) return;

  state.roundLocked = true;

  const appChoice = randomChoice();
  const result = compareChoices(playerChoice, appChoice);

  if (result === "Ganaste") {
    state.playerScore += 1;
    setResultBadge("Ganaste", "success");
  } else if (result === "Perdiste") {
    state.appScore += 1;
    setResultBadge("Perdiste", "danger");
  } else {
    setResultBadge("Empate", "warning");
  }

  setScore();
  setGestureLabel(`${playerChoice} vs ${appChoice}`);
  setStatus(`App eligió ${appChoice}. ${result}.`);

  setTimeout(() => {
    hideResultBadge();
    setGestureLabel("Listo");
    setStatus("Haz tu gesto");
    state.roundLocked = false;
  }, 1800);
}

function stopStream() {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => {
      try { track.stop(); } catch {}
    });
    state.stream = null;
  }
}

function stopCamera() {
  state.running = false;

  if (state.rafId) {
    cancelAnimationFrame(state.rafId);
    state.rafId = null;
  }

  stopStream();

  if (video) video.srcObject = null;

  if (canvas) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  if (hint) hint.classList.remove("hidden");
}

async function loadModel() {
  if (state.handLandmarker) return;

  setStatus("Cargando modelo...");
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);

  state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 2,
  });

  state.ready = true;
  setStatus("Modelo listo.");
}

async function refreshCameraList() {
  try {
    const temp = await navigator.mediaDevices.getUserMedia({ video: true });
    temp.getTracks().forEach((track) => track.stop());
  } catch {}

  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter((d) => d.kind === "videoinput");

  if (!select) return;
  select.innerHTML = "";

  if (cameras.length === 0) {
    select.innerHTML = "<option value=''>Sin cámaras</option>";
    return;
  }

  cameras.forEach((cam, index) => {
    const option = document.createElement("option");
    option.value = cam.deviceId;
    option.textContent = cam.label || `Cámara ${index + 1}`;
    select.appendChild(option);
  });
}

function startIntroCountdown() {
  const values = [3, 2, 1];
  let index = 0;

  showIntro();
  showCountdown(values[index]);
  setStatus("Preparando partida...");

  const interval = setInterval(() => {
    index++;

    if (index < values.length) {
      showCountdown(values[index]);
      return;
    }

    clearInterval(interval);
    hideCountdown();
    hideIntro();

    startCamera(select?.value || undefined)
      .then(() => {
        setStatus("Haz tu gesto");
        setGestureLabel("Listo");
      })
      .catch((error) => {
        setStatus(`Error: ${error.message}`);
        console.error(error);
      });
  }, 1000);
}

async function startCamera(deviceId = null) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Tu navegador no soporta acceso a cámara.");
  }

  stopCamera();

  const constraints = {
    audio: false,
    video: deviceId
      ? { deviceId: { exact: deviceId } }
      : { facingMode: "user" },
  };

  state.stream = await navigator.mediaDevices.getUserMedia(constraints);

  video.srcObject = state.stream;
  await video.play();

  const track = state.stream.getVideoTracks()[0];
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;

  canvas.width = w;
  canvas.height = h;

  state.running = true;
  state.lastVideoTime = -1;

  if (hint) hint.classList.add("hidden");

  setStatus(`Cámara activa: ${track?.label || "desconocida"}`);
  state.rafId = requestAnimationFrame(detectLoop);
}

function detectLoop() {
  if (!state.running || !state.handLandmarker) return;

  if (video.readyState >= 2 && video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const results = state.handLandmarker.detectForVideo(video, performance.now());
    const hands = results.landmarks || [];

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (hands.length === 0) {
      hideGesture();
      setGestureLabel("Sin mano");
      return;
    }

    const lm = hands[0];
    const gesture = classifyGesture(lm);

    if (gesture) {
      setGestureImage(gesture);
      setGestureLabel(
        gesture === "paper" ? "Papel" :
        gesture === "rock" ? "Piedra" :
        gesture === "scissors" ? "Tijera" : "Gesto"
      );

      if (!state.roundLocked) {
        resolveRound(gesture);
      }
    } else {
      hideGesture();
      setGestureLabel("No válido");
    }
  }

  state.rafId = requestAnimationFrame(detectLoop);
}

async function initApp() {
  try {
    if (!navigator.mediaDevices) {
      throw new Error("Este navegador no soporta acceso a la cámara.");
    }

    await refreshCameraList();
    await loadModel();
    setScore();
    showIntro();
    startIntroCountdown();
  } catch (error) {
    setStatus(`Error inicial: ${error.message}`);
    console.error(error);
  }
}

select.addEventListener("change", async () => {
  if (state.running) {
    try {
      await startCamera(select.value || undefined);
    } catch (error) {
      setStatus(`Error: ${error.message}`);
      console.error(error);
    }
  }
});

initApp();
