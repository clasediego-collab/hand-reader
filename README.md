# 🖐️ Hand Reader

Detector de manos en tiempo real que corre **directamente en el navegador**, sin backend y sin subir video a ningún servidor.

Abre la web en cualquier dispositivo con cámara —PC, celular o tablet— y cada uno procesa su propio video localmente con **MediaPipe Hand Landmarker**.

## ✨ Características

- 🎥 Cada dispositivo usa su propia cámara.
- 🖐️ Detecta hasta 2 manos con 21 puntos (landmarks) en 3D.
- 📱 Funciona en PC, Android e iOS con el mismo código.
- 🔒 Privado: el video nunca sale del dispositivo.
- 🪞 Modo espejo activable.
- 🎛️ Selector de cámara (frontal, trasera, externas en PC).
- ⚡ Sin instalación: es una web estática.

## 🧰 Stack

- [MediaPipe Tasks Vision](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) — detección de manos.
- `getUserMedia` — acceso a la cámara del dispositivo.
- Canvas 2D — dibujo de los landmarks.
- HTML, CSS y JavaScript puro. Sin frameworks ni build.

## ▶️ Correr en local

La cámara requiere contexto seguro, así que hay que servir los archivos por HTTP (no abrir con `file://`).

```bash
python3 -m http.server 8000
