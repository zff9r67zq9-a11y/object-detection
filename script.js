/* Live object detection — TensorFlow.js + COCO-SSD, all in-browser.
   COCO-SSD recognizes a fixed set of ~80 classes; it cannot detect anything
   outside that set. */

(() => {
  "use strict";

  // ---------- Elements ----------
  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const ctx = overlay.getContext("2d");

  const screenEl = document.getElementById("screen");
  const screenMsg = document.getElementById("screenMsg");
  const screenSub = document.getElementById("screenSub");
  const startBtn = document.getElementById("startBtn");

  const hudTop = document.getElementById("hudTop");
  const fpsPill = document.getElementById("fpsPill");
  const hudBottom = document.getElementById("hudBottom");
  const guessVal = document.getElementById("guessVal");

  const controls = document.getElementById("controls");
  const thresholdInput = document.getElementById("threshold");
  const thresholdVal = document.getElementById("thresholdVal");
  const muteBtn = document.getElementById("muteBtn");
  const ignorePersonBtn = document.getElementById("ignorePersonBtn");
  const stopBtn = document.getElementById("stopBtn");
  const chips = document.getElementById("chips");

  const customUrl = document.getElementById("customUrl");
  const loadModelBtn = document.getElementById("loadModelBtn");
  const customStatus = document.getElementById("customStatus");
  const customLine = document.getElementById("customLine");
  const customVal = document.getElementById("customVal");

  // ---------- State ----------
  let model = null;        // COCO-SSD detector (bounding boxes, 80 classes)
  let classifier = null;   // MobileNet classifier (whole-frame guess, ~1000 classes)
  let customModel = null;  // optional Teachable Machine model (e.g. iPhone-model ID)
  let modelPromise = null; // in-flight detector load (started on page open)
  let classifierPromise = null; // in-flight classifier load (background)
  let stream = null;
  let running = false;
  let detecting = false;          // true while a forward pass is in flight
  let rafId = null;
  let threshold = 0.6;
  let voiceOn = true;
  let ignorePerson = false;  // when true, drop all "person" detections

  let lastDetectTime = 0;
  const MIN_INTERVAL_MS = 1000 / 15; // cap detection at ~15 Hz; skip frames otherwise

  // For "announce only when a new class appears": classes visible last frame.
  let spokenClasses = new Set();
  // Last best-guess class we announced, so we don't repeat it every frame.
  let lastGuessSpoken = "";
  const GUESS_MIN = 0.18; // min classifier confidence before we trust/announce it
  // Same idea for the custom model's top label.
  let lastCustomSpoken = "";
  const CUSTOM_MIN = 0.6; // custom model needs decent confidence before announcing
  // FPS smoothing
  let fpsEMA = 0;

  // A stable color per class label so boxes don't flicker between colors.
  const colorCache = new Map();
  function colorFor(label) {
    if (!colorCache.has(label)) {
      let h = 0;
      for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) % 360;
      colorCache.set(label, `hsl(${h}, 85%, 60%)`);
    }
    return colorCache.get(label);
  }

  // ImageNet labels look like "lighter, light, igniter, ignitor" — keep the
  // first, most common synonym.
  function cleanLabel(name) {
    return name.split(",")[0].trim();
  }

  // ---------- UI helpers ----------
  function showScreen({ msg, sub, button, spinner, error }) {
    screenEl.hidden = false;
    screenEl.classList.toggle("error", !!error);
    screenMsg.innerHTML = msg;
    screenSub.textContent = sub || "";
    screenSub.style.display = sub ? "" : "none";

    if (spinner) {
      startBtn.hidden = true;
      if (!document.getElementById("__spinner")) {
        const s = document.createElement("div");
        s.className = "spinner";
        s.id = "__spinner";
        startBtn.insertAdjacentElement("afterend", s);
      }
    } else {
      const s = document.getElementById("__spinner");
      if (s) s.remove();
      if (button) {
        startBtn.hidden = false;
        startBtn.disabled = false;
        startBtn.textContent = button;
      } else {
        startBtn.hidden = true;
      }
    }
  }

  function hideScreen() {
    screenEl.hidden = true;
    screenEl.classList.remove("error");
  }

  // ---------- Camera ----------
  async function getCameraStream() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("UNSUPPORTED");
    }
    // Prefer the rear camera; fall back to the front camera if unavailable.
    const tryConstraints = [
      { video: { facingMode: { exact: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
      { video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
      { video: { facingMode: "user" }, audio: false },
      { video: true, audio: false },
    ];
    let lastErr;
    for (const c of tryConstraints) {
      try {
        return await navigator.mediaDevices.getUserMedia(c);
      } catch (err) {
        lastErr = err;
        // Permission denied is terminal — don't keep trying.
        if (err && (err.name === "NotAllowedError" || err.name === "SecurityError")) throw err;
      }
    }
    throw lastErr || new Error("NO_CAMERA");
  }

  // Match the overlay canvas pixel buffer to the *displayed* video box, so we
  // draw in CSS pixels and never worry about the native stream resolution here.
  function sizeCanvasToDisplay() {
    const dpr = window.devicePixelRatio || 1;
    const w = video.clientWidth;
    const h = video.clientHeight;
    overlay.width = Math.round(w * dpr);
    overlay.height = Math.round(h * dpr);
    overlay.style.width = w + "px";
    overlay.style.height = h + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // The video uses object-fit: cover, so the native frame is scaled by the
  // larger ratio and center-cropped. Convert model coords (native pixels) into
  // displayed CSS pixels using the same transform.
  function makeMapper() {
    const vw = video.videoWidth || 1;
    const vh = video.videoHeight || 1;
    const dw = video.clientWidth;
    const dh = video.clientHeight;
    const scale = Math.max(dw / vw, dh / vh);   // cover
    const offsetX = (dw - vw * scale) / 2;
    const offsetY = (dh - vh * scale) / 2;
    return (x, y, w, h) => ({
      x: x * scale + offsetX,
      y: y * scale + offsetY,
      w: w * scale,
      h: h * scale,
    });
  }

  // ---------- Drawing ----------
  function drawDetections(predictions) {
    const map = makeMapper();
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.lineWidth = 3;
    ctx.font = "600 15px -apple-system, system-ui, sans-serif";
    ctx.textBaseline = "top";

    for (const p of predictions) {
      const color = colorFor(p.class);
      const [bx, by, bw, bh] = p.bbox;
      const { x, y, w, h } = map(bx, by, bw, bh);

      // Box
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);

      // Label background
      const label = `${p.class} ${Math.round(p.score * 100)}%`;
      const padX = 7, padY = 4;
      const tw = ctx.measureText(label).width;
      const th = 20;
      let ly = y - th - 2;
      if (ly < 0) ly = y + 2; // keep label inside frame at the top edge
      ctx.fillStyle = color;
      ctx.fillRect(x - 1.5, ly, tw + padX * 2, th);

      // Label text
      ctx.fillStyle = "#07120a";
      ctx.fillText(label, x - 1.5 + padX, ly + padY);
    }
  }

  // ---------- Speech ----------
  function speak(text) {
    if (!voiceOn || !("speechSynthesis" in window)) return;
    try {
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 1.05;
      u.pitch = 1;
      u.volume = 1;
      window.speechSynthesis.speak(u);
    } catch (_) { /* ignore */ }
  }

  // ---------- Detected chips + announcements ----------
  function updateDetectedUI(predictions) {
    // Count by class
    const counts = new Map();
    for (const p of predictions) counts.set(p.class, (counts.get(p.class) || 0) + 1);

    // Render chips
    if (counts.size === 0) {
      chips.innerHTML = '<span class="chip-empty">Nothing above threshold</span>';
    } else {
      chips.innerHTML = "";
      for (const [cls, n] of [...counts.entries()].sort((a, b) => b[1] - a[1])) {
        const el = document.createElement("span");
        el.className = "chip";
        el.style.borderColor = colorFor(cls);
        el.style.color = colorFor(cls);
        el.innerHTML = n > 1
          ? `${cls} <span class="count">×${n}</span>`
          : cls;
        chips.appendChild(el);
      }
    }

    // Announce only classes that are newly present this frame.
    const current = new Set(counts.keys());
    const fresh = [...current].filter((c) => !spokenClasses.has(c));
    if (fresh.length) speak(fresh.join(", "));
    spokenClasses = current;
  }

  // ---------- Whole-frame "best guess" (classifier) ----------
  function updateGuessUI(guesses, detected) {
    // null = classifier still downloading in the background.
    if (guesses === null) {
      guessVal.textContent = "loading…";
      guessVal.classList.add("low");
      return;
    }
    const top = guesses && guesses[0];
    if (!top || top.probability < GUESS_MIN) {
      guessVal.textContent = "—";
      guessVal.classList.add("low");
      lastGuessSpoken = "";
      return;
    }

    const label = cleanLabel(top.className);
    guessVal.textContent = `${label} ${Math.round(top.probability * 100)}%`;
    guessVal.classList.remove("low");

    // Announce only when the guess changes — and skip it if COCO already boxed
    // something with that name (avoid saying the same object twice).
    const alsoBoxed = detected.some((p) => p.class === label);
    if (label !== lastGuessSpoken && !alsoBoxed) {
      speak(label);
      lastGuessSpoken = label;
    }
  }

  // ---------- Custom model (Teachable Machine) ----------
  function setCustomStatus(msg, kind) {
    customStatus.textContent = msg;
    customStatus.classList.remove("ok", "err");
    if (kind) customStatus.classList.add(kind);
  }

  const TM_LIB = "https://cdn.jsdelivr.net/npm/@teachablemachine/image@0.8/dist/teachablemachine-image.min.js";
  function ensureScript(src) {
    return new Promise((resolve, reject) => {
      if ([...document.scripts].some((s) => s.src === src)) return resolve();
      const s = document.createElement("script");
      s.src = src;
      s.onload = resolve;
      s.onerror = () => reject(new Error("script load failed"));
      document.head.appendChild(s);
    });
  }

  async function loadCustomModel() {
    let base = customUrl.value.trim();
    if (!base) { setCustomStatus("Paste a Teachable Machine model URL first.", "err"); return; }
    // Teachable Machine shares a folder URL; the files live under it.
    if (!base.endsWith("/")) base += "/";
    loadModelBtn.disabled = true;
    setCustomStatus("Loading custom model…");
    try {
      // Pull the Teachable Machine library only now (kept out of page load).
      if (typeof tmImage === "undefined") await ensureScript(TM_LIB);
      const m = await tmImage.load(base + "model.json", base + "metadata.json");
      customModel = m;
      lastCustomSpoken = "";
      customLine.hidden = false;
      setCustomStatus(`Loaded — ${m.getTotalClasses()} classes. Point the camera at a phone.`, "ok");
    } catch (err) {
      console.error(err);
      customModel = null;
      customLine.hidden = true;
      setCustomStatus("Couldn't load that model. Use the model's shared URL (ending in its folder).", "err");
    } finally {
      loadModelBtn.disabled = false;
    }
  }

  function updateCustomUI(preds) {
    if (!customModel || !preds || !preds.length) return;
    // Teachable Machine returns every class unsorted — pick the highest.
    let top = preds[0];
    for (const p of preds) if (p.probability > top.probability) top = p;

    const pct = Math.round(top.probability * 100);
    customVal.textContent = `${top.className} ${pct}%`;
    customVal.classList.toggle("low", top.probability < CUSTOM_MIN);

    if (top.probability >= CUSTOM_MIN && top.className !== lastCustomSpoken) {
      speak(top.className);
      lastCustomSpoken = top.className;
    }
  }

  // ---------- Detection loop ----------
  async function loop() {
    if (!running) return;

    const now = performance.now();
    const due = now - lastDetectTime >= MIN_INTERVAL_MS;

    // Skip this frame if a detection is still running or we're ahead of pace.
    if (!detecting && due && video.readyState >= 2) {
      detecting = true;
      lastDetectTime = now;
      try {
        // Detector always runs. Classifier and custom model join in only once
        // they've finished loading in the background.
        const hasClf = !!classifier;
        const hasCustom = !!customModel;
        const tasks = [model.detect(video, 20)];
        if (hasClf) tasks.push(classifier.classify(video, 3));
        if (hasCustom) tasks.push(customModel.predict(video));
        const results = await Promise.all(tasks);

        const raw = results[0];
        const guesses = hasClf ? results[1] : null;
        const customPreds = hasCustom ? results[hasClf ? 2 : 1] : null;

        const filtered = raw.filter(
          (p) => p.score >= threshold && !(ignorePerson && p.class === "person")
        );
        drawDetections(filtered);
        updateDetectedUI(filtered);
        updateGuessUI(guesses, filtered);
        updateCustomUI(customPreds);

        // FPS (exponential moving average of detection rate)
        const dt = performance.now() - now;
        const inst = 1000 / Math.max(dt, 1);
        fpsEMA = fpsEMA ? fpsEMA * 0.85 + inst * 0.15 : inst;
        fpsPill.textContent = `${fpsEMA.toFixed(0)} fps`;
      } catch (err) {
        console.error("detect failed", err);
      } finally {
        detecting = false;
      }
    }

    rafId = requestAnimationFrame(loop);
  }

  // ---------- Model preloading ----------
  // Kick off downloads the moment the page opens so the wait overlaps with the
  // user reading the intro. Each promise also warms up its WebGL shaders on a
  // blank canvas, so the first real frame isn't janky.
  function preloadModels() {
    if (!modelPromise) {
      modelPromise = (async () => {
        await tf.ready();
        // Full 'mobilenet_v2' base = more reliable boxes than the lite base.
        const m = await cocoSsd.load({ base: "mobilenet_v2" });
        try { await m.detect(warmCanvas()); } catch (_) {}
        model = m;
        return m;
      })();
      modelPromise.catch(() => {}); // handled where awaited; avoid unhandled rejection
    }
    if (!classifierPromise) {
      classifierPromise = (async () => {
        await tf.ready();
        const c = await mobilenet.load({ version: 2, alpha: 1.0 });
        try { await c.classify(warmCanvas()); } catch (_) {}
        classifier = c; // loop starts using it automatically once set
        return c;
      })();
      classifierPromise.catch(() => {});
    }
  }

  function warmCanvas() {
    const c = document.createElement("canvas");
    c.width = c.height = 128;
    return c;
  }

  // ---------- Start / stop ----------
  async function start() {
    startBtn.disabled = true;

    // 1) The detector is the only thing the live feed needs. It's usually
    // already loaded (we preload on page open), so this resolves instantly;
    // only show a spinner if it isn't ready yet. The classifier keeps loading
    // in the background and wires itself in when done.
    preloadModels();
    if (!model) {
      showScreen({ msg: "Warming up detector…", sub: "Almost there — first load only.", spinner: true });
      try {
        model = await modelPromise;
      } catch (err) {
        console.error(err);
        modelPromise = null; // allow a fresh attempt on retry
        showScreen({
          msg: "Couldn't load the detector.",
          sub: "Check your internet connection and try again.",
          button: "Retry",
          error: true,
        });
        startBtn.disabled = false;
        return;
      }
    }

    // 2) Start the camera.
    showScreen({ msg: "Requesting camera…", sub: "Allow access when prompted.", spinner: true });
    try {
      stream = await getCameraStream();
    } catch (err) {
      console.error(err);
      let msg, sub;
      if (err.message === "UNSUPPORTED") {
        msg = "Camera not supported here.";
        sub = "Your browser doesn't expose getUserMedia. Try Safari or Chrome over HTTPS.";
      } else if (err.name === "NotAllowedError" || err.name === "SecurityError") {
        msg = "Camera permission denied.";
        sub = "Enable camera access for this site in your browser settings, then retry.";
      } else if (err.name === "NotFoundError" || err.name === "OverconstrainedError") {
        msg = "No usable camera found.";
        sub = "This device doesn't seem to have an available camera.";
      } else {
        msg = "Couldn't start the camera.";
        sub = (err && err.message) || "Unknown error.";
      }
      showScreen({ msg, sub, button: "Retry", error: true });
      startBtn.disabled = false;
      return;
    }

    video.srcObject = stream;
    await video.play().catch(() => {});

    // Wait for real dimensions before sizing the canvas.
    if (!video.videoWidth) {
      await new Promise((res) => video.addEventListener("loadedmetadata", res, { once: true }));
    }

    sizeCanvasToDisplay();
    hideScreen();
    hudTop.hidden = false;
    hudBottom.hidden = false;
    controls.hidden = false;

    customLine.hidden = !customModel; // show the custom row only if one is loaded

    running = true;
    detecting = false;
    spokenClasses = new Set();
    lastGuessSpoken = "";
    lastCustomSpoken = "";
    lastDetectTime = 0;
    loop();
  }

  function stop() {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    video.srcObject = null;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    spokenClasses = new Set();
    lastGuessSpoken = "";
    lastCustomSpoken = "";

    hudTop.hidden = true;
    hudBottom.hidden = true;
    controls.hidden = true;
    chips.innerHTML = '<span class="chip-empty">Nothing detected yet</span>';
    showScreen({
      msg: "Camera stopped.",
      sub: "Tap to start detecting again.",
      button: "Start camera",
    });
  }

  // ---------- Events ----------
  startBtn.addEventListener("click", start);
  stopBtn.addEventListener("click", stop);

  thresholdInput.addEventListener("input", () => {
    threshold = thresholdInput.value / 100;
    thresholdVal.textContent = `${thresholdInput.value}%`;
  });

  muteBtn.addEventListener("click", () => {
    voiceOn = !voiceOn;
    muteBtn.setAttribute("aria-pressed", String(voiceOn));
    muteBtn.textContent = voiceOn ? "🔊 Voice on" : "🔇 Voice off";
    if (!voiceOn && "speechSynthesis" in window) window.speechSynthesis.cancel();
  });

  loadModelBtn.addEventListener("click", loadCustomModel);
  customUrl.addEventListener("keydown", (e) => { if (e.key === "Enter") loadCustomModel(); });

  ignorePersonBtn.addEventListener("click", () => {
    ignorePerson = !ignorePerson;
    ignorePersonBtn.setAttribute("aria-pressed", String(!ignorePerson));
    ignorePersonBtn.textContent = ignorePerson ? "🚫 People hidden" : "🧍 People shown";
    // Stop "person" lingering in the spoken-set so it re-announces if shown again.
    if (ignorePerson) spokenClasses.delete("person");
  });

  // Keep the overlay aligned when the viewport changes (rotation, resize).
  let resizeTimer;
  function onResize() {
    if (!running) return;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(sizeCanvasToDisplay, 100);
  }
  window.addEventListener("resize", onResize);
  window.addEventListener("orientationchange", onResize);

  // Initial idle screen.
  showScreen({
    msg: "Real-time object detection that runs entirely in your browser. Nothing is uploaded.",
    sub: "Uses your rear camera on phones. You'll be asked for permission.",
    button: "Start camera",
  });

  // Start downloading the models right away so they're ready by the time the
  // user taps Start. Runs in the background; failures are surfaced on Start.
  preloadModels();
})();
