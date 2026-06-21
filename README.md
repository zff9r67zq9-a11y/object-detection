# Live Object Detection

Single-page, in-browser real-time object detection. Vanilla JS + TensorFlow.js
+ COCO-SSD, loaded from CDN. No build step, no backend, nothing uploaded —
inference runs on-device.

## Files
- `index.html` — markup + CDN script tags
- `style.css` — responsive dark UI
- `script.js` — camera, detection loop, overlay, speech

## What it detects
Two models run together:

- **COCO-SSD** (`mobilenet_v2` base) draws **bounding boxes** for a fixed set of
  **~80 classes** (person, car, dog, cup, laptop, cell phone, chair…). It cannot
  box anything outside that list.
- **MobileNet** classifier gives a **whole-frame "best guess"** drawn from
  **~1000 ImageNet classes** — this covers many things COCO can't box (e.g. a
  lighter), shown in the bottom overlay. Because it classifies the whole frame,
  it labels *what's there* but not *where* it is (no box).

So if COCO doesn't know an object, the best-guess line often still names it —
but it's a guess over the frame, not a precise detection.

## Run locally
The camera (`getUserMedia`) only works on `localhost` or HTTPS. From this folder:

```bash
python3 -m http.server 8000
```

Then open **http://localhost:8000** and tap **Start camera**.

`localhost` is treated as a secure context, so plain HTTP is fine on the same
machine. (Any static server works: `npx serve`, `php -S localhost:8000`, etc.)

## Run on your iPhone over the network
Phones hitting your computer by LAN IP (`http://192.168.x.x`) are **not** a
secure context, so the camera will be blocked — you need real HTTPS. The
easiest path is a tunnel or a static host. Run the local server above, then in
another terminal start a tunnel, e.g. `npx localtunnel --port 8000` or
`cloudflared tunnel --url http://localhost:8000` (or `ngrok http 8000`). Open
the resulting `https://…` URL in iPhone Safari and allow camera access. To
deploy permanently instead, drop these three files on any static HTTPS host —
Netlify, Vercel, GitHub Pages, or Cloudflare Pages — by dragging the folder
into their dashboard or running their CLI; they all serve over HTTPS by default,
so the camera works from any device on any network.

## Controls
- **Start / Stop** — camera only starts on an explicit tap; never auto-starts.
- **Confidence threshold** — live slider (default 60%); boxes below it are hidden.
- **Voice** — speaks each object class aloud the first time it appears in view
  (not repeated every frame). Toggle off with the voice button.

## Notes
- Rear camera (`facingMode: "environment"`) is preferred, with automatic
  fallback to the front camera.
- Detection is throttled to ~15 Hz and skips frames while a pass is in flight,
  so it stays responsive on phones.
- The overlay canvas is scaled to the displayed video box and accounts for
  `object-fit: cover` center-cropping, so boxes stay aligned across viewport
  and resolution changes.
