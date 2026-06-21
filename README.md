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

### Custom model (e.g. iPhone-model ID)
Neither built-in model can name a *specific* product (which iPhone, which car
trim, etc.) — those classes don't exist in COCO or ImageNet. To recognize your
own categories you can plug in a **Teachable Machine** image model:

1. Go to https://teachablemachine.withgoogle.com → **Image Project → Standard**.
2. Make a class per thing you want (e.g. `iPhone SE`, `iPhone 15 Pro`), and add
   lots of photos of each from different angles/lighting. **More, varied photos
   = better.**
3. **Train**, then **Export → TensorFlow.js → Upload (shareable link)**. Copy the
   URL it gives you (looks like `https://teachablemachine.withgoogle.com/models/XXXX/`).
4. Start the camera here, paste that URL into **Custom model**, tap **Load**. Its
   top label shows in the bottom overlay (blue) and is spoken when confident.

**Reality check:** lookalike phones (iPhone 14 vs 15 vs 16) are extremely hard to
tell apart visually and will be unreliable. Coarse, visually-distinct splits
(home-button iPhone vs notch vs Dynamic Island; single vs triple camera) work far
better. Vision can't read the exact model the way Settings → About can.

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

## Loading speed
- Models **start downloading the moment the page opens** (in the background),
  so by the time you tap Start they're usually ready and it's near-instant.
- The live feed + boxes appear as soon as the **detector** is ready; the
  MobileNet classifier finishes loading in the background and the "best guess"
  line switches from `loading…` to a label when it's done — you don't wait on it.
- Both models warm up their GPU shaders during preload, so the first frame
  isn't janky. The Teachable Machine library is only fetched if you use it.
- Model files are served with long cache headers, so repeat visits load from
  the browser cache (no re-download) unless you hard-refresh.

## Notes
- Rear camera (`facingMode: "environment"`) is preferred, with automatic
  fallback to the front camera.
- Detection is throttled to ~15 Hz and skips frames while a pass is in flight,
  so it stays responsive on phones.
- The overlay canvas is scaled to the displayed video box and accounts for
  `object-fit: cover` center-cropping, so boxes stay aligned across viewport
  and resolution changes.
