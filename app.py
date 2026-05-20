"""
Pixel Transport Morph — Streamlit App
Luminance-sorted approximate optimal transport between two images.
Source pixel colors are preserved — only positions animate.
Canvas adapts to target image aspect ratio.
"""

import streamlit as st
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="Pixel Transport", layout="centered")

# ─── Python: image processing & transport mapping ───────────────────────────

def sample_pixels(img: Image.Image, count: int) -> np.ndarray:
    """
    Sample up to `count` non-transparent pixels from a PIL Image.
    Returns array of shape (N, 6): [x_norm, y_norm, r, g, b, luminance]
    Coordinates normalized to [0, 1] relative to the image's own dimensions.
    """
    img = img.convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Adaptive step: for very large images, increase stride
    step = max(2, int(np.sqrt(h * w / (count * 4))))
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    ys, xs = ys.ravel(), xs.ravel()
    pixels = arr[ys, xs]

    # Filter out near-transparent pixels
    mask = pixels[:, 3] > 20
    xs, ys, pixels = xs[mask], ys[mask], pixels[mask]

    # Reservoir sample if too many
    n = len(xs)
    if n > count:
        idx = np.random.choice(n, count, replace=False)
        xs, ys, pixels = xs[idx], ys[idx], pixels[idx]

    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    return np.column_stack([xs / w, ys / h, r, g, b, lum])


def build_particle_data(source_img: Image.Image, target_img: Image.Image, count: int) -> str:
    """
    Sample both images, sort by luminance, pair 1:1.
    Only source colors are kept — target supplies positions only.
    """
    src = sample_pixels(source_img, count)
    tgt = sample_pixels(target_img, count)

    # Sort both by luminance (column 5) for approximate optimal transport
    src = src[src[:, 5].argsort()]
    tgt = tgt[tgt[:, 5].argsort()]

    # Match counts
    n = min(len(src), len(tgt))
    src, tgt = src[:n], tgt[:n]

    # Random stagger delay per particle
    delays = np.random.uniform(0, 0.15, n)

    # Pack as flat arrays for compact JSON (much smaller than array-of-objects)
    # Each particle: sx, sy, r, g, b, tx, ty, delay
    flat = np.column_stack([
        src[:, 0], src[:, 1],           # source x, y (normalized)
        src[:, 2], src[:, 3], src[:, 4], # source r, g, b (kept forever)
        tgt[:, 0], tgt[:, 1],           # target x, y (normalized)
        delays,
    ])

    # Round for smaller JSON
    flat[:, :2] = np.round(flat[:, :2], 4)
    flat[:, 5:7] = np.round(flat[:, 5:7], 4)
    flat[:, 7] = np.round(flat[:, 7], 4)
    flat[:, 2:5] = np.round(flat[:, 2:5], 0)

    return json.dumps(flat.tolist())


# ─── JS: canvas animation component ────────────────────────────────────────

def render_animation_html(particle_json: str, duration_ms: int,
                          canvas_w: int, canvas_h: int) -> str:
    return f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0a0a0a;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    font-family: 'SF Mono', 'Fira Code', monospace;
    color: #ccc;
    gap: 14px;
    padding: 10px;
  }}
  canvas {{
    border: 1px solid #1a1a1a;
    border-radius: 2px;
    max-width: 100%;
    height: auto;
  }}
  .controls {{
    display: flex;
    gap: 12px;
    align-items: center;
  }}
  button {{
    background: #1a1a1a;
    color: #e0e0e0;
    border: 1px solid #333;
    padding: 8px 20px;
    font-family: inherit;
    font-size: 13px;
    cursor: pointer;
    letter-spacing: 0.05em;
    border-radius: 2px;
    transition: background 0.2s;
  }}
  button:hover {{ background: #2a2a2a; }}
  button:disabled {{ opacity: 0.3; cursor: default; }}
  .bar-track {{
    width: min({canvas_w}px, 90vw);
    height: 3px;
    background: #1a1a1a;
    border-radius: 1px;
    overflow: hidden;
  }}
  .bar-fill {{
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #e63946, #ff6b35);
  }}
  .label {{
    font-size: 11px;
    color: #555;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }}
</style>
</head>
<body>
  <canvas id="c" width="{canvas_w}" height="{canvas_h}"></canvas>
  <div class="bar-track"><div class="bar-fill" id="bar"></div></div>
  <div class="controls">
    <button id="playBtn" onclick="play()">Play</button>
    <button id="revBtn" onclick="reverse()" disabled>Reverse</button>
    <span class="label" id="stateLabel">ready</span>
  </div>

<script>
// Flat array: each sub-array is [sx, sy, r, g, b, tx, ty, delay]
const raw = {particle_json};
const N = raw.length;
const DURATION = {duration_ms};
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const bar = document.getElementById('bar');
const playBtn = document.getElementById('playBtn');
const revBtn = document.getElementById('revBtn');
const stateLabel = document.getElementById('stateLabel');
const W = canvas.width, H = canvas.height;

let animId = null;
let currentT = 0;

function ease(t) {{
  return t < 0.5 ? 4*t*t*t : 1 - Math.pow(-2*t+2, 3) / 2;
}}

function draw(t) {{
  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, W, H);
  for (let i = 0; i < N; i++) {{
    const p = raw[i];
    // p = [sx, sy, r, g, b, tx, ty, delay]
    const delay = p[7];
    const pt = Math.max(0, Math.min(1, (t - delay) / (1 - delay)));
    const e = ease(pt);
    const x = (p[0] + (p[5] - p[0]) * e) * W;
    const y = (p[1] + (p[6] - p[1]) * e) * H;
    // Color stays fixed — source colors only
    ctx.fillStyle = 'rgb(' + p[2] + ',' + p[3] + ',' + p[4] + ')';
    ctx.fillRect(x - 1, y - 1, 2.5, 2.5);
  }}
  bar.style.width = (t * 100) + '%';
  currentT = t;
}}

function play() {{
  if (animId) return;
  playBtn.disabled = true;
  revBtn.disabled = true;
  stateLabel.textContent = 'morphing...';
  const start = performance.now();
  const startT = currentT;
  function frame(now) {{
    const elapsed = now - start;
    const t = Math.min(1, startT + (1 - startT) * (elapsed / DURATION));
    draw(t);
    if (t < 1) {{
      animId = requestAnimationFrame(frame);
    }} else {{
      animId = null;
      playBtn.disabled = true;
      revBtn.disabled = false;
      stateLabel.textContent = 'target reached';
    }}
  }}
  animId = requestAnimationFrame(frame);
}}

function reverse() {{
  if (animId) return;
  playBtn.disabled = true;
  revBtn.disabled = true;
  stateLabel.textContent = 'reversing...';
  const start = performance.now();
  const startT = currentT;
  function frame(now) {{
    const elapsed = now - start;
    const t = Math.max(0, startT - startT * (elapsed / DURATION));
    draw(t);
    if (t > 0) {{
      animId = requestAnimationFrame(frame);
    }} else {{
      animId = null;
      playBtn.disabled = false;
      revBtn.disabled = true;
      stateLabel.textContent = 'source reached';
    }}
  }}
  animId = requestAnimationFrame(frame);
}}

draw(0);
</script>
</body>
</html>
"""


# ─── Streamlit UI ───────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .block-container { max-width: 720px; }
    h1 { letter-spacing: 0.05em; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Pixel Transport Morph")
st.caption(
    "Pixels from the source image travel to the target image's positions — "
    "but keep their original colors. The target's structure emerges from the source's palette."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Source image** (provides pixels & colors)")
    source_file = st.file_uploader(
        "Upload source", type=["png", "jpg", "jpeg", "webp"], key="src",
        label_visibility="collapsed",
    )

with col2:
    st.markdown("**Target image** (provides structure)")
    target_file = st.file_uploader(
        "Upload target", type=["png", "jpg", "jpeg", "webp"], key="tgt",
        label_visibility="collapsed",
    )

with st.sidebar:
    st.header("Parameters")
    particle_count = st.slider("Particle count", 5000, 30000, 15000, step=1000)
    duration_ms = st.slider("Animation duration (ms)", 1000, 8000, 3000, step=500)
    max_dim = st.slider("Max canvas dimension (px)", 300, 800, 550, step=50)


# ─── Default images ─────────────────────────────────────────────────────────

def make_noise_image(w=400, h=300):
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)

def make_gradient_circle(w=400, h=300):
    img = Image.new("RGB", (w, h), (10, 10, 10))
    arr = np.array(img, dtype=np.float64)
    cx, cy = w / 2, h / 2
    radius = min(w, h) * 0.4
    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    mask = dist < radius
    t = dist[mask] / radius
    arr[mask, 0] = 230 * (1 - t) + 29 * t
    arr[mask, 1] = 57 * (1 - t) + 123 * t
    arr[mask, 2] = 70 * (1 - t) + 157 * t
    return Image.fromarray(arr.astype(np.uint8))


source_img = Image.open(source_file) if source_file else make_noise_image()
target_img = Image.open(target_file) if target_file else make_gradient_circle()

# Show thumbnails
col1, col2 = st.columns(2)
with col1:
    st.image(source_img, caption="Source", use_container_width=True)
with col2:
    st.image(target_img, caption="Target", use_container_width=True)


# ─── Compute canvas dimensions from target aspect ratio ─────────────────────

tw, th = target_img.size
aspect = tw / th
if tw >= th:
    canvas_w = max_dim
    canvas_h = int(max_dim / aspect)
else:
    canvas_h = max_dim
    canvas_w = int(max_dim * aspect)

# ─── Build particles and render ─────────────────────────────────────────────

with st.spinner("Computing transport mapping..."):
    particle_json = build_particle_data(source_img, target_img, particle_count)

st.components.v1.html(
    render_animation_html(particle_json, duration_ms, canvas_w, canvas_h),
    height=canvas_h + 80,
)
