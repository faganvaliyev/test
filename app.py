"""
Pixel Transport Morph — Streamlit App
Luminance-sorted approximate optimal transport between two images,
rendered as a live 60fps canvas animation in the browser.
"""

import streamlit as st
import numpy as np
from PIL import Image
import json
import io

st.set_page_config(page_title="Pixel Transport", layout="centered")

# ─── Python: image processing & transport mapping ───────────────────────────

def sample_pixels(img: Image.Image, count: int) -> np.ndarray:
    """
    Sample up to `count` non-transparent pixels from a PIL Image.
    Returns array of shape (N, 6): [x_norm, y_norm, r, g, b, luminance]
    """
    img = img.convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Subsample grid for speed
    ys, xs = np.mgrid[0:h:2, 0:w:2]
    ys, xs = ys.ravel(), xs.ravel()
    pixels = arr[ys, xs]  # shape (N, 4)

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

    result = np.column_stack([
        xs / w,   # normalized x
        ys / h,   # normalized y
        r, g, b,
        lum,
    ])
    return result


def build_particle_data(source_img: Image.Image, target_img: Image.Image, count: int) -> str:
    """
    Sample both images, sort by luminance, pair 1:1, return JSON string
    with source and target positions/colors for each particle.
    """
    src = sample_pixels(source_img, count)
    tgt = sample_pixels(target_img, count)

    # Sort both by luminance (column 5)
    src = src[src[:, 5].argsort()]
    tgt = tgt[tgt[:, 5].argsort()]

    # Match counts
    n = min(len(src), len(tgt))
    src, tgt = src[:n], tgt[:n]

    # Random stagger delay per particle
    delays = np.random.uniform(0, 0.15, n)

    particles = []
    for i in range(n):
        particles.append({
            "sx": round(float(src[i, 0]), 4),
            "sy": round(float(src[i, 1]), 4),
            "sr": int(src[i, 2]),
            "sg": int(src[i, 3]),
            "sb": int(src[i, 4]),
            "tx": round(float(tgt[i, 0]), 4),
            "ty": round(float(tgt[i, 1]), 4),
            "tr": int(tgt[i, 2]),
            "tg": int(tgt[i, 3]),
            "tb": int(tgt[i, 4]),
            "d": round(float(delays[i]), 4),
        })

    return json.dumps(particles)


# ─── JS: canvas animation component ────────────────────────────────────────

def render_animation_html(particle_json: str, duration_ms: int, canvas_size: int) -> str:
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
    height: 100vh;
    font-family: 'SF Mono', 'Fira Code', monospace;
    color: #ccc;
    gap: 16px;
  }}
  canvas {{
    border: 1px solid #1a1a1a;
    border-radius: 2px;
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
    width: {canvas_size}px;
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
  <canvas id="c" width="{canvas_size}" height="{canvas_size}"></canvas>
  <div class="bar-track"><div class="bar-fill" id="bar"></div></div>
  <div class="controls">
    <button id="playBtn" onclick="play()">Play</button>
    <button id="revBtn" onclick="reverse()" disabled>Reverse</button>
    <span class="label" id="stateLabel">ready</span>
  </div>

<script>
const particles = {particle_json};
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
  for (let i = 0; i < particles.length; i++) {{
    const p = particles[i];
    const pt = Math.max(0, Math.min(1, (t - p.d) / (1 - p.d)));
    const e = ease(pt);
    const x = (p.sx + (p.tx - p.sx) * e) * W;
    const y = (p.sy + (p.ty - p.sy) * e) * H;
    const r = Math.round(p.sr + (p.tr - p.sr) * e);
    const g = Math.round(p.sg + (p.tg - p.sg) * e);
    const b = Math.round(p.sb + (p.tb - p.sb) * e);
    ctx.fillStyle = 'rgb('+r+','+g+','+b+')';
    ctx.fillRect(x-1, y-1, 2.5, 2.5);
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

// Draw initial state
draw(0);
</script>
</body>
</html>
"""


# ─── Streamlit UI ───────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .block-container { max-width: 700px; }
    h1 { letter-spacing: 0.05em; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Pixel Transport Morph")
st.caption("Upload two images. Pixels from the source morph into the target via luminance-sorted optimal transport.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Source image**")
    source_file = st.file_uploader("Upload source", type=["png", "jpg", "jpeg", "webp"], key="src", label_visibility="collapsed")

with col2:
    st.markdown("**Target image**")
    target_file = st.file_uploader("Upload target", type=["png", "jpg", "jpeg", "webp"], key="tgt", label_visibility="collapsed")

with st.sidebar:
    st.header("Parameters")
    particle_count = st.slider("Particle count", 5000, 30000, 15000, step=1000)
    duration_ms = st.slider("Animation duration (ms)", 1000, 8000, 3000, step=500)
    canvas_size = st.slider("Canvas size (px)", 300, 700, 500, step=50)

# Load images — use defaults if not uploaded
def make_noise_image(size=300):
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)

def make_gradient_circle(size=300):
    img = Image.new("RGB", (size, size), (10, 10, 10))
    arr = np.array(img, dtype=np.float64)
    cx, cy = size / 2, size / 2
    ys, xs = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    mask = dist < size * 0.4
    t = dist[mask] / (size * 0.4)
    arr[mask, 0] = (230 * (1 - t) + 29 * t)
    arr[mask, 1] = (57 * (1 - t) + 123 * t)
    arr[mask, 2] = (70 * (1 - t) + 157 * t)
    return Image.fromarray(arr.astype(np.uint8))

if source_file:
    source_img = Image.open(source_file)
else:
    source_img = make_noise_image()

if target_file:
    target_img = Image.open(target_file)
else:
    target_img = make_gradient_circle()

# Show thumbnails
col1, col2 = st.columns(2)
with col1:
    st.image(source_img, caption="Source", use_container_width=True)
with col2:
    st.image(target_img, caption="Target", use_container_width=True)

# Build particles and render
with st.spinner("Computing transport mapping..."):
    particle_json = build_particle_data(source_img, target_img, particle_count)

st.components.v1.html(
    render_animation_html(particle_json, duration_ms, canvas_size),
    height=canvas_size + 80,
)
