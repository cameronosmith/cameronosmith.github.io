// claude-as-cad page — "Robot prototypes" gallery (the opening video's CAD | real grid, made interactive).
//
//   import { mountGallery } from "./hero/gallery_pane.js";
//   mountGallery(document.querySelector("#cac-gallery"), { descriptions: { arm: "…" } });
//
// Each card: the build's CAD on the left (assembles part by part when it first comes into view, then
// orbits slowly; drag to orbit it yourself, double-click to replay the assembly) joined to a photo of
// the real build on the right, with its name (and an optional description) underneath. One shared
// WebGL renderer draws every card into its own 2D canvas, so the page doesn't add six WebGL contexts.
// Builds: assets/gallery/gallery_<key>.{json,bin} (tools/export_gallery.py, from the video's renders).
import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { buildGeometry, loadBundle } from "./arm.js?v=36b142ff4a";
import { GALLERY_ITEMS, galleryGrid } from "./template.js?v=36b142ff4a";

const BASE_URL = new URL(".", import.meta.url);
const DROP = 0.42; // seconds each piece takes to land (as the renders)
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const backOut = (t) => { const s = 1.3; t = clamp(t) - 1; return t * t * ((s + 1) * t + s) + 1; };
const ss = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };

let shared = null; // one renderer for every card
function sharedRenderer() {
  if (shared) return shared;
  try {
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true, powerPreference: "default" });
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    shared = renderer;
  } catch {
    shared = false;
  }
  return shared;
}

export function mountGallery(root, options = {}) {
  if (!root) return null;
  root.classList.add("cac-hero");
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  let grid = root.querySelector(".cac-gal");
  if (!grid) {
    grid = galleryGrid(options.descriptions || {});
    root.innerHTML = "";
    root.appendChild(grid);
  }
  const cards = GALLERY_ITEMS.map((item) => {
    const fig = grid.querySelector(`[data-build="${item.key}"]`);
    return { item, fig, canvas: fig.querySelector("canvas"), loaded: false, visible: false };
  });
  const renderer = sharedRenderer();
  if (!renderer) {
    for (const c of cards) c.fig.querySelector(".cac-gal-cad").classList.add("cac-gal-nogl");
    return null;
  }

  // ---------------------------------------------------------------- per-card scene
  async function load(card) {
    card.loaded = true;
    let bundle;
    try {
      bundle = await loadBundle(BASE_URL, `gallery/gallery_${card.item.key}`);
    } catch (error) {
      console.warn("claude-as-cad gallery: couldn't load", card.item.key, error);
      return;
    }
    const { manifest, bin } = bundle;
    const scene = new THREE.Scene();
    const lo = new THREE.Vector3(...manifest.lo);
    const hi = new THREE.Vector3(...manifest.hi);
    const ctr = lo.clone().add(hi).multiplyScalar(0.5);
    const R = hi.clone().sub(lo).length() / 2;
    scene.add(new THREE.HemisphereLight(0xffffff, 0xb8bcc4, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.radius = 5;
    const sh = key.shadow.camera;
    sh.left = sh.bottom = -2 * R;
    sh.right = sh.top = 2 * R;
    sh.near = 0.01;
    sh.far = 20 * R;
    key.position.set(ctr.x + 2 * R, ctr.y - 3 * R, ctr.z + 5 * R);
    key.target.position.copy(ctr);
    scene.add(key, key.target);
    const fill = new THREE.DirectionalLight(0xffffff, 0.6);
    fill.position.set(ctr.x - 3 * R, ctr.y + 2 * R, ctr.z + R);
    scene.add(fill);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(40 * R, 40 * R), new THREE.ShadowMaterial({ opacity: 0.16 }));
    ground.position.set(ctr.x, ctr.y, lo.z - 0.0005);
    ground.receiveShadow = true;
    scene.add(ground);
    const cfg = manifest.cfg;
    // assembly order as the renders: by link, bottom-up within it (or strictly bottom-up)
    const order = manifest.parts.map((p, i) => ({ p, i })).sort((a, b) => (cfg.order === "z" ? a.p.lo[2] - b.p.lo[2] : a.p.li - b.p.li || a.p.center[2] - b.p.center[2]));
    const meshes = order.map(({ p }) => {
      const color = new THREE.Color().setRGB(p.rgba[0], p.rgba[1], p.rgba[2], THREE.SRGBColorSpace);
      const m = new THREE.Mesh(buildGeometry(p, bin), new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.05, transparent: true }));
      m.castShadow = m.receiveShadow = true;
      scene.add(m);
      return m;
    });
    const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 50);
    camera.up.set(0, 0, 1);
    // one distance that keeps the whole build in frame at every azimuth of a full orbit
    const corners = [];
    for (const x of [lo.x, hi.x]) for (const y of [lo.y, hi.y]) for (const z of [lo.z, hi.z]) corners.push(new THREE.Vector3(x, y, z));
    const el0 = THREE.MathUtils.degToRad(cfg.el);
    const fitsAt = (d, a) => {
      camera.position.set(ctr.x + d * Math.cos(el0) * Math.cos(a), ctr.y + d * Math.cos(el0) * Math.sin(a), ctr.z + d * Math.sin(el0));
      camera.lookAt(ctr);
      camera.updateMatrixWorld();
      return corners.every((c) => { const q = c.clone().project(camera); return Math.abs(q.x) <= 0.98 && Math.abs(q.y) <= 0.98 && q.z < 1; });
    };
    let dist = 0;
    for (let k = 0; k < 24; k += 1) {
      const a = (k / 24) * Math.PI * 2;
      let lo_ = R * 0.2;
      let hi_ = R * 40;
      for (let it = 0; it < 30; it += 1) { const mid = (lo_ + hi_) / 2; if (fitsAt(mid, a)) hi_ = mid; else lo_ = mid; }
      dist = Math.max(dist, hi_);
    }
    dist *= 0.9; // read at thumbnail size (the video's grid zoomed its CAD tiles too); box corners are generous
    const controls = new OrbitControls(camera, card.canvas);
    controls.target.copy(ctr);
    controls.enableZoom = false;
    controls.enablePan = false;
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    Object.assign(card, { scene, camera, controls, meshes, ctr, R, dist, cfg, az: cfg.az0, el: cfg.el, t0: null, userUntil: 0 });
    controls.addEventListener("start", () => { card.userUntil = Infinity; });
    controls.addEventListener("end", () => {
      // resume the slow orbit from wherever the visitor left it
      const d = camera.position.clone().sub(ctr);
      card.az = (Math.atan2(d.y, d.x) * 180) / Math.PI;
      card.el = (Math.asin(d.z / d.length()) * 180) / Math.PI;
      card.userUntil = performance.now() + 3500;
    });
    card.canvas.addEventListener("dblclick", () => { card.t0 = performance.now(); });
    card.ctx = card.canvas.getContext("2d");
    setCamera(card);
    if (reduceMotion) card.t0 = -1e9; // fully assembled, still
  }
  function setCamera(card) {
    const a = THREE.MathUtils.degToRad(card.az);
    const e = THREE.MathUtils.degToRad(card.el);
    const d = card.dist;
    card.camera.position.set(card.ctr.x + d * Math.cos(e) * Math.cos(a), card.ctr.y + d * Math.cos(e) * Math.sin(a), card.ctr.z + d * Math.sin(e));
    card.camera.lookAt(card.ctr);
    card.controls.target.copy(card.ctr);
  }

  // ---------------------------------------------------------------- one render loop for the grid
  let last = performance.now();
  let raf = 0;
  function frame(now) {
    raf = 0;
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    let any = false;
    for (const card of cards) {
      if (!card.visible || !card.scene) continue;
      any = true;
      if (card.t0 === null) {
        // assemble once the card is really on screen (it loads a little earlier)
        const r = card.fig.getBoundingClientRect();
        if (r.top < window.innerHeight * 0.92 && r.bottom > window.innerHeight * 0.08) card.t0 = now;
      }
      const t = card.t0 === null ? -1 : (now - card.t0) / 1000;
      const N = card.meshes.length;
      const stagger = (card.cfg.dur - DROP) / Math.max(1, N - 1);
      card.meshes.forEach((m, k) => {
        const u = clamp((t - k * stagger) / DROP);
        const e = backOut(u);
        m.visible = u > 0;
        m.position.set(0, 0, (1 - e) * 0.55 * card.R); // drop in from above, slight overshoot
        m.material.opacity = ss(0, 0.35, u);
        m.material.transparent = u < 1;
      });
      if (!reduceMotion && now > card.userUntil) {
        card.az += ((card.cfg.az1 - card.cfg.az0) / 3) * dt; // the renders' orbit speed
        setCamera(card);
      }
      card.controls.update();
      // draw this card through the shared renderer
      const w = card.canvas.clientWidth;
      const h = card.canvas.clientHeight;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const W = Math.max(1, Math.round(w * dpr));
      const H = Math.max(1, Math.round(h * dpr));
      if (card.canvas.width !== W || card.canvas.height !== H) {
        card.canvas.width = W;
        card.canvas.height = H;
      }
      const size = renderer.getSize(new THREE.Vector2());
      if (size.x !== W || size.y !== H || renderer.getPixelRatio() !== 1) {
        renderer.setPixelRatio(1);
        renderer.setSize(W, H, false);
      }
      card.camera.aspect = W / H;
      card.camera.updateProjectionMatrix();
      renderer.render(card.scene, card.camera);
      card.ctx.clearRect(0, 0, W, H);
      card.ctx.drawImage(renderer.domElement, 0, 0, W, H);
    }
    if (any) raf = requestAnimationFrame(frame);
  }
  const kick = () => { if (!raf) { last = performance.now(); raf = requestAnimationFrame(frame); } };
  const io = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      const card = cards.find((c) => c.fig === entry.target);
      if (!card) continue;
      card.visible = entry.isIntersecting;
      if (entry.isIntersecting && !card.loaded) load(card).then(kick);
    }
    kick();
  }, { rootMargin: "200px 0px" });
  for (const c of cards) io.observe(c.fig);
  if (options.debug) window.__cacGallery = { cards };
  return { destroy() { io.disconnect(); cancelAnimationFrame(raf); } };
}
