// claude-as-cad hero, pane 5 — "05 · Putting it together: a whole arm" (1:1 with the video).
//
//   import { mountArmBuildPane } from "./hero/armbuild_pane.js";
//   mountArmBuildPane(document.querySelector("#cac-armbuild"), { fallbackVideo, fallbackPoster });
//
// Plays clip 06 step for step: the same stage files the take was recorded from (flattened arm,
// assets/armbuild2.*), the director's camera moves, clicks (with the editor's own captured
// click-to-reference strings), ⌘C/⌘V copies, renames, typed and dragged fields in the primitive-editor
// panel, Save to edit log and the agent's live reloads, interleaved with the video's dialogue
// (tools/build_armbuild_script.py → assets/armbuild2_script.json). Visitors can orbit any time, click
// any face to copy its reference, and jump between build steps.
import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js?v=68c77e28da";
import { buildGeometry, loadBundle } from "./arm.js?v=68c77e28da";
import { ARMBUILD_HISTORY, armBuildTemplate, el } from "./template.js?v=68c77e28da";

const BASE_URL = new URL(".", import.meta.url);
const FPS = 25;
const PASTE_OFFSET_MM = 20; // the editor pastes a copy +20 mm along X
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const mj = (t) => t * t * t * (10 - 15 * t + 6 * t * t); // minimum-jerk, as the director
const clamp01 = (t) => Math.max(0, Math.min(1, t));
const deg = THREE.MathUtils.degToRad;
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
class Aborted extends Error {}

function webglAvailable() {
  try {
    const canvas = document.createElement("canvas");
    return !!(window.WebGLRenderingContext && (canvas.getContext("webgl2") || canvas.getContext("webgl")));
  } catch {
    return false;
  }
}
function showFallback(root, options) {
  root.innerHTML = "";
  if (!options.fallbackVideo) return;
  const fig = el("div", "cac-fallback");
  const video = el("video");
  Object.assign(video, { src: options.fallbackVideo, poster: options.fallbackPoster || "", autoplay: true, muted: true, loop: true, playsInline: true, controls: true });
  fig.appendChild(video);
  root.appendChild(fig);
}

export async function mountArmBuildPane(root, options = {}) {
  if (!root) return null;
  if (!webglAvailable()) {
    showFallback(root, options);
    return null;
  }
  root.classList.add("cac-hero");
  let win = root.querySelector(".cac-window");
  if (!win) {
    win = armBuildTemplate();
    root.innerHTML = "";
    root.appendChild(win);
  }
  const $ = (sel) => win.querySelector(sel);
  const ui = {
    viewer: $(".cac-viewer"), canvas: $(".cac-canvas"), marker: $(".cac-marker"), toast: $(".cac-toast"), toastBody: $(".cac-toast-body"),
    beats: $(".cac-ab-beats"), keys: $(".cac-ab-keys"), log: $(".cac-log"), input: $(".cac-input"), inputText: $(".cac-input-text"),
    keycap: $(".cac-keycap"), key2: $(".cac-key2"), keylabel: $(".cac-keylabel"), ring: $(".cac-ring"), cursor: $(".cac-vcursor"), play: $(".cac-play"),
    peBadge: $(".cac-pe-badge"), peEmpty: $(".cac-pe-empty"), peSel: $(".cac-pe-sel"), peName: $(".cac-pe-name-text"),
    peCaret: $(".cac-pe-caret"), peSub: $(".cac-pe-sub"), peCount: $(".cac-pe-count"), peEdits: $(".cac-pe-edits"), peSaved: $(".cac-pe-saved")
  };
  const field = (i) => win.querySelector(`[data-f="${i}"]`);
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

  // ---------------------------------------------------------------- scene
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas: ui.canvas, antialias: true, alpha: true, powerPreference: "default" });
  } catch {
    showFallback(root, options);
    return null;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 20); // the editor's fov
  camera.up.set(0, 0, 1);
  scene.add(new THREE.HemisphereLight(0xffffff, 0xb3b8c2, 1.35));
  const key = new THREE.DirectionalLight(0xffffff, 1.9);
  key.position.set(0.6, -0.8, 1.4);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  Object.assign(key.shadow.camera, { left: -0.6, right: 0.6, top: 0.6, bottom: -0.6, near: 0.1, far: 4 });
  key.shadow.radius = 4;
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.55);
  fill.position.set(-1, 0.6, 0.5);
  scene.add(fill);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), new THREE.ShadowMaterial({ opacity: 0.12 }));
  ground.receiveShadow = true;
  scene.add(ground);
  const controls = new OrbitControls(camera, ui.canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.09;
  controls.enableZoom = false;
  controls.minDistance = 0.08;
  controls.maxDistance = 3;

  let bundle;
  let script;
  try {
    [bundle, script] = await Promise.all([
      loadBundle(BASE_URL, "armbuild2"),
      fetch(new URL("assets/armbuild2_script.json", BASE_URL), { cache: "no-cache" }).then((r) => {
        if (!r.ok) throw new Error(`armbuild2_script.json ${r.status}`);
        return r.json();
      })
    ]);
  } catch (error) {
    console.warn("claude-as-cad arm-build pane: falling back", error);
    renderer.dispose();
    showFallback(root, options);
    return null;
  }
  const { manifest, bin } = bundle;
  const geoms = manifest.meshes.map((m) => {
    const g = buildGeometry(m, bin);
    g.userData.triMap = m.triMapType === "u16" ? new Uint16Array(bin, m.triMap[0], m.triMap[1] / 2) : new Uint32Array(bin, m.triMap[0], m.triMap[1] / 4);
    return g;
  });
  const world = new THREE.Group();
  scene.add(world);

  // ---------------------------------------------------------------- editor model: stage parts + edits
  // part = { key, name, label, meshIdx, mesh, base (Matrix4 as loaded), c0 (world AABB center of base), copy, edit }
  let parts = new Map();
  let copies = []; // editor copies persist across the agent's live reloads (they're edits)
  const moves = new Map(); // name -> edit for MOVE edits of stage parts
  let currentStage = null;
  let selected = null;
  let clipboardPart = null;
  let dirty = true;

  function aabbCenter(geom, M) {
    const pos = geom.attributes.position;
    const b = new THREE.Box3();
    const v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i += 1) b.expandByPoint(v.fromBufferAttribute(pos, i).applyMatrix4(M));
    return b.getCenter(new THREE.Vector3());
  }
  function makeMesh(meshIdx, color, name) {
    const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.05, emissive: new THREE.Color("#ffffff"), emissiveIntensity: 0 });
    const m = new THREE.Mesh(geoms[meshIdx], mat);
    m.matrixAutoUpdate = false;
    m.castShadow = true;
    m.name = name;
    world.add(m);
    return m;
  }
  // the editor's edit transform: world = T(loc) · R · T(−c0) · base, R = fixed-axis X, then Y, then Z
  function applyEdit(p) {
    if (!p.edit) {
      p.mesh.matrix.copy(p.base);
    } else {
      const c = new THREE.Vector3(...p.edit.loc).multiplyScalar(0.001);
      const R = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(deg(p.edit.rot[0]), deg(p.edit.rot[1]), deg(p.edit.rot[2]), "ZYX"));
      p.mesh.matrix.copy(new THREE.Matrix4().makeTranslation(c.x, c.y, c.z).multiply(R).multiply(new THREE.Matrix4().makeTranslation(-p.c0.x, -p.c0.y, -p.c0.z)).multiply(p.base));
    }
    p.mesh.matrixWorldNeedsUpdate = true;
    dirty = true;
  }
  function loadStage(file) {
    const entries = manifest.stages[file];
    const next = new Map();
    for (const [meshIdx, M, rgba, name] of entries) {
      const key = `${name}#${M.map((x) => x.toFixed(5)).join(",")}`;
      let p = parts.get(key);
      if (p) {
        parts.delete(key);
      } else {
        const base = new THREE.Matrix4().fromArray(M).transpose();
        const color = new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace);
        p = { key, name, meshIdx, mesh: makeMesh(meshIdx, color, name), base, c0: aabbCenter(geoms[meshIdx], base), copy: false, edit: null };
      }
      p.edit = moves.get(name) || null;
      applyEdit(p);
      next.set(key, p);
    }
    for (const p of parts.values()) {
      if (selected === p) select(null);
      p.mesh.removeFromParent();
      p.mesh.material.dispose();
    }
    parts = next;
    currentStage = file;
    hideCoincidentCopies();
    dirty = true;
  }
  // once the agent's reload puts a real part where a copy sits (same box, within 1.5 mm), the real
  // part draws it — the copy stays in the edit list, as in the editor, but isn't drawn on top of it
  function box(geom, M) {
    const pos = geom.attributes.position;
    const b = new THREE.Box3();
    const v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i += 3) b.expandByPoint(v.fromBufferAttribute(pos, i).applyMatrix4(M));
    return b;
  }
  function hideCoincidentCopies() {
    const real = [...parts.values()].map((p) => box(geoms[p.meshIdx], p.mesh.matrix));
    for (const c of copies) {
      const b = box(geoms[c.meshIdx], c.mesh.matrix);
      const cc = b.getCenter(new THREE.Vector3());
      const cs = b.getSize(new THREE.Vector3());
      c.mesh.visible = !real.some((r) => r.getCenter(new THREE.Vector3()).distanceTo(cc) < 0.0015 && r.getSize(new THREE.Vector3()).sub(cs).length() < 0.003);
    }
  }
  function partByName(name, near) {
    let best = null;
    let bestD = Infinity;
    for (const p of [...parts.values(), ...copies]) {
      if (p.name !== name || !p.mesh.visible) continue;
      const d = near ? aabbCenter(geoms[p.meshIdx], p.mesh.matrix).distanceTo(near) : 0;
      if (d < bestD) { best = p; bestD = d; }
    }
    return best;
  }

  // ---------------------------------------------------------------- the primitive-editor panel
  const fmt = (v) => {
    const a = Math.abs(v) < 0.0005 ? 0 : v;
    return Math.abs(a - Math.round(a)) < 0.0005 ? String(Math.round(a)) : a.toFixed(Math.abs(a) >= 100 ? 1 : 2);
  };
  function fieldsOf(p) {
    const e = p.edit || { loc: aabbCenter(geoms[p.meshIdx], p.base).multiplyScalar(1000).toArray(), rot: [0, 0, 0] };
    return [...e.loc, ...e.rot, 1, 1, 1];
  }
  function showFields(p) { fieldsOf(p).forEach((v, i) => { field(i).textContent = fmt(v); }); }
  function select(p) {
    if (selected) selected.mesh.material.emissiveIntensity = 0;
    selected = p;
    ui.peEmpty.hidden = !!p;
    ui.peSel.hidden = !p;
    if (!p) return;
    p.mesh.material.emissiveIntensity = 0.16; // selection: a neutral lift, not a hue shift
    ui.peName.textContent = p.label || p.name;
    ui.peSub.textContent = p.copy ? `copy of ${p.name} · parent: arm` : "part · link arm";
    showFields(p);
    dirty = true;
  }
  function ensureEdit(p) {
    if (!p.edit) p.edit = { loc: aabbCenter(geoms[p.meshIdx], p.base).multiplyScalar(1000).toArray(), rot: [0, 0, 0] };
    if (!p.copy) moves.set(p.name, p.edit);
  }
  function setField(p, idx, v) {
    ensureEdit(p);
    if (idx < 3) p.edit.loc[idx] = v;
    else if (idx < 6) p.edit.rot[idx - 3] = v;
    applyEdit(p);
    field(idx).textContent = fmt(v);
  }
  let notLogged = 0;
  function renderEdits() {
    const rows = [...copies.map((c) => `<div class="cac-pe-cube">■ ${esc(c.label)} (arm)</div>`), ...[...moves.keys()].map((n) => `<div>${esc(n)} (arm) — moved</div>`)];
    ui.peEdits.innerHTML = rows.join("");
    ui.peCount.textContent = String(rows.length);
    ui.peBadge.hidden = !rows.length;
    ui.peBadge.textContent = notLogged ? `${notLogged} not logged` : "logged";
    ui.peBadge.classList.toggle("cac-pe-badge-ok", !notLogged);
  }
  function paste() {
    if (!clipboardPart) return;
    const src = clipboardPart;
    const stem = src.name.replace(/\.stl$/, "");
    let n = 2;
    while (copies.some((c) => c.label === `${stem}_${n}`)) n += 1;
    const loc = aabbCenter(geoms[src.meshIdx], src.mesh.matrix).multiplyScalar(1000).toArray();
    loc[0] += PASTE_OFFSET_MM;
    const p = { key: `copy:${stem}:${copies.length}`, name: src.name, label: `${stem}_${n}`, meshIdx: src.meshIdx, base: src.mesh.matrix.clone(), copy: true,
      mesh: makeMesh(src.meshIdx, src.mesh.material.color.clone(), src.name), edit: { loc, rot: [0, 0, 0] } };
    p.c0 = aabbCenter(geoms[src.meshIdx], p.base);
    applyEdit(p);
    copies.push(p);
    notLogged += 1;
    renderEdits();
    select(p);
  }

  // ---------------------------------------------------------------- render loop + timing
  let visible = true;
  let rafId = 0;
  let firstFrame = false;
  const frameHooks = new Set();
  function resize() {
    const rect = ui.viewer.getBoundingClientRect();
    renderer.setSize(Math.max(1, Math.round(rect.width)), Math.max(1, Math.round(rect.height)), false);
    camera.aspect = Math.max(1, rect.width) / Math.max(1, rect.height);
    camera.updateProjectionMatrix();
    dirty = true;
  }
  function frame(now) {
    rafId = 0;
    const animating = frameHooks.size > 0;
    for (const hook of [...frameHooks]) hook(now);
    const orbiting = controls.update();
    if (animating || orbiting || dirty || !firstFrame) {
      dirty = false;
      renderer.render(scene, camera);
      if (!firstFrame) {
        firstFrame = true;
        ui.viewer.style.setProperty("--cac-poster", "none");
      }
      updateOverlays();
    }
    if (visible) rafId = requestAnimationFrame(frame);
  }
  const startLoop = () => { if (!rafId && visible) rafId = requestAnimationFrame(frame); };
  new ResizeObserver(resize).observe(ui.viewer);
  resize();

  let instant = false; // fast-forward (jumping to a step): every wait/tween completes at once
  const check = (token) => { if (token?.aborted) throw new Aborted(); };
  const wait = (ms, token) => {
    if (instant) { check(token); return Promise.resolve(); }
    return new Promise((resolve, reject) => {
      const start = performance.now();
      const hook = () => {
        if (token?.aborted) { frameHooks.delete(hook); reject(new Aborted()); }
        else if (performance.now() - start >= ms) { frameHooks.delete(hook); resolve(); }
      };
      frameHooks.add(hook);
    });
  };
  const tween = (ms, fn, token, ease = easeInOut) => {
    if (instant) { check(token); fn(1, 1); dirty = true; return Promise.resolve(); }
    return new Promise((resolve, reject) => {
      const start = performance.now();
      const hook = () => {
        if (token?.aborted) { frameHooks.delete(hook); reject(new Aborted()); return; }
        const t = clamp01((performance.now() - start) / ms);
        fn(ease(t), t);
        dirty = true;
        if (t >= 1) { frameHooks.delete(hook); resolve(); }
      };
      frameHooks.add(hook);
    });
  };

  // ---------------------------------------------------------------- camera: the director's camto
  let userCamera = false;
  controls.addEventListener("start", () => { userCamera = true; });
  function camTarget(a) {
    let t = a.at.map((v) => v / 1000);
    if (a.shift) {
      const azr = deg(a.az);
      t = [t[0] - Math.sin(azr) * a.shift * a.dist, t[1] + Math.cos(azr) * a.shift * a.dist, t[2]];
    }
    return new THREE.Vector3(...t);
  }
  const fromSph = (t, r, az, el) => new THREE.Vector3(t.x + r * Math.cos(deg(el)) * Math.cos(deg(az)), t.y + r * Math.cos(deg(el)) * Math.sin(deg(az)), t.z + r * Math.sin(deg(el)));
  function setCam(a) {
    const t = camTarget(a);
    controls.target.copy(t);
    camera.position.copy(fromSph(t, a.dist, a.az, a.el));
    camera.lookAt(t);
    dirty = true;
  }
  async function camto(a, token, speed) {
    if (userCamera) return wait(Math.min(400, ((a.n || 30) / FPS / speed) * 1000), token);
    const T1 = camTarget(a);
    const T0 = controls.target.clone();
    const d = camera.position.clone().sub(T0);
    const r0 = d.length();
    const az0 = (Math.atan2(d.y, d.x) * 180) / Math.PI;
    const el0 = (Math.asin(d.z / r0) * 180) / Math.PI;
    let daz = a.az - az0;
    while (daz > 180) daz -= 360;
    while (daz < -180) daz += 360;
    await tween(((a.n || 30) / FPS / speed) * 1000, (e) => {
      if (userCamera) return;
      const T = T0.clone().lerp(T1, e);
      controls.target.copy(T);
      camera.position.copy(fromSph(T, r0 + (a.dist - r0) * e, az0 + daz * e, el0 + (a.el - el0) * e));
      camera.lookAt(T);
    }, token, mj);
  }

  // after the last step: ease out to the whole finished arm (the director's angle, fitted to its box)
  async function frameAll(token) {
    const b = new THREE.Box3();
    for (const p of parts.values()) if (p.mesh.visible) b.expandByObject(p.mesh);
    const sphere = b.getBoundingSphere(new THREE.Sphere());
    const dist = sphere.radius / Math.sin(deg(camera.fov) / 2) * 1.05;
    await camto({ at: sphere.center.clone().multiplyScalar(1000).toArray(), dist, az: -40, el: 18, shift: 0.1, n: 40 }, token, 1);
  }

  // ---------------------------------------------------------------- overlays: cursor, ring, toast, keys
  const winRect = () => win.getBoundingClientRect();
  function projectToWindow(w) {
    const v = w.clone().project(camera);
    const vr = ui.viewer.getBoundingClientRect();
    const wr = winRect();
    return { x: vr.left - wr.left + (v.x * 0.5 + 0.5) * vr.width, y: vr.top - wr.top + (-v.y * 0.5 + 0.5) * vr.height };
  }
  function clampToViewer(p) {
    const vr = ui.viewer.getBoundingClientRect();
    const wr = winRect();
    const x0 = vr.left - wr.left;
    const y0 = vr.top - wr.top;
    return { x: Math.max(x0 + 6, Math.min(x0 + vr.width - 6, p.x)), y: Math.max(y0 + 6, Math.min(y0 + vr.height - 6, p.y)) };
  }
  const elementPoint = (node, fx = 0.5, fy = 0.55) => {
    const r = node.getBoundingClientRect();
    const wr = winRect();
    return { x: r.left - wr.left + r.width * fx, y: r.top - wr.top + r.height * fy };
  };
  const inputCaretPoint = () => elementPoint(win.querySelector(".cac-panel .cac-caret"), 0.6, 0.7);
  let cursorPos = null;
  const moveCursor = () => { ui.cursor.style.transform = `translate(${cursorPos.x}px, ${cursorPos.y}px)`; };
  function showCursor(point) { cursorPos = { ...point }; ui.cursor.hidden = false; moveCursor(); }
  const hideCursor = () => { ui.cursor.hidden = true; };
  const cursorKind = (k) => ui.cursor.classList.toggle("cac-vcursor-scrub", k === "scrub");
  function glide(getTarget, token, speed = 1300) {
    if (instant || !cursorPos) { cursorPos = getTarget(); moveCursor(); return Promise.resolve(); }
    const from = { ...cursorPos };
    const first = getTarget();
    const dist = Math.hypot(first.x - from.x, first.y - from.y);
    const ms = Math.max(300, Math.min(780, (dist / speed) * 1000));
    const bend = Math.min(50, dist * 0.16);
    return tween(ms, (e) => {
      const to = getTarget();
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const len = Math.hypot(dx, dy) || 1;
      const arc = Math.sin(Math.PI * e) * bend;
      cursorPos = { x: from.x + dx * e + (-dy / len) * arc, y: from.y + dy * e + (dx / len) * arc };
      moveCursor();
    }, token);
  }
  function ringAt(point, ring = ui.ring) {
    if (instant) return;
    ring.style.left = `${point.x}px`;
    ring.style.top = `${point.y}px`;
    ring.classList.remove("cac-ring-on");
    void ring.offsetWidth;
    ring.classList.add("cac-ring-on");
  }
  let markerWorld = null;
  function placeToast(toast, w) {
    if (!w || toast.hidden) return;
    const p = projectToWindow(w);
    const vr = ui.viewer.getBoundingClientRect();
    const wr = winRect();
    const lx = p.x - (vr.left - wr.left);
    const ly = p.y - (vr.top - wr.top);
    const tw = toast.offsetWidth;
    const th = toast.offsetHeight;
    toast.style.left = `${Math.max(8, Math.min(vr.width - tw - 8, lx - tw * 0.35))}px`;
    toast.style.top = `${Math.max(8, Math.min(vr.height - th - 8, ly + 18 + th < vr.height - 8 ? ly + 18 : ly - th - 18))}px`;
  }
  function placeMarker(marker, w) {
    const p = projectToWindow(w);
    const vr = ui.viewer.getBoundingClientRect();
    const wr = winRect();
    marker.style.transform = `translate(${p.x - (vr.left - wr.left)}px, ${p.y - (vr.top - wr.top)}px)`;
  }
  function updateOverlays() {
    if (markerWorld) { placeMarker(ui.marker, markerWorld); placeToast(ui.toast, markerWorld); }
    updateVisitorClick();
  }
  async function showKeys(caps, frames, token) {
    if (instant) return;
    ui.keys.innerHTML = caps.map((c) => `<kbd>${esc(c)}</kbd>`).join("");
    ui.keys.hidden = false;
    await wait(Math.max(260, (frames / FPS) * 1000), token);
    ui.keys.hidden = true;
  }

  // ---------------------------------------------------------------- the director ops
  async function pick(op, token) {
    const hit = new THREE.Vector3(...op.hit).multiplyScalar(0.001);
    const target = () => clampToViewer(projectToWindow(hit));
    await glide(target, token);
    ringAt(target());
    const p = partByName(op.name, hit);
    if (p) select(p);
    if (!instant) {
      markerWorld = hit;
      ui.marker.hidden = false;
      ui.toastBody.textContent = op.clip;
      ui.toast.hidden = false;
      updateOverlays();
      await wait(900, token);
      ui.toast.hidden = true;
      ui.marker.hidden = true;
      markerWorld = null;
    }
  }
  async function scrub(idx, target, token, speed) {
    if (!selected) return;
    const node = field(idx);
    await glide(() => elementPoint(node), token);
    cursorKind("scrub");
    node.classList.add("cac-pe-active");
    const from = fieldsOf(selected)[idx];
    const x0 = cursorPos.x;
    const span = Math.max(-60, Math.min(60, (target - from) * 0.9));
    const overshoot = (target - from) * 0.05;
    await tween(Math.min(1300, 450 + Math.abs(target - from) * 6) / speed, (e, t) => {
      const settle = Math.min(1, t / 0.8);
      const v = t >= 1 ? target : from + (target - from) * settle + overshoot * Math.sin(Math.PI * clamp01((t - 0.55) / 0.45));
      setField(selected, idx, v);
      cursorPos = { x: x0 + span * settle, y: cursorPos.y };
      moveCursor();
    }, token, (t) => t);
    node.classList.remove("cac-pe-active");
    cursorKind("arrow");
  }
  async function typeField(idx, value, token) {
    if (!selected) return;
    const node = field(idx);
    await glide(() => elementPoint(node), token);
    ringAt(cursorPos);
    node.classList.add("cac-pe-active");
    if (!instant) {
      node.textContent = "";
      for (const ch of String(value)) { node.textContent += ch; await wait(70, token); }
      await showKeys(["⏎"], 6, token);
    }
    node.classList.remove("cac-pe-active");
    setField(selected, idx, value);
  }
  async function rename(text, token) {
    if (!selected) return;
    await glide(() => elementPoint(ui.peName, 0.3), token);
    ringAt(cursorPos);
    if (!instant) {
      await showKeys(["⌘", "A"], 6, token);
      ui.peCaret.hidden = false;
      ui.peName.textContent = "";
      for (const ch of text) { ui.peName.textContent += ch; await wait(55, token); }
      await showKeys(["⏎"], 6, token);
      ui.peCaret.hidden = true;
    }
    ui.peName.textContent = text;
    selected.label = text;
    renderEdits();
  }
  async function clickButton(text, token) {
    const node = win.querySelector(`[data-btn="${text}"]`);
    if (!node) return;
    await glide(() => elementPoint(node), token);
    ringAt(cursorPos);
    if (text === "Save to edit log") {
      notLogged = 0;
      renderEdits();
      if (!instant) {
        ui.peSaved.hidden = false;
        await wait(700, token);
        ui.peSaved.hidden = true;
      }
    }
  }
  async function playOps(ops, speed, token) {
    for (const op of ops) {
      check(token);
      if (op.op === "idle") await wait((op.n / FPS / speed) * 1000, token);
      else if (op.op === "camto") await camto(op, token, speed);
      else if (op.op === "pick") await pick(op, token);
      else if (op.op === "keys") {
        if (op.combo === "Control+c") clipboardPart = selected;
        await showKeys(op.caps, op.hold || 8, token);
        if (op.combo === "Control+v") paste();
      } else if (op.op === "rename") await rename(op.text, token);
      else if (op.op === "typefield") await typeField(op.idx, op.value, token);
      else if (op.op === "scrub") await scrub(op.idx, op.target, token, speed);
      else if (op.op === "click_button") await clickButton(op.text, token);
      else if (op.op === "stage") { loadStage(op.file); await wait(((op.hold || 2) / FPS) * 1000 + 250, token); }
      else if (op.op === "stageseq") {
        for (const f of op.files) { loadStage(f); await wait(((op.hold || 2) / FPS) * 1000 + 40, token); }
      }
    }
  }

  // ---------------------------------------------------------------- chat
  function logLine(kind, html, extra = "") {
    const line = el("div", `cac-line cac-${kind} ${extra}`.trim(), html);
    ui.log.appendChild(line);
    ui.log.scrollTop = ui.log.scrollHeight;
    return line;
  }
  const dimOlder = () => { for (const n of ui.log.querySelectorAll(".cac-user, .cac-agent, .cac-tool")) n.classList.add("cac-dim"); };
  async function typeChat(text, token, cps = 70) {
    if (instant) { ui.inputText.insertAdjacentText("beforeend", text); return; }
    for (const ch of text) {
      check(token);
      ui.inputText.insertAdjacentText("beforeend", ch);
      await wait(ch === " " ? 1000 / (cps * 1.8) : (1000 / cps) * (0.7 + Math.random() * 0.6), token);
    }
  }
  async function chatKeycap(second, label, token) {
    if (instant) return;
    ui.key2.textContent = second;
    ui.keylabel.textContent = label;
    ui.keycap.hidden = false;
    ui.keycap.classList.remove("cac-key-press");
    void ui.keycap.offsetWidth;
    ui.keycap.classList.add("cac-key-press");
    await wait(400, token);
    ui.keycap.hidden = true;
  }
  async function think(ms, token) {
    if (instant) return;
    const line = logLine("thinking", `<span class="cac-spin">·</span> Thinking…`);
    const glyph = line.querySelector(".cac-spin");
    const frames = ["·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢"];
    try {
      for (let t = 0, f = 0; t < ms; t += 110, f += 1) { glyph.textContent = frames[f % frames.length]; await wait(110, token); }
    } finally {
      line.remove();
    }
  }
  async function ask(items, token) {
    await glide(inputCaretPoint, token);
    ui.input.classList.add("cac-input-active");
    for (const [kind, val] of items) {
      if (kind === "type") {
        await glide(inputCaretPoint, token);
        await typeChat(val, token);
      } else if (kind === "click") {
        await playOps(val, 1, token);
      } else if (kind === "paste") {
        await glide(inputCaretPoint, token);
        await chatKeycap("V", "paste", token);
        ui.inputText.insertAdjacentHTML("beforeend", `<span class="cac-ref">${esc(val)}</span>`);
      }
    }
    await wait(500, token);
    const html = ui.inputText.innerHTML;
    ui.inputText.innerHTML = "";
    ui.input.classList.remove("cac-input-active");
    dimOlder();
    logLine("user", `&gt; ${html}`);
    await think(500, token);
  }
  async function run(step, token) {
    const node = logLine("tool", `<span class="cac-bullet cac-ok">●</span> ${esc(step.title)}<div class="cac-tool-out"></div>`);
    const out = node.querySelector(".cac-tool-out");
    for (const [i, text] of step.lines.entries()) {
      out.insertAdjacentHTML("beforeend", `<div>${i === 0 ? "⎿ " : "&nbsp;&nbsp;"}${esc(text)}</div>`);
      ui.log.scrollTop = ui.log.scrollHeight;
      const hooks = step.between?.[String(i)];
      if (hooks) {
        for (const h of hooks) {
          if (h.t === "play") await playOps(h.ops, h.speed || 1, token);
          else if (h.t === "idle") await wait(h.ms, token);
        }
      } else {
        await wait(160, token);
      }
    }
  }
  async function say(text, token) {
    const node = logLine("agent", `<span class="cac-bullet">●</span> `);
    if (instant) { node.insertAdjacentText("beforeend", text); return; }
    for (const word of text.split(" ")) {
      check(token);
      node.insertAdjacentText("beforeend", `${word} `);
      ui.log.scrollTop = ui.log.scrollHeight;
      await wait(16, token);
    }
    await wait(300, token);
  }

  // ---------------------------------------------------------------- the run, beats, jumping
  const beatChips = script.script.filter((s) => s.t === "beat").map((s, k) => {
    const b = el("button", "cac-ab-beat", `<span>${esc(s.name)}</span>${esc(s.title)}`);
    b.type = "button";
    b.title = `${s.name} · ${s.title}`;
    b.addEventListener("click", () => jumpTo(k));
    ui.beats.appendChild(b);
    return b;
  });
  let activeBeat = 0;
  function markBeat(k) {
    activeBeat = Math.min(k, beatChips.length - 1);
    beatChips.forEach((b, j) => {
      b.classList.toggle("cac-ab-on", j === k);
      b.classList.toggle("cac-ab-done", j < k);
    });
  }
  function resetAll() {
    for (const p of [...parts.values(), ...copies]) { p.mesh.removeFromParent(); p.mesh.material.dispose(); }
    parts = new Map();
    copies = [];
    moves.clear();
    selected = null;
    clipboardPart = null;
    notLogged = 0;
    select(null);
    renderEdits();
    ui.log.innerHTML = "";
    for (const item of ARMBUILD_HISTORY) logLine("meta", esc(item.text));
    ui.inputText.innerHTML = "";
    ui.input.classList.remove("cac-input-active");
    for (const n of [ui.toast, ui.marker, ui.keys, ui.keycap]) n.hidden = true;
    markerWorld = null;
    loadStage(script.startStage);
    if (!userCamera) setCam(script.startCam);
  }
  // run the script; steps before `startBeat` are fast-forwarded (state only), the rest plays live
  async function runScript(startBeat, token, { stopAfterBeat = Infinity } = {}) {
    resetAll();
    instant = startBeat > 0;
    let k = -1;
    for (const s of script.script) {
      check(token);
      if (s.t === "beat") {
        k += 1;
        if (k > stopAfterBeat) break;
        if (k === startBeat) instant = false;
        markBeat(k);
        if (!instant) {
          if (!cursorPos) cursorPos = inputCaretPoint();
          ui.cursor.hidden = false;
          moveCursor();
        }
        continue;
      }
      if (s.t === "play") await playOps(s.ops, s.speed || 1, token);
      else if (s.t === "ask") await ask(s.items, token);
      else if (s.t === "run") await run(s, token);
      else if (s.t === "say") await say(s.text, token);
      else if (s.t === "note") { await say(s.text, token); await think(500, token); }
    }
    if (stopAfterBeat === Infinity || stopAfterBeat >= beatChips.length - 1) await frameAll(token);
    instant = false;
  }
  let replayToken = null;
  let running = false;
  async function loop(startBeat = 0) {
    if (replayToken) replayToken.aborted = true;
    frameHooks.clear();
    const token = { aborted: false };
    replayToken = token;
    running = true;
    try {
      for (let b = startBeat; ; b = 0) {
        userCamera = false;
        await runScript(b, token);
        markBeat(beatChips.length);
        hideCursor();
        await wait(5000, token);
      }
    } catch (error) {
      instant = false;
      if (!(error instanceof Aborted)) throw error;
    } finally {
      if (replayToken === token) running = false;
    }
  }
  function stopReplay() {
    if (replayToken) replayToken.aborted = true;
    replayToken = null;
    running = false;
    frameHooks.clear();
    instant = false;
  }
  async function showEndOf(k) {
    // reduced motion: the state at the end of step k, no animation
    stopReplay();
    const token = { aborted: false };
    replayToken = token;
    userCamera = false;
    instant = true;
    try {
      await runScript(beatChips.length + 1, token, { stopAfterBeat: k });
    } finally {
      instant = false;
      hideCursor();
      markBeat(k);
    }
  }
  function jumpTo(k) {
    if (reduceMotion) { showEndOf(k); return; }
    stopReplay();
    userCamera = false;
    loop(k);
  }

  // ---------------------------------------------------------------- visitor click-to-reference (any visible face)
  const visitor = { ring: ui.ring.cloneNode(true), marker: ui.marker.cloneNode(true), toast: ui.toast.cloneNode(true), mesh: null, local: null, highlight: null, gen: 0 };
  ui.ring.after(visitor.ring);
  ui.marker.after(visitor.marker);
  ui.toast.after(visitor.toast);
  visitor.ring.classList.remove("cac-ring-on");
  visitor.marker.hidden = true;
  visitor.toast.hidden = true;
  visitor.toast.classList.add("cac-toast-visitor");
  const visitorTitle = visitor.toast.querySelector(".cac-toast-title");
  const visitorBody = visitor.toast.querySelector(".cac-toast-body");
  const pickRay = new THREE.Raycaster();
  function updateVisitorClick() {
    if (!visitor.mesh) return;
    const w = visitor.local.clone().applyMatrix4(visitor.mesh.matrixWorld);
    placeMarker(visitor.marker, w);
    placeToast(visitor.toast, w);
  }
  function triPlane(g, t, n, cen) {
    const pos = g.attributes.position;
    const idx = g.index.array;
    const a = new THREE.Vector3().fromBufferAttribute(pos, idx[3 * t]);
    const b = new THREE.Vector3().fromBufferAttribute(pos, idx[3 * t + 1]);
    const c = new THREE.Vector3().fromBufferAttribute(pos, idx[3 * t + 2]);
    n.subVectors(c, b).cross(new THREE.Vector3().subVectors(a, b)).normalize();
    cen.copy(a).add(b).add(c).multiplyScalar(1 / 3);
  }
  function coplanarRegion(g, t0) {
    if (!g.userData.cornerTris) {
      const pos = g.attributes.position;
      const idx = g.index.array;
      const keyOf = new Map();
      const corner = new Uint32Array(idx.length);
      for (let c = 0; c < idx.length; c += 1) {
        const v = idx[c];
        const k = `${Math.round(pos.getX(v) * 1e5)},${Math.round(pos.getY(v) * 1e5)},${Math.round(pos.getZ(v) * 1e5)}`;
        let id = keyOf.get(k);
        if (id === undefined) { id = keyOf.size; keyOf.set(k, id); }
        corner[c] = id;
      }
      const cornerTris = Array.from({ length: keyOf.size }, () => []);
      for (let c = 0; c < corner.length; c += 1) cornerTris[corner[c]].push((c / 3) | 0);
      Object.assign(g.userData, { corner, cornerTris });
    }
    const { corner, cornerTris } = g.userData;
    const n0 = new THREE.Vector3();
    const c0 = new THREE.Vector3();
    triPlane(g, t0, n0, c0);
    const d0 = n0.dot(c0);
    const n = new THREE.Vector3();
    const cen = new THREE.Vector3();
    const seen = new Uint8Array(corner.length / 3);
    const region = [t0];
    seen[t0] = 1;
    for (let r = 0; r < region.length && region.length < 20000; r += 1) {
      for (let k = 0; k < 3; k += 1) {
        for (const u of cornerTris[corner[3 * region[r] + k]]) {
          if (seen[u]) continue;
          seen[u] = 1;
          triPlane(g, u, n, cen);
          if (n.dot(n0) > 0.999 && Math.abs(n0.dot(cen) - d0) < 0.0003) region.push(u);
        }
      }
    }
    return { tris: region, normal: n0 };
  }
  function clearVisitorClick() {
    visitor.gen += 1;
    if (visitor.highlight) {
      visitor.highlight.removeFromParent();
      visitor.highlight.geometry.dispose();
      visitor.highlight.material.dispose();
      visitor.highlight = null;
    }
    visitor.toast.hidden = true;
    visitor.marker.hidden = true;
    visitor.toast.classList.remove("cac-fade");
    visitor.marker.classList.remove("cac-fade");
    visitor.mesh = null;
    dirty = true;
  }
  function visitorClick(clientX, clientY) {
    const rect = ui.canvas.getBoundingClientRect();
    pickRay.setFromCamera(new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1), camera);
    world.updateMatrixWorld(true);
    const pickables = [...parts.values(), ...copies].map((p) => p.mesh).filter((m) => m.visible);
    const hit = pickRay.intersectObjects(pickables, false)[0];
    clearVisitorClick();
    if (!hit) return;
    const m = hit.object;
    const g = m.geometry;
    const region = coplanarRegion(g, hit.faceIndex);
    const local = hit.point.clone().applyMatrix4(m.matrixWorld.clone().invert()); // mesh-local m = STL mm / 1000
    const nWorld = region.normal.clone().transformDirection(m.matrixWorld);
    const comps = [nWorld.x, nWorld.y, nWorld.z];
    let k = 0;
    for (let j = 1; j < 3; j += 1) if (Math.abs(comps[j]) > Math.abs(comps[k])) k = j;
    const f1 = (v) => (Math.abs(v) < 0.05 ? 0 : v).toFixed(1);
    const f2 = (v) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2);
    const w = hit.point.clone().multiplyScalar(1000);
    const l = local.clone().multiplyScalar(1000);
    const text = `[${m.name} in arm @ (${f1(w.x)}, ${f1(w.y)}, ${f1(w.z)})mm | local (${f1(l.x)}, ${f1(l.y)}, ${f1(l.z)})mm n(${[region.normal.x, region.normal.y, region.normal.z].map(f2).join(", ")}) tri ${g.userData.triMap[hit.faceIndex]}, ${comps[k] >= 0 ? "+" : "-"}${"XYZ"[k]} side face]`;
    const pos = g.attributes.position;
    const idx = g.index.array;
    const out = new Float32Array(region.tris.length * 9);
    const lift = region.normal.clone().multiplyScalar(0.00025);
    const v = new THREE.Vector3();
    region.tris.forEach((t, i) => {
      for (let c = 0; c < 3; c += 1) { v.fromBufferAttribute(pos, idx[3 * t + c]).add(lift); out.set([v.x, v.y, v.z], i * 9 + c * 3); }
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(out, 3));
    const hl = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: "#22c4f0", transparent: true, opacity: 0.85, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }));
    hl.renderOrder = 2;
    m.add(hl);
    Object.assign(visitor, { mesh: m, local, highlight: hl });
    const wr = winRect();
    ringAt({ x: clientX - wr.left, y: clientY - wr.top }, visitor.ring);
    visitor.marker.hidden = false;
    visitorTitle.textContent = "Copied to clipboard:";
    visitorBody.textContent = text;
    visitor.toast.hidden = false;
    updateVisitorClick();
    dirty = true;
    try { navigator.clipboard.writeText(text).catch(() => { visitorTitle.textContent = "Face reference:"; }); } catch { visitorTitle.textContent = "Face reference:"; }
    const gen = visitor.gen;
    setTimeout(() => {
      if (gen !== visitor.gen) return;
      visitor.toast.classList.add("cac-fade");
      visitor.marker.classList.add("cac-fade");
      tween(400, (e) => { hl.material.opacity = 0.85 * (1 - e); }, null).catch(() => {});
      setTimeout(() => { if (gen === visitor.gen) clearVisitorClick(); }, 450);
    }, 1000);
  }

  // ---------------------------------------------------------------- interaction + lifecycle
  let pressAt = null;
  ui.canvas.addEventListener("pointerdown", (event) => {
    controls.enableZoom = true;
    if (event.button === 0) pressAt = { x: event.clientX, y: event.clientY, t: performance.now() };
  });
  ui.canvas.addEventListener("pointerup", (event) => {
    if (!pressAt || event.button !== 0) return;
    const moved = Math.hypot(event.clientX - pressAt.x, event.clientY - pressAt.y);
    const quick = performance.now() - pressAt.t < 650;
    pressAt = null;
    if (moved < 6 && quick) visitorClick(event.clientX, event.clientY);
  });
  ui.canvas.addEventListener("pointerleave", () => { controls.enableZoom = false; });
  ui.play.addEventListener("click", () => { ui.play.hidden = true; loop(0); });
  const io = new IntersectionObserver((entries) => {
    const nowVisible = entries.some((e) => e.isIntersecting);
    if (nowVisible === visible) return;
    visible = nowVisible;
    if (visible) {
      startLoop();
      if (!reduceMotion && !running) loop(activeBeat);
    } else if (running && !reduceMotion) {
      stopReplay();
    }
  }, { threshold: 0.15 });
  io.observe(win);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && running && !reduceMotion) stopReplay();
    else if (!document.hidden && visible && !reduceMotion && !running) loop(activeBeat);
  });
  if (options.debug) {
    window.__cacBuild = {
      THREE, camera, controls, jumpTo,
      get stage() { return currentStage; }, get beat() { return activeBeat; }, get running() { return running; },
      get copies() { return copies.map((c) => ({ label: c.label, visible: c.mesh.visible, loc: c.edit.loc.map((x) => +x.toFixed(2)), rot: c.edit.rot })); }
    };
  }
  startLoop();
  resetAll();
  if (reduceMotion || options.autoplay === false) {
    await showEndOf(beatChips.length - 1);
    if (reduceMotion) ui.play.hidden = false;
  } else {
    loop(0);
  }
  return {
    destroy() {
      io.disconnect();
      stopReplay();
      cancelAnimationFrame(rafId);
      renderer.dispose();
    }
  };
}
