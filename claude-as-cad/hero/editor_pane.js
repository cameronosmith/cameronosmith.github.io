// claude-as-cad hero, pane 2 — "02 · Primitive editor".
//
//   import { mountEditorPane } from "./hero/editor_pane.js";
//   mountEditorPane(document.querySelector("#cac-editor"), { fallbackVideo, fallbackPoster });
//
// Same viewer setup as pane 1 (embedded transparent viewer, orbit/zoom any time, click any part to
// copy its reference), running a scripted replay of the ST3095 holder session: Claude's first-draft
// holder is in the wrong orientation, the visitor-side cursor deletes it in the primitive editor,
// drops in a box and scrubs it to Cameron's envelope, saves the edit log, and asks Claude to carve
// the real holder from the box. The carve plays the generator's real stages (servo pocket →
// yoke-sweep relief → 7 screw holes), baked by tools/export_editor_scene.py.
import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js?v=68c77e28da";
import { buildGeometry, loadBundle } from "./arm.js?v=68c77e28da";
import { EDITOR_HISTORY, editorPaneTemplate, el } from "./template.js?v=68c77e28da";

const BASE_URL = new URL(".", import.meta.url);

// The real session (Sep 1, ~/cad_robot2), as the blog clip replays it; the editor steps here are a
// re-enactment with his exact box.
const ASK = "can you take the 3095 servo here and add a servo holder to mount it and also a yoke on the servo horns";
const GEN_TOOL = "Bash(python gen_3095_module.py)";
const GEN_OUT = [
  "holder: vol 7.3cm3, bounds [[-36.0, -17.5, -17.5], [46.0, -13.9, 17.5]]",
  "yoke: vol 29.7cm3, bounds [[-46.1, -24.3, -26.0], [1.4, 44.3, 26.0]]",
  "holder^servo: -1.0mm3",
  "yoke^servo: -1.0mm3",
  "yoke^holder: 0.0mm3",
  "rotation: bridge r_in 23.5 vs case corner r 21.9 vs plate arc 22.0 -> CLEAR",
  "st3095_module.urdf written"
];
const DRAFT_REPLY = "First-draft module is up: blue holder, orange yoke. The holder is a face plate screwed straight into the servo's 6 tapped M2.5 holes, so no heat-set inserts. The yoke seats on both horn discs with 8 M3 screws per side, and its arc bridge clears the case through a full 360°.";
const WRONG = "the servo holder is in the wrong orientation. i'll block out the holder myself in the editor";
const WRONG_REPLY = "Sounds good. Delete my holder, drop in a box for the envelope you want, then Save to edit log and tell me.";
const CARVE = "can you look at the holder_box geometry i added in the editor, that's the rough box that the servo holder should use. carve out the servo and screw holes with appropriate margins from it";
const READ_TOOL = "Read(~/cad_editor_edits/EDIT_LOG.md)";
const READ_OUT = [
  "## 2026-10-06 14:43:01 — st3095_module.urdf",
  "DELETE st3095_holder.stl (m) (center (5.0, 0.0, 15.7)mm)",
  'CUBE "holder_box" (m): center (20.0, -0.0, -10.0)mm | rot (0.0, 0.0, 0.0)deg | size 35.4 x 39.4 x 59.4mm'
];
const CARVE_TOOL = "Bash(python gen_3095_holder_frombox.py)";
const CARVE_OUT = [
  "box (servo frame): [[2.3, -19.7, -19.7], [37.7, 39.7, 19.7]] watertight: True",
  "12 case solids in pocket (of 12)",
  "pocket built: 101.0 cm3  (case hull dilated +0.25 mm)",
  "carve 1/3 servo pocket: 83.0 -> 37.0 cm3, open toward -x (servo slides in endwise)",
  "carve 2/3 yoke-sweep relief r22.8..30.3 about the output axis: removes 5.22 cm3",
  "carve 3/3 screw holes: 4 floor + 3 ceiling, D2.8 through + D5.5 counterbore, 2.8 mm web under the heads",
  "holder: vol 30.8cm3, bounds [[6.4, -19.7, -19.7], [37.7, 39.7, 19.7]], wt=True",
  "max penetration into servo: -0.147mm",
  "holder ^ yoke (q0): 0.0mm3"
];
const CARVE_REPLY = "Done: your box envelope, carved into a working holder (30.8 cm³, watertight). The pocket is the servo case dilated 0.25 mm; your box wraps the +x half, so the servo slides in endwise. 7 screw holes go into the servo's tapped M2.5s (Ø2.8 through, Ø5.5 counterbores, 2.8 mm web): 4 in the floor, 3 in the ceiling. A relief ring keeps the yoke clear through full rotation.";

const HOLDER_STAGES = ["s0_box", "s1_pocket", "s2_sweep", "s3_holder"];
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const clamp01 = (t) => Math.max(0, Math.min(1, t));
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
  if (!options.fallbackVideo && !options.fallbackPoster) return;
  const fig = el("div", "cac-fallback");
  if (options.fallbackVideo) {
    const video = el("video");
    Object.assign(video, { src: options.fallbackVideo, poster: options.fallbackPoster || "", autoplay: true, muted: true, loop: true, playsInline: true, controls: true });
    fig.appendChild(video);
  } else {
    const img = el("img");
    Object.assign(img, { src: options.fallbackPoster, alt: "The claude-as-cad primitive editor next to a Claude Code session" });
    fig.appendChild(img);
  }
  root.appendChild(fig);
}

export async function mountEditorPane(root, options = {}) {
  if (!root) return null;
  if (!webglAvailable()) {
    showFallback(root, options);
    return null;
  }
  root.classList.add("cac-hero");
  let win = root.querySelector(".cac-window");
  if (!win) {
    win = editorPaneTemplate();
    root.innerHTML = "";
    root.appendChild(win);
  }
  const $ = (sel) => win.querySelector(sel);
  const ui = {
    viewer: $(".cac-viewer"), canvas: $(".cac-canvas"), marker: $(".cac-marker"), toast: $(".cac-toast"),
    hint: $(".cac-hint"), log: $(".cac-log"), input: $(".cac-input"), inputText: $(".cac-input-text"),
    ring: $(".cac-ring"), cursor: $(".cac-vcursor"), play: $(".cac-play"), leaders: $(".cac-leaders"),
    calloutGood: $(".cac-callout-good"), calloutBad: $(".cac-callout-bad"),
    pe: $(".cac-pe"), peBadge: $(".cac-pe-badge"), peEmpty: $(".cac-pe-empty"), peSel: $(".cac-pe-sel"),
    peName: $(".cac-pe-name-text"), peNameCaret: $(".cac-pe-caret"), peSub: $(".cac-pe-sub"), peDel: $(".cac-pe-del"),
    peAdd: $(".cac-pe-add"), peCount: $(".cac-pe-count"), peEdits: $(".cac-pe-edits"), peSave: $(".cac-pe-save"),
    peSaved: $(".cac-pe-saved"), peGroupSize: $('[data-group="size"]'), peGroupRot: $('[data-group="rot"]')
  };
  const field = (f) => win.querySelector(`[data-f="${f}"]`);
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

  // ---------------------------------------------------------------- three.js scene
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
  const camera = new THREE.PerspectiveCamera(30, 1, 0.005, 10);
  camera.up.set(0, 0, 1);
  scene.add(new THREE.HemisphereLight(0xffffff, 0xb3b8c2, 1.35));
  const key = new THREE.DirectionalLight(0xffffff, 1.9);
  key.position.set(0.25, -0.35, 0.6);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  Object.assign(key.shadow.camera, { left: -0.12, right: 0.12, top: 0.12, bottom: -0.12, near: 0.05, far: 2 });
  key.shadow.radius = 4;
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.55);
  fill.position.set(-0.4, 0.3, 0.2);
  scene.add(fill);

  const controls = new OrbitControls(camera, ui.canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.09;
  controls.enableZoom = false; // enabled after the visitor clicks into the viewer
  controls.minDistance = 0.06;
  controls.maxDistance = 1.2;

  let manifest;
  let bin;
  try {
    ({ manifest, bin } = await loadBundle(BASE_URL, "editor"));
  } catch (error) {
    console.warn("claude-as-cad editor pane: falling back", error);
    renderer.dispose();
    showFallback(root, options);
    return null;
  }

  // servo frame (the STLs) -> editor world (Z up), as the module URDF placed it in the clip
  const module = new THREE.Group();
  module.rotation.set(...manifest.worldFromServo.rpy, "ZYX");
  scene.add(module);
  const parts = new Map();
  for (const entry of manifest.meshes) {
    const cut = entry.role === "cut";
    const material = new THREE.MeshStandardMaterial({
      color: entry.color,
      roughness: entry.role === "servo" ? 0.62 : 0.5,
      metalness: entry.role === "yoke" ? 0.12 : 0.04,
      transparent: true,
      opacity: 1,
      depthWrite: !cut
    });
    const mesh = new THREE.Mesh(buildGeometry(entry, bin), material);
    mesh.name = entry.stl || entry.name;
    mesh.userData.key = entry.name;
    mesh.castShadow = !cut;
    if (entry.triMap) {
      mesh.userData.triMap = entry.triMapType === "u16" ? new Uint16Array(bin, entry.triMap[0], entry.triMap[1] / 2) : new Uint32Array(bin, entry.triMap[0], entry.triMap[1] / 4);
    }
    module.add(mesh);
    parts.set(entry.name, mesh);
  }
  const part = (name) => parts.get(name);
  const setOpacity = (mesh, a) => {
    mesh.material.opacity = a;
    mesh.visible = a > 0.01;
    mesh.material.depthWrite = mesh.userData.key.startsWith("cut") ? false : a > 0.97;
  };

  // Cameron's box: an editor cube in world space (centre/size in mm, straight from the edit log)
  const boxMaterial = new THREE.MeshStandardMaterial({ color: "#f09a45", roughness: 0.6, transparent: true, opacity: 0.55, depthWrite: false });
  const box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), boxMaterial);
  const boxEdges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), new THREE.LineBasicMaterial({ color: "#c46a17", transparent: true, opacity: 0.9 }));
  box.add(boxEdges);
  scene.add(box);
  const boxState = { center: [0, 0, 0], size: [20, 20, 20] };
  function applyBox() {
    box.position.set(...boxState.center.map((v) => v / 1000));
    box.scale.set(...boxState.size.map((v) => v / 1000));
    dirty = true;
  }

  // framing + a shadow catcher under the module
  scene.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(part("servo"));
  bounds.expandByObject(part("yoke"));
  bounds.expandByObject(part("s3_holder"));
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3()).length();
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.ShadowMaterial({ opacity: 0.12 }));
  ground.position.set(center.x, center.y, bounds.min.z - 0.0005);
  ground.receiveShadow = true;
  scene.add(ground);
  key.target.position.copy(center);
  scene.add(key.target);
  // 3/4 view; the target is pushed to screen-right so the module sits left of the editor panel
  function computeDefaultView() {
    const dir = new THREE.Vector3(0.95, -1.25, 0.72).normalize();
    const right = new THREE.Vector3().crossVectors(dir.clone().negate(), new THREE.Vector3(0, 0, 1)).normalize();
    const narrow = ui.viewer.getBoundingClientRect().width < 600;
    const target = center.clone().addScaledVector(right, size * (narrow ? 0.5 : 0.42));
    return { position: target.clone().addScaledVector(dir, size * (narrow ? 3.9 : 3.1)), target };
  }
  const defaultView = computeDefaultView();
  // once the editor panel steps aside, the carve plays centred
  function computeCarveView() {
    const v = computeDefaultView();
    const dir = v.position.clone().sub(v.target).normalize();
    return { position: center.clone().addScaledVector(dir, size * 2.7), target: center.clone() };
  }
  camera.position.copy(defaultView.position);
  controls.target.copy(defaultView.target);
  controls.update();

  // ---------------------------------------------------------------- render loop
  let visible = true;
  let rafId = 0;
  let dirty = true;
  let firstFrame = false;
  const frameHooks = new Set();
  function resize() {
    const rect = ui.viewer.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
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
  const startLoop = () => {
    if (!rafId && visible) rafId = requestAnimationFrame(frame);
  };
  new ResizeObserver(resize).observe(ui.viewer);
  resize();

  const check = (token) => {
    if (token?.aborted) throw new Aborted();
  };
  const wait = (ms, token) => new Promise((resolve, reject) => {
    const start = performance.now();
    const hook = () => {
      if (token?.aborted) {
        frameHooks.delete(hook);
        reject(new Aborted());
      } else if (performance.now() - start >= ms) {
        frameHooks.delete(hook);
        resolve();
      }
    };
    frameHooks.add(hook);
  });
  const tween = (ms, fn, token, ease = easeInOut) => new Promise((resolve, reject) => {
    const start = performance.now();
    const hook = () => {
      if (token?.aborted) {
        frameHooks.delete(hook);
        reject(new Aborted());
        return;
      }
      const t = clamp01((performance.now() - start) / ms);
      fn(ease(t), t);
      dirty = true;
      if (t >= 1) {
        frameHooks.delete(hook);
        resolve();
      }
    };
    frameHooks.add(hook);
  });
  // The visitor owns the camera once they orbit/zoom; scripted moves then just keep the beat.
  let userCamera = false;
  controls.addEventListener("start", () => { userCamera = true; });
  function cameraTo(view, ms, token) {
    if (userCamera || reduceMotion) return wait(Math.min(ms, 300), token);
    const p0 = camera.position.clone();
    const t0 = controls.target.clone();
    return tween(ms, (e) => {
      if (userCamera) return;
      camera.position.lerpVectors(p0, view.position, e);
      controls.target.lerpVectors(t0, view.target, e);
    }, token);
  }
  const fade = (mesh, to, ms, token) => {
    const from = mesh.material.opacity * (mesh.visible ? 1 : 0);
    return tween(ms, (e) => setOpacity(mesh, from + (to - from) * e), token);
  };

  // ---------------------------------------------------------------- screen-space helpers
  const winRect = () => win.getBoundingClientRect();
  function projectToWindow(world) {
    const v = world.clone().project(camera);
    const vr = ui.viewer.getBoundingClientRect();
    const wr = winRect();
    return { x: vr.left - wr.left + (v.x * 0.5 + 0.5) * vr.width, y: vr.top - wr.top + (-v.y * 0.5 + 0.5) * vr.height };
  }
  function elementPoint(node, fx = 0.5, fy = 0.5) {
    const r = node.getBoundingClientRect();
    const wr = winRect();
    return { x: r.left - wr.left + r.width * fx, y: r.top - wr.top + r.height * fy };
  }
  const inputCaretPoint = () => elementPoint(win.querySelector(".cac-caret"), 0.6, 0.7);
  // a mesh's bounding-box centre in world space (servo-frame bounds from the bake)
  function partCenterWorld(name) {
    const b = manifest.meshes.find((m) => m.name === name).bounds;
    const local = new THREE.Vector3((b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2, (b[0][2] + b[1][2]) / 2);
    module.updateMatrixWorld(true);
    return module.localToWorld(local);
  }
  function partTopWorld(name) {
    const b = manifest.meshes.find((m) => m.name === name).bounds;
    // servo frame +y is world -z (rotation −90° about X), so world-up is the servo frame's min y
    const local = new THREE.Vector3((b[0][0] + b[1][0]) / 2, b[0][1], (b[0][2] + b[1][2]) / 2);
    module.updateMatrixWorld(true);
    return module.localToWorld(local);
  }

  // callouts with leader lines, re-projected every frame
  const callouts = [
    { node: ui.calloutGood, world: () => partTopWorld("yoke"), offset: [-120, -56] },
    { node: ui.calloutBad, world: () => partCenterWorld("first"), offset: [70, 70] }
  ];
  function updateOverlays() {
    const vr = ui.viewer.getBoundingClientRect();
    const wr = winRect();
    const ox = vr.left - wr.left;
    const oy = vr.top - wr.top;
    let lines = "";
    for (const c of callouts) {
      if (c.node.hidden) continue;
      const p = projectToWindow(c.world());
      const lx = Math.max(6, Math.min(vr.width - c.node.offsetWidth - 6, p.x - ox + c.offset[0]));
      const ly = Math.max(6, Math.min(vr.height - c.node.offsetHeight - 6, p.y - oy + c.offset[1]));
      c.node.style.transform = `translate(${lx}px, ${ly}px)`;
      const ax = lx + c.node.offsetWidth / 2;
      const ay = ly + (c.offset[1] < 0 ? c.node.offsetHeight : 0);
      lines += `<line x1="${ax}" y1="${ay}" x2="${p.x - ox}" y2="${p.y - oy}" class="${c.node === ui.calloutBad ? "bad" : "good"}" style="opacity:${c.node.style.opacity || 1}"/><circle cx="${p.x - ox}" cy="${p.y - oy}" r="2.6" class="${c.node === ui.calloutBad ? "bad" : "good"}" style="opacity:${c.node.style.opacity || 1}"/>`;
    }
    ui.leaders.innerHTML = lines;
    updateVisitorClick();
  }

  // ---------------------------------------------------------------- virtual cursor
  let cursorPos = null;
  const moveCursor = () => {
    ui.cursor.style.transform = `translate(${cursorPos.x}px, ${cursorPos.y}px)`;
  };
  function showCursor(point) {
    cursorPos = { ...point };
    ui.cursor.hidden = false;
    moveCursor();
  }
  const hideCursor = () => {
    ui.cursor.hidden = true;
  };
  const cursorKind = (kind) => ui.cursor.classList.toggle("cac-vcursor-scrub", kind === "scrub");
  function glide(getTarget, token, speed = 1100) {
    const from = { ...cursorPos };
    const first = getTarget();
    const dist = Math.hypot(first.x - from.x, first.y - from.y);
    const ms = Math.max(380, Math.min(900, (dist / speed) * 1000));
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
    ring.style.left = `${point.x}px`;
    ring.style.top = `${point.y}px`;
    ring.classList.remove("cac-ring-on");
    void ring.offsetWidth;
    ring.classList.add("cac-ring-on");
  }
  async function clickAt(getTarget, token) {
    await glide(getTarget, token);
    ringAt(cursorPos);
    await wait(180, token);
  }

  // ---------------------------------------------------------------- mock Claude panel
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  function logLine(kind, html, extraClass = "") {
    const line = el("div", `cac-line cac-${kind} ${extraClass}`.trim(), html);
    ui.log.appendChild(line);
    ui.log.scrollTop = ui.log.scrollHeight;
    return line;
  }
  function resetChat() {
    ui.log.innerHTML = "";
    for (const item of EDITOR_HISTORY) logLine("meta", esc(item.text));
    ui.inputText.innerHTML = "";
    ui.input.classList.remove("cac-input-active");
  }
  function dimOlder() {
    for (const line of ui.log.querySelectorAll(".cac-user, .cac-agent, .cac-tool")) line.classList.add("cac-dim");
  }
  async function typeText(text, token, cps = 42) {
    for (const ch of text) {
      check(token);
      ui.inputText.insertAdjacentText("beforeend", ch);
      await wait(ch === " " ? 1000 / (cps * 1.8) : (1000 / cps) * (0.7 + Math.random() * 0.6), token);
    }
  }
  async function say(text, token) {
    await glide(inputCaretPoint, token);
    ui.input.classList.add("cac-input-active");
    await typeText(text, token);
    await wait(250, token);
    ui.inputText.innerHTML = "";
    ui.input.classList.remove("cac-input-active");
    dimOlder();
    logLine("user", `&gt; ${esc(text)}`);
  }
  async function think(ms, token) {
    const line = logLine("thinking", `<span class="cac-spin">·</span> Thinking…`);
    const frames = ["·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢"];
    const glyph = line.querySelector(".cac-spin");
    try {
      for (let t = 0, f = 0; t < ms; t += 110, f += 1) {
        glyph.textContent = frames[f % frames.length];
        await wait(110, token);
      }
    } finally {
      line.remove();
    }
  }
  async function tool(title, lines, token, between = {}) {
    const node = logLine("tool", `<span class="cac-bullet cac-ok">●</span> ${esc(title)}<div class="cac-tool-out"></div>`);
    const out = node.querySelector(".cac-tool-out");
    for (const [index, text] of lines.entries()) {
      out.insertAdjacentHTML("beforeend", `<div>${index === 0 ? "⎿ " : "&nbsp;&nbsp;"}${esc(text)}</div>`);
      ui.log.scrollTop = ui.log.scrollHeight;
      if (between[index]) await between[index]();
      else await wait(150, token);
    }
  }
  async function reply(text, token) {
    const node = logLine("agent", `<span class="cac-bullet">●</span> `);
    for (const word of text.split(" ")) {
      check(token);
      node.insertAdjacentText("beforeend", `${word} `);
      ui.log.scrollTop = ui.log.scrollHeight;
      await wait(30, token);
    }
  }

  // ---------------------------------------------------------------- primitive editor panel (mock)
  const fmt = (v) => (Math.abs(v) < 0.05 ? 0 : v).toFixed(Math.abs(v - Math.round(v)) < 0.05 ? 0 : 1);
  let edits = [];
  let notLogged = 0;
  function renderEdits() {
    ui.peCount.textContent = String(edits.length);
    ui.peEdits.innerHTML = edits.map((e) => `<div class="${e.cube ? "cac-pe-cube" : ""}">${e.cube ? "■ " : ""}${esc(e.text)}</div>`).join("");
    ui.peBadge.hidden = notLogged === 0 && !ui.peBadge.dataset.logged;
    ui.peBadge.textContent = notLogged ? `${notLogged} not logged` : "logged";
    ui.peBadge.classList.toggle("cac-pe-badge-ok", !notLogged);
  }
  function selectNone() {
    ui.peEmpty.hidden = false;
    ui.peSel.hidden = true;
  }
  function selectHolder() {
    ui.peEmpty.hidden = true;
    ui.peSel.hidden = false;
    ui.peName.textContent = "st3095_holder.stl";
    ui.peSub.textContent = "part · link m";
    const c = manifest.deletedHolderCenter;
    ["loc0", "loc1", "loc2"].forEach((f, i) => { field(f).textContent = fmt(c[i]); });
    ui.peGroupSize.hidden = true;
    ui.peGroupRot.hidden = false;
    ui.peDel.textContent = "Delete part";
  }
  function selectCube(name) {
    ui.peEmpty.hidden = true;
    ui.peSel.hidden = false;
    ui.peName.textContent = name;
    ui.peSub.textContent = "cube · parent: m";
    ui.peGroupSize.hidden = false;
    ui.peGroupRot.hidden = false;
    ui.peDel.textContent = "Delete cube";
    showBoxFields();
  }
  function showBoxFields() {
    boxState.center.forEach((v, i) => { field(`loc${i}`).textContent = fmt(v); });
    boxState.size.forEach((v, i) => { field(`size${i}`).textContent = fmt(v); });
  }
  function resetEditorPanel() {
    edits = [];
    notLogged = 0;
    delete ui.peBadge.dataset.logged;
    renderEdits();
    selectNone();
    ui.pe.classList.remove("cac-pe-gone");
    ui.peSaved.hidden = true;
    ui.peNameCaret.hidden = true;
    for (const node of win.querySelectorAll(".cac-pe-active")) node.classList.remove("cac-pe-active");
  }
  // drag a number field left/right like the editor (Blender-style), overshoot then settle
  async function scrub(fieldName, to, token, apply) {
    const node = field(fieldName);
    await glide(() => elementPoint(node, 0.5, 0.55), token);
    cursorKind("scrub");
    node.classList.add("cac-pe-active");
    const from = parseFloat(node.textContent) || 0;
    const startX = cursorPos.x;
    const span = Math.max(-60, Math.min(60, (to - from) * 1.6));
    const overshoot = (to - from) * 0.06;
    await tween(Math.min(1100, 420 + Math.abs(to - from) * 14), (e, t) => {
      const settle = t < 0.8 ? e / 0.8 : 1;
      const v = from + (to - from) * Math.min(1, settle) + overshoot * Math.sin(Math.PI * clamp01((t - 0.55) / 0.45));
      node.textContent = fmt(t >= 1 ? to : v);
      apply(t >= 1 ? to : v);
      cursorPos = { x: startX + span * Math.min(1, settle), y: cursorPos.y };
      moveCursor();
    }, token, (t) => t);
    node.classList.remove("cac-pe-active");
    cursorKind("arrow");
  }

  // ---------------------------------------------------------------- scene states
  function resetScene() {
    for (const [name, mesh] of parts) setOpacity(mesh, name === "servo" ? 1 : 0);
    box.visible = false;
    boxMaterial.opacity = 0.55;
    boxEdges.material.opacity = 0.9;
    part("first").material.emissive.set(0x000000);
    ui.calloutGood.hidden = true;
    ui.calloutBad.hidden = true;
    resetEditorPanel();
    dirty = true;
  }
  function finalState() {
    resetScene();
    setOpacity(part("yoke"), 1);
    setOpacity(part("s3_holder"), 1);
    resetChat();
    dimOlder();
    for (const [kind, text] of [["user", ASK], ["agent", DRAFT_REPLY], ["user", WRONG], ["agent", WRONG_REPLY]]) {
      logLine(kind, kind === "user" ? `&gt; ${esc(text)}` : `<span class="cac-bullet">●</span> ${esc(text)}`, "cac-dim");
    }
    logLine("user", `&gt; ${esc(CARVE)}`);
    logLine("tool", `<span class="cac-bullet cac-ok">●</span> ${esc(CARVE_TOOL)}<div class="cac-tool-out"><div>⎿ ${esc(CARVE_OUT[6])}</div></div>`);
    logLine("agent", `<span class="cac-bullet">●</span> ${esc(CARVE_REPLY)}`);
    edits = [{ text: "st3095_holder.stl (m) — deleted" }, { text: "holder_box (m)", cube: true }];
    ui.peBadge.dataset.logged = "1";
    renderEdits();
    ui.pe.classList.add("cac-pe-gone");
    hideCursor();
  }

  // ---------------------------------------------------------------- the replay
  let replayToken = null;
  async function replayOnce(token) {
    userCamera = false;
    resetScene();
    resetChat();
    showCursor(inputCaretPoint());
    await wait(500, token);

    // 1. Claude's first draft: holder in the wrong orientation
    await say(ASK, token);
    await think(900, token);
    await tool(GEN_TOOL, GEN_OUT, token, {
      2: async () => {
        await Promise.all([fade(part("yoke"), 1, 600, token), fade(part("first"), 1, 600, token)]);
      }
    });
    await reply(DRAFT_REPLY, token);
    await wait(500, token);
    for (const c of [ui.calloutGood, ui.calloutBad]) {
      c.hidden = false;
      c.style.opacity = "0";
      await tween(300, (e) => { c.style.opacity = String(e); }, token);
      await wait(350, token);
    }
    await wait(1300, token);

    // 2. "wrong orientation — I'll block it out myself"
    await say(WRONG, token);
    await think(600, token);
    await reply(WRONG_REPLY, token);
    await tween(300, (e) => {
      ui.calloutGood.style.opacity = ui.calloutBad.style.opacity = String(1 - e);
    }, token);
    ui.calloutGood.hidden = ui.calloutBad.hidden = true;
    await wait(300, token);

    // 3. in the editor: select + delete the holder, add a box, scrub it to the envelope, save
    await clickAt(() => projectToWindow(partCenterWorld("first")), token);
    part("first").material.emissive.set("#2a6fd6");
    part("first").material.emissiveIntensity = 0.35;
    selectHolder();
    await wait(700, token);
    await clickAt(() => elementPoint(ui.peDel), token);
    await fade(part("first"), 0, 380, token);
    edits.push({ text: "st3095_holder.stl (m) — deleted" });
    notLogged += 1;
    renderEdits();
    selectNone();
    await wait(350, token);
    await clickAt(() => elementPoint(ui.peAdd), token);
    // "+ Add cube" spawns a 20 mm cube at the selected part's centre (editor behaviour)
    boxState.center = [...manifest.deletedHolderCenter];
    boxState.size = [20, 20, 20];
    applyBox();
    box.visible = true;
    await tween(260, (e) => { box.scale.setScalar(0.02 * (0.6 + 0.4 * e)); }, token);
    applyBox();
    edits.push({ text: "cube1 (m)", cube: true });
    notLogged += 1;
    renderEdits();
    selectCube("cube1");
    await wait(400, token);
    const target = manifest.box;
    for (let i = 0; i < 3; i += 1) {
      await scrub(`size${i}`, target.size[i], token, (v) => { boxState.size[i] = v; applyBox(); });
    }
    for (const i of [0, 2]) {
      await scrub(`loc${i}`, target.center[i], token, (v) => { boxState.center[i] = v; applyBox(); });
    }
    // rename it
    await clickAt(() => elementPoint(ui.peName, 0.3), token);
    ui.peNameCaret.hidden = false;
    ui.peName.textContent = "";
    for (const ch of "holder_box") {
      ui.peName.textContent += ch;
      await wait(55, token);
    }
    ui.peNameCaret.hidden = true;
    edits[1].text = "holder_box (m)";
    renderEdits();
    await wait(300, token);
    await clickAt(() => elementPoint(ui.peSave), token);
    notLogged = 0;
    ui.peBadge.dataset.logged = "1";
    renderEdits();
    ui.peSaved.hidden = false;
    await wait(1000, token);
    ui.pe.classList.add("cac-pe-gone"); // the agent opens the viewer: the editor panel steps aside
    const toCarve = cameraTo(computeCarveView(), 1200, token);

    // 4. Claude carves the real holder from the box
    await say(CARVE, token);
    await think(900, token);
    await tool(READ_TOOL, READ_OUT, token);
    const s = (n) => part(n);
    await tool(CARVE_TOOL, CARVE_OUT, token, {
      0: async () => {
        await Promise.all([
          tween(600, (e) => { boxMaterial.opacity = 0.55 * (1 - e); boxEdges.material.opacity = 0.9 * (1 - e); }, token),
          fade(s("s0_box"), 1, 600, token)
        ]);
        box.visible = false;
      },
      2: async () => {
        setOpacity(s("cut_pocket"), 0);
        await Promise.all([fade(s("cut_pocket"), 0.75, 450, token), fade(s("s0_box"), 0.35, 450, token)]);
        await wait(500, token);
      },
      3: async () => {
        setOpacity(s("s1_pocket"), 0.35);
        setOpacity(s("s0_box"), 0);
        await Promise.all([fade(s("cut_pocket"), 0, 500, token), fade(s("s1_pocket"), 1, 500, token)]);
        await wait(250, token);
      },
      4: async () => {
        await Promise.all([fade(s("cut_annulus"), 0.75, 400, token), fade(s("s1_pocket"), 0.45, 400, token)]);
        await wait(350, token);
        setOpacity(s("s2_sweep"), 0.45);
        setOpacity(s("s1_pocket"), 0);
        await Promise.all([fade(s("cut_annulus"), 0, 450, token), fade(s("s2_sweep"), 1, 450, token)]);
      },
      5: async () => {
        await fade(s("s2_sweep"), 0.5, 300, token);
        for (let k = 0; k < 7; k += 1) {
          await fade(s(`cut_hole${k}`), 0.9, 160, token);
          await wait(90, token);
        }
        setOpacity(s("s3_holder"), 0.5);
        setOpacity(s("s2_sweep"), 0);
        await Promise.all([
          fade(s("s3_holder"), 1, 500, token),
          ...Array.from({ length: 7 }, (_, k) => fade(s(`cut_hole${k}`), 0, 500, token))
        ]);
      }
    });
    await toCarve;
    await reply(CARVE_REPLY, token);
    hideCursor();
    await wait(4200, token);
    // loop reset: fade back to the bare servo, camera back beside the editor panel
    await Promise.all([fade(s("s3_holder"), 0, 600, token), fade(part("yoke"), 0, 600, token), cameraTo(computeDefaultView(), 900, token)]);
  }

  let running = false;
  async function runReplayLoop({ once = false } = {}) {
    if (replayToken) replayToken.aborted = true;
    const token = { aborted: false };
    replayToken = token;
    running = true;
    try {
      do {
        await replayOnce(token);
      } while (!once && !token.aborted);
      if (once) finalState();
    } catch (error) {
      if (!(error instanceof Aborted)) throw error;
    } finally {
      if (replayToken === token) running = false;
    }
  }
  function stopReplay() {
    if (replayToken) replayToken.aborted = true;
    replayToken = null;
    running = false;
  }

  // ---------------------------------------------------------------- visitor click-to-reference
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
  function placeToast(toast, world) {
    if (!world || toast.hidden) return;
    const p = projectToWindow(world);
    const vr = ui.viewer.getBoundingClientRect();
    const wr = winRect();
    const localX = p.x - (vr.left - wr.left);
    const localY = p.y - (vr.top - wr.top);
    const tw = toast.offsetWidth;
    const th = toast.offsetHeight;
    toast.style.left = `${Math.max(8, Math.min(vr.width - tw - 8, localX - tw * 0.35))}px`;
    const below = localY + 18;
    toast.style.top = `${Math.max(8, Math.min(vr.height - th - 8, below + th < vr.height - 8 ? below : localY - th - 18))}px`;
  }
  function updateVisitorClick() {
    if (!visitor.mesh) return;
    const world = visitor.mesh.localToWorld(visitor.local.clone());
    const p = projectToWindow(world);
    const vr = ui.viewer.getBoundingClientRect();
    const wr = winRect();
    visitor.marker.style.transform = `translate(${p.x - (vr.left - wr.left)}px, ${p.y - (vr.top - wr.top)}px)`;
    placeToast(visitor.toast, world);
  }
  function triPlane(mesh, t, n, cen) {
    const pos = mesh.geometry.attributes.position;
    const idx = mesh.geometry.index.array;
    const a = new THREE.Vector3().fromBufferAttribute(pos, idx[3 * t]);
    const b = new THREE.Vector3().fromBufferAttribute(pos, idx[3 * t + 1]);
    const c = new THREE.Vector3().fromBufferAttribute(pos, idx[3 * t + 2]);
    n.subVectors(c, b).cross(new THREE.Vector3().subVectors(a, b)).normalize();
    cen.copy(a).add(b).add(c).multiplyScalar(1 / 3);
  }
  function coplanarRegion(mesh, t0) {
    if (!mesh.userData.cornerTris) {
      const pos = mesh.geometry.attributes.position;
      const idx = mesh.geometry.index.array;
      const keyOf = new Map();
      const corner = new Uint32Array(idx.length);
      for (let c = 0; c < idx.length; c += 1) {
        const v = idx[c];
        const k = `${Math.round(pos.getX(v) * 1e5)},${Math.round(pos.getY(v) * 1e5)},${Math.round(pos.getZ(v) * 1e5)}`;
        let id = keyOf.get(k);
        if (id === undefined) {
          id = keyOf.size;
          keyOf.set(k, id);
        }
        corner[c] = id;
      }
      const cornerTris = Array.from({ length: keyOf.size }, () => []);
      for (let c = 0; c < corner.length; c += 1) cornerTris[corner[c]].push((c / 3) | 0);
      Object.assign(mesh.userData, { corner, cornerTris });
    }
    const { corner, cornerTris } = mesh.userData;
    const n0 = new THREE.Vector3();
    const c0 = new THREE.Vector3();
    triPlane(mesh, t0, n0, c0);
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
          triPlane(mesh, u, n, cen);
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
    scene.updateMatrixWorld(true);
    const pickables = [...parts.values()].filter((m) => m.userData.triMap && m.visible && m.material.opacity > 0.5);
    const hit = pickRay.intersectObjects(pickables, false)[0];
    clearVisitorClick();
    if (!hit) return;
    const mesh = hit.object;
    const region = coplanarRegion(mesh, hit.faceIndex);
    const local = mesh.worldToLocal(hit.point.clone());
    const nWorld = region.normal.clone().transformDirection(mesh.matrixWorld);
    const w = hit.point.clone().multiplyScalar(1000);
    const l = local.clone().multiplyScalar(1000);
    const comps = [nWorld.x, nWorld.y, nWorld.z];
    let k = 0;
    for (let j = 1; j < 3; j += 1) if (Math.abs(comps[j]) > Math.abs(comps[k])) k = j;
    const f1 = (v) => (Math.abs(v) < 0.05 ? 0 : v).toFixed(1);
    const f2 = (v) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2);
    const tag = `[${mesh.name} in m @ (${f1(w.x)}, ${f1(w.y)}, ${f1(w.z)})mm | local (${f1(l.x)}, ${f1(l.y)}, ${f1(l.z)})mm n(${[region.normal.x, region.normal.y, region.normal.z].map(f2).join(", ")}) tri ${mesh.userData.triMap[hit.faceIndex]}, ${comps[k] >= 0 ? "+" : "-"}${"XYZ"[k]} side face]`;
    // highlight the flat face
    const pos = mesh.geometry.attributes.position;
    const idx = mesh.geometry.index.array;
    const out = new Float32Array(region.tris.length * 9);
    const lift = region.normal.clone().multiplyScalar(0.00015);
    const v = new THREE.Vector3();
    region.tris.forEach((t, i) => {
      for (let c = 0; c < 3; c += 1) {
        v.fromBufferAttribute(pos, idx[3 * t + c]).add(lift);
        out.set([v.x, v.y, v.z], i * 9 + c * 3);
      }
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(out, 3));
    const hl = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: "#22c4f0", transparent: true, opacity: 0.85, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }));
    hl.renderOrder = 2;
    mesh.add(hl);
    Object.assign(visitor, { mesh, local, highlight: hl });
    const wr = winRect();
    ringAt({ x: clientX - wr.left, y: clientY - wr.top }, visitor.ring);
    visitor.marker.hidden = false;
    visitorTitle.textContent = "Copied to clipboard:";
    visitorBody.textContent = tag;
    visitor.toast.hidden = false;
    updateVisitorClick();
    dirty = true;
    try {
      navigator.clipboard.writeText(tag).catch(() => { visitorTitle.textContent = "Face reference:"; });
    } catch {
      visitorTitle.textContent = "Face reference:";
    }
    const gen = visitor.gen;
    setTimeout(() => {
      if (gen !== visitor.gen) return;
      visitor.toast.classList.add("cac-fade");
      visitor.marker.classList.add("cac-fade");
      tween(400, (e) => { hl.material.opacity = 0.85 * (1 - e); }, null).catch(() => {});
      setTimeout(() => { if (gen === visitor.gen) clearVisitorClick(); }, 450);
    }, 1000);
  }

  // ---------------------------------------------------------------- interaction
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
  ui.canvas.addEventListener("pointerleave", () => {
    controls.enableZoom = false;
  });
  ui.play.addEventListener("click", () => {
    ui.play.hidden = true;
    runReplayLoop({ once: true }).then(() => {
      ui.play.hidden = false;
    });
  });
  const io = new IntersectionObserver((entries) => {
    const nowVisible = entries.some((e) => e.isIntersecting);
    if (nowVisible === visible) return;
    visible = nowVisible;
    if (visible) {
      startLoop();
      if (!reduceMotion && !running) runReplayLoop();
    } else if (running && !reduceMotion) {
      stopReplay();
      frameHooks.clear();
      finalState();
    }
  }, { threshold: 0.15 });
  io.observe(win);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && running && !reduceMotion) {
      stopReplay();
      frameHooks.clear();
      finalState();
    } else if (!document.hidden && visible && !reduceMotion && !running) {
      runReplayLoop();
    }
  });

  if (options.debug) {
    window.__cacEditor = { THREE, camera, controls, parts, box, projectToWindow, partCenterWorld, get running() { return running; } };
  }
  startLoop();
  if (reduceMotion || options.autoplay === false) {
    finalState();
    if (reduceMotion) ui.play.hidden = false;
  } else {
    runReplayLoop();
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
