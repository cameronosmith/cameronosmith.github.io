// claude-as-cad interactive hero.
//
//   import { mountClaudeCadHero } from "./hero/hero.js";
//   mountClaudeCadHero(document.querySelector("#cac-hero"), { fallbackVideo, fallbackPoster });
//
// Two beats:
//  1. The gripper TCP follows the visitor's cursor (damped-least-squares IK on the real arm).
//  2. A scripted replay of click-to-reference: a virtual cursor clicks two faces, a mock
//     Claude Code panel types "join the faces of … with the face …" with ⌘V pastes, and the trapezoidal
//     connector grows between the faces. It is labelled as a scripted replay in the UI.
import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { loadArm } from "./arm.js?v=36b142ff4a";
import { HISTORY, el, template } from "./template.js?v=36b142ff4a";

const BASE_URL = new URL(".", import.meta.url);

const TOOL_CALL = "Bash(Loft J4 connector on the 90deg yoke)";
const TOOL_OUTPUT = [
  "yoke face y=24.00 x[-13.0,13.0] z[-29.3,11.6]",
  "holder bottom y=43.91 x[-12.4,11.7] z[-24.0,6.0]",
  "conn: vol 18cm3 span y[24.0,43.9] wt=True"
];
const REPLY = "Joined the faces with our normal trapezoidal connector — an 8-corner hull lofted from the yoke face to the holder face. Flush at both faces (0.0 mm³ interference), clears the servo. It's live in the viewer.";

const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const easeOut = (t) => 1 - Math.pow(1 - t, 3);
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
    Object.assign(img, { src: options.fallbackPoster, alt: "The claude-as-cad viewer next to a Claude Code session" });
    fig.appendChild(img);
  }
  root.appendChild(fig);
}

export async function mountClaudeCadHero(root, options = {}) {
  if (!root) return null;
  if (!webglAvailable()) {
    showFallback(root, options);
    return null;
  }
  root.classList.add("cac-hero");
  let win = root.querySelector(".cac-window");
  if (!win) {
    win = template();
    root.innerHTML = "";
    root.appendChild(win);
  }
  const $ = (sel) => win.querySelector(sel);
  const ui = {
    viewer: $(".cac-viewer"), canvas: $(".cac-canvas"), marker: $(".cac-marker"), toast: $(".cac-toast"),
    toastBody: $(".cac-toast-body"), chips: [...win.querySelectorAll("[data-chip]")], hint: $(".cac-hint"),
    loading: $(".cac-loading"), log: $(".cac-log"), input: $(".cac-input"), inputText: $(".cac-input-text"),
    keycap: $(".cac-keycap"), ring: $(".cac-ring"), cursor: $(".cac-vcursor"), play: $(".cac-play"), panel: $(".cac-panel")
  };
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  // Cursor-follow after the replay is opt-in: on the page the intro viewer does the following, and
  // this pane just loops its replay so a visitor arriving late still sees the whole thing.
  const FOLLOW = options.follow === true;

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
  renderer.localClippingEnabled = true;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 20);
  camera.up.set(0, 0, 1);
  scene.add(new THREE.HemisphereLight(0xffffff, 0xb3b8c2, 1.35));
  const key = new THREE.DirectionalLight(0xffffff, 1.9);
  key.position.set(0.6, -0.8, 1.4);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  Object.assign(key.shadow.camera, { left: -0.5, right: 0.5, top: 0.5, bottom: -0.5, near: 0.1, far: 4 });
  key.shadow.radius = 4;
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.55);
  fill.position.set(-1, 0.6, 0.5);
  scene.add(fill);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), new THREE.ShadowMaterial({ opacity: 0.13 }));
  ground.receiveShadow = true;
  scene.add(ground);

  const controls = new OrbitControls(camera, ui.canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.09;
  controls.enableZoom = false; // enabled after the visitor clicks into the viewer
  controls.minDistance = 0.12;
  controls.maxDistance = 3;

  let arm;
  try {
    arm = await loadArm(BASE_URL);
  } catch (error) {
    console.warn("claude-as-cad hero: falling back", error);
    renderer.dispose();
    showFallback(root, options);
    return null;
  }
  ui.loading?.remove();
  scene.add(arm.root);

  // Growing connector: front faces in orange, back faces flat-filled so the clipped solid reads as solid.
  const connector = arm.meshes.get("link4_conn_ce8.stl");
  const clipPlane = new THREE.Plane();
  connector.material.clippingPlanes = [clipPlane];
  connector.material.emissive = new THREE.Color("#f28a2b");
  connector.material.emissiveIntensity = 0;
  const connectorBack = new THREE.Mesh(connector.geometry, new THREE.MeshBasicMaterial({ color: "#c46714", side: THREE.BackSide, clippingPlanes: [clipPlane] }));
  connector.add(connectorBack);
  const link4 = arm.links.get(arm.targets.link);
  const baseLink = arm.links.get("base");
  const [loft0, loft1] = arm.targets.range;
  let connectorProgress = 0;
  // Clipping planes are world-space, but the cut lives in link_4's frame: re-derive the world
  // plane from link_4's current pose before every render, or the cut stays behind when the arm
  // moves (e.g. during TCP-follow) and slices the finished connector away.
  const clipNormal = new THREE.Vector3(0, -1, 0);
  const updateClipPlane = () => {
    const cut = loft0 + (loft1 - loft0) * connectorProgress;
    clipPlane.set(clipNormal, connectorProgress >= 0.999 ? cut + 1 : cut); // fully built: never clip
    link4.updateMatrixWorld(true);
    clipPlane.applyMatrix4(link4.matrixWorld);
  };
  const setConnectorProgress = (t) => {
    connectorProgress = t;
    connector.visible = t > 0.001;
    updateClipPlane();
  };
  const highlights = arm.targets.faces.map((face) => arm.meshes.get(face.highlight));
  const setHighlight = (i, opacity) => {
    highlights[i].visible = opacity > 0.01;
    highlights[i].material.opacity = opacity;
  };

  // Home pose = the URDF default (J4 yoke at its 90° clocking, as in the clip).
  const HOME = arm.getPose();
  const REST = { ...HOME };

  // ---------------------------------------------------------------- framing
  const box = new THREE.Box3();
  arm.root.traverse((o) => {
    if (o.isMesh && o.userData.role !== "highlight") box.expandByObject(o);
  });
  const armCenter = box.getCenter(new THREE.Vector3());
  const armSize = box.getSize(new THREE.Vector3()).length();
  const defaultDir = new THREE.Vector3(1.0, -1.35, 0.72).normalize();
  const defaultView = {
    target: armCenter.clone().add(new THREE.Vector3(0, 0, -0.02)),
    position: armCenter.clone().addScaledVector(defaultDir, armSize * 1.3)
  };
  camera.position.copy(defaultView.position);
  controls.target.copy(defaultView.target);

  const faceWorld = (i, out = new THREE.Vector3()) => arm.linkWorldPoint(arm.targets.link, new THREE.Vector3().fromArray(arm.targets.faces[i].point), out);
  const faceNormalWorld = (i) => {
    const q = new THREE.Quaternion();
    link4.getWorldQuaternion(q);
    return new THREE.Vector3().fromArray(arm.targets.faces[i].normal).applyQuaternion(q).normalize();
  };
  function replayShots() {
    // The connector gap is a ~20 mm slot between a face pointing one way (n1) and a face pointing
    // back (−n1), so each face is only visible from the side at a shallow angle into the slot.
    arm.root.updateMatrixWorld(true);
    const p1 = faceWorld(0);
    const p2 = faceWorld(1);
    const mid = p1.clone().add(p2).multiplyScalar(0.5);
    const n1 = faceNormalWorld(0);
    let base = new THREE.Vector3().subVectors(defaultView.position, mid);
    base.addScaledVector(n1, -base.dot(n1));
    if (base.lengthSq() < 1e-8) base = new THREE.Vector3(1, 0, 0);
    base.normalize();
    const rad = THREE.MathUtils.degToRad;
    // Orbit about n1 by `rotDeg` from the default view's side, then tilt `elevDeg` along n1; the
    // camera sits `dist` from the face but aims partway (`pull`) toward the arm's centre, so the
    // shot keeps the angled look into the slot while most of the arm stays in frame.
    const shot = (rotDeg, elevDeg, dist, focus, pull) => {
      const side = base.clone().applyQuaternion(new THREE.Quaternion().setFromAxisAngle(n1, rad(rotDeg)));
      return {
        position: focus.clone().add(side.multiplyScalar(Math.cos(rad(elevDeg))).addScaledVector(n1, Math.sin(rad(elevDeg))).multiplyScalar(dist)),
        target: focus.clone().lerp(armCenter, pull)
      };
    };
    // Phone-width viewers get tighter shots, or the 20 mm faces would be a few pixels.
    const narrow = ui.viewer.getBoundingClientRect().width < 600;
    const d = narrow ? 0.62 : 1;
    const pull = narrow ? 0.3 : 0.5;
    return {
      face1: shot(-22, 17, 0.7 * d, p1.clone().lerp(mid, 0.35), pull),
      face2: shot(-22, -15, 0.7 * d, p2.clone().lerp(mid, 0.35), pull),
      // 3/4 view so the trapezoid's taper reads as it fills (like the reference clip)
      grow: shot(-45, 15, 0.75 * d, mid.clone(), pull)
    };
  }

  // ---------------------------------------------------------------- render loop
  let visible = true;
  let rafId = 0;
  let lastTime = performance.now();
  let firstFrame = false;
  let dirty = true;
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
    const dt = Math.min(0.05, (now - lastTime) / 1000);
    lastTime = now;
    const animating = frameHooks.size > 0;
    for (const hook of [...frameHooks]) hook(dt, now);
    const following = updateFollow(dt);
    const orbiting = controls.update();
    // Render only when something changed, so an idle hero costs almost nothing.
    if (!(animating || following || orbiting || dirty || !firstFrame)) {
      if (visible) rafId = requestAnimationFrame(frame);
      return;
    }
    dirty = false;
    updateClipPlane();
    renderer.render(scene, camera);
    if (!firstFrame) {
      firstFrame = true;
      ui.viewer.style.setProperty("--cac-poster", "none"); // live render replaces the loader's poster
    }
    updateMarker();
    if (visible) rafId = requestAnimationFrame(frame);
  }
  const startLoop = () => {
    if (!rafId && visible) {
      lastTime = performance.now();
      rafId = requestAnimationFrame(frame);
    }
  };
  new ResizeObserver(resize).observe(ui.viewer);
  resize();

  // Promise helpers bound to the loop; every await checks the replay's abort signal.
  let replayToken = null;
  const check = (token) => {
    if (token.aborted) throw new Aborted();
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
      if (t >= 1) {
        frameHooks.delete(hook);
        resolve();
      }
    };
    frameHooks.add(hook);
  });
  // Set once the visitor orbits/zooms during a replay: from then on the replay leaves the camera
  // to them and its cursor/markers simply track wherever the faces are on screen.
  let userCamera = false;
  function cameraTo(view, ms, token) {
    if (userCamera && token) return wait(Math.min(ms, 400), token); // keep the beat, skip the move
    if (reduceMotion) {
      camera.position.copy(view.position);
      controls.target.copy(view.target);
      return Promise.resolve();
    }
    const p0 = camera.position.clone();
    const t0 = controls.target.clone();
    return tween(ms, (e) => {
      if (userCamera && token) return; // the visitor grabbed the camera mid-move
      camera.position.lerpVectors(p0, view.position, e);
      controls.target.lerpVectors(t0, view.target, e);
    }, token);
  }

  // ---------------------------------------------------------------- screen-space helpers
  const winRect = () => win.getBoundingClientRect();
  function projectToWindow(world) {
    const v = world.clone().project(camera);
    const vr = ui.viewer.getBoundingClientRect();
    const wr = winRect();
    return { x: vr.left - wr.left + (v.x * 0.5 + 0.5) * vr.width, y: vr.top - wr.top + (-v.y * 0.5 + 0.5) * vr.height };
  }
  function inputCaretPoint() {
    const caret = win.querySelector(".cac-caret").getBoundingClientRect();
    const wr = winRect();
    return { x: caret.left - wr.left + 4, y: caret.top - wr.top + caret.height * 0.7 };
  }
  let markerWorld = null;
  let toastWorld = null;
  function placeToast(toast = ui.toast, world = toastWorld) {
    if (!world || toast.hidden) return;
    const p = projectToWindow(world);
    const vr = ui.viewer.getBoundingClientRect();
    const wr = winRect();
    const localX = p.x - (vr.left - wr.left);
    const localY = p.y - (vr.top - wr.top);
    const tw = toast.offsetWidth;
    const th = toast.offsetHeight;
    const left = Math.max(8, Math.min(vr.width - tw - 8, localX - tw * 0.35));
    const below = localY + 18;
    const top = below + th < vr.height - 8 ? below : localY - th - 18;
    toast.style.left = `${left}px`;
    toast.style.top = `${Math.max(8, Math.min(vr.height - th - 8, top))}px`;
  }
  function placeMarker(marker, world) {
    const p = projectToWindow(world);
    const vr = ui.viewer.getBoundingClientRect();
    const wr = winRect();
    marker.style.transform = `translate(${p.x - (vr.left - wr.left)}px, ${p.y - (vr.top - wr.top)}px)`;
  }
  function updateMarker() {
    placeToast();
    updateVisitorClick();
    if (!markerWorld) return;
    const p = projectToWindow(markerWorld);
    const vr = ui.viewer.getBoundingClientRect();
    const wr = winRect();
    ui.marker.style.transform = `translate(${p.x - (vr.left - wr.left)}px, ${p.y - (vr.top - wr.top)}px)`;
  }

  // ---------------------------------------------------------------- virtual cursor
  let cursorPos = null;
  function showCursor(point) {
    cursorPos = { ...point };
    ui.cursor.hidden = false;
    ui.cursor.style.transform = `translate(${cursorPos.x}px, ${cursorPos.y}px)`;
  }
  function hideCursor() {
    ui.cursor.hidden = true;
  }
  function glide(getTarget, token, speed = 1100) {
    const from = { ...cursorPos };
    const first = getTarget();
    const dist = Math.hypot(first.x - from.x, first.y - from.y);
    const ms = Math.max(420, Math.min(950, (dist / speed) * 1000));
    const bend = Math.min(60, dist * 0.18);
    return tween(ms, (e) => {
      const to = getTarget();
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const len = Math.hypot(dx, dy) || 1;
      const arc = Math.sin(Math.PI * e) * bend;
      cursorPos = { x: from.x + dx * e + (-dy / len) * arc, y: from.y + dy * e + (dx / len) * arc };
      ui.cursor.style.transform = `translate(${cursorPos.x}px, ${cursorPos.y}px)`;
    }, token);
  }
  function ringAt(point, ring = ui.ring) {
    ring.style.left = `${point.x}px`;
    ring.style.top = `${point.y}px`;
    ring.classList.remove("cac-ring-on");
    void ring.offsetWidth;
    ring.classList.add("cac-ring-on");
  }

  // ---------------------------------------------------------------- reference tags (live from geometry)
  // Same format as the viewer's click-to-reference: base-frame point, STL-local point + normal,
  // STL triangle index and the face's side in the base frame.
  function formatTag(meshName, linkName, world, localMm, localNormal, nWorld, triIndex) {
    const base = baseLink.worldToLocal(world.clone()).multiplyScalar(1000);
    const qb = new THREE.Quaternion();
    baseLink.getWorldQuaternion(qb);
    const nBase = nWorld.clone().applyQuaternion(qb.invert());
    const comps = [nBase.x, nBase.y, nBase.z];
    let k = 0;
    for (let j = 1; j < 3; j += 1) if (Math.abs(comps[j]) > Math.abs(comps[k])) k = j;
    const sideLabel = `${comps[k] >= 0 ? "+" : "-"}${"XYZ"[k]} side face`;
    const f1 = (v) => (Math.abs(v) < 0.05 ? 0 : v).toFixed(1);
    const f2 = (v) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2);
    return `[${meshName} in ${linkName} @ (${f1(base.x)}, ${f1(base.y)}, ${f1(base.z)})mm | local (${f1(localMm.x)}, ${f1(localMm.y)}, ${f1(localMm.z)})mm n(${[localNormal.x, localNormal.y, localNormal.z].map(f2).join(", ")}) tri ${triIndex}, ${sideLabel}]`;
  }
  function tagFor(i) {
    const face = arm.targets.faces[i];
    arm.root.updateMatrixWorld(true);
    const local = new THREE.Vector3().fromArray(face.point).sub(new THREE.Vector3().fromArray(face.visualOrigin)).multiplyScalar(1000);
    return formatTag(face.mesh, arm.targets.link, faceWorld(i), local, new THREE.Vector3().fromArray(face.normal), faceNormalWorld(i), face.triIndex);
  }

  // ---------------------------------------------------------------- mock Claude panel
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  function logLine(kind, html, extraClass = "") {
    const line = el("div", `cac-line cac-${kind} ${extraClass}`.trim(), html);
    ui.log.appendChild(line);
    ui.log.scrollTop = ui.log.scrollHeight;
    return line;
  }
  function resetPanel() {
    ui.log.innerHTML = "";
    for (const item of HISTORY) {
      if (item.kind === "meta") logLine("meta", esc(item.text));
      if (item.kind === "user") logLine("user", `&gt; ${esc(item.text)}`, "cac-dim");
      if (item.kind === "agent") logLine("agent", `<span class="cac-bullet">●</span> ${esc(item.text)}`, "cac-dim");
    }
    ui.inputText.innerHTML = "";
    ui.input.classList.remove("cac-input-active");
  }
  function promptHtml(tags) {
    return `join the faces of <span class="cac-ref">${esc(tags[0])}</span> with the face <span class="cac-ref">${esc(tags[1])}</span>`;
  }
  function finishPanel(tags) {
    resetPanel();
    logLine("user", `&gt; ${promptHtml(tags)}`);
    logLine("tool", `<span class="cac-bullet cac-ok">●</span> ${esc(TOOL_CALL)}<div class="cac-tool-out">${TOOL_OUTPUT.map((l, i) => `<div>${i === 0 ? "⎿ " : "&nbsp;&nbsp;"}${esc(l)}</div>`).join("")}</div>`);
    logLine("agent", `<span class="cac-bullet">●</span> ${esc(REPLY)}`);
  }
  async function typeText(text, token, cps = 26) {
    for (const ch of text) {
      check(token);
      ui.inputText.insertAdjacentText("beforeend", ch);
      await wait(ch === " " ? 1000 / (cps * 1.8) : (1000 / cps) * (0.7 + Math.random() * 0.6), token);
    }
  }
  async function pasteTag(tag, token) {
    ui.keycap.hidden = false;
    ui.keycap.classList.remove("cac-key-press");
    void ui.keycap.offsetWidth;
    ui.keycap.classList.add("cac-key-press");
    await wait(330, token);
    ui.inputText.insertAdjacentHTML("beforeend", `<span class="cac-ref">${esc(tag)}</span>`);
    ui.input.scrollLeft = ui.input.scrollWidth;
    lightChip("chat");
    await wait(380, token);
    ui.keycap.hidden = true;
  }
  function lightChip(name) {
    ui.chips.find((c) => c.dataset.chip === name)?.classList.add("cac-chip-on");
  }
  function resetChips() {
    for (const chip of ui.chips) chip.classList.remove("cac-chip-on");
  }

  // ---------------------------------------------------------------- click a face
  async function clickFace(i, token, tags) {
    // Re-projected every frame, so the click lands on the face even while the visitor orbits.
    const target = () => {
      const p = projectToWindow(faceWorld(i));
      const vr = ui.viewer.getBoundingClientRect();
      const wr = winRect();
      const x0 = vr.left - wr.left;
      const y0 = vr.top - wr.top;
      return { x: Math.max(x0 + 6, Math.min(x0 + vr.width - 6, p.x)), y: Math.max(y0 + 6, Math.min(y0 + vr.height - 6, p.y)) };
    };
    await glide(target, token);
    const p = target();
    ringAt(p);
    markerWorld = faceWorld(i);
    ui.marker.hidden = false;
    updateMarker();
    await tween(140, (e) => setHighlight(i, 0.85 * e), token);
    lightChip("point");
    lightChip("face");
    lightChip("id");
    ui.toastBody.textContent = tags[i];
    ui.toast.hidden = false;
    toastWorld = markerWorld;
    placeToast();
    await wait(1300, token);
    ui.toast.hidden = true;
    toastWorld = null;
    ui.marker.hidden = true;
    markerWorld = null;
  }

  // ---------------------------------------------------------------- visitor click-to-reference
  // Works on its own, in any mode (also during the replay): click any face of the arm and get the
  // viewer's red ring, marker, cyan face highlight and "Copied to clipboard" reference — and the
  // reference really is copied. Uses its own DOM so it never fights the replay's.
  const pickables = [...arm.meshes.values()].filter((m) => m.userData.triMap);
  const visitor = {
    ring: ui.ring.cloneNode(true),
    marker: ui.marker.cloneNode(true),
    toast: ui.toast.cloneNode(true),
    mesh: null, // clicked part
    local: null, // clicked point in that part's frame
    highlight: null,
    gen: 0
  };
  ui.ring.after(visitor.ring);
  ui.marker.after(visitor.marker);
  ui.toast.after(visitor.toast);
  visitor.ring.classList.remove("cac-ring-on");
  visitor.marker.hidden = true;
  visitor.toast.hidden = true;
  visitor.toast.classList.add("cac-toast-visitor");
  const visitorTitle = visitor.toast.querySelector(".cac-toast-title");
  const visitorBody = visitor.toast.querySelector(".cac-toast-body");
  const highlightMaterial = new THREE.MeshBasicMaterial({
    color: "#22c4f0", transparent: true, opacity: 0.85, depthWrite: false,
    polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2
  });
  const pickRay = new THREE.Raycaster();

  // Triangles adjacent by shared corner positions (corners are split at creases, so not by index).
  function triangleAdjacency(mesh) {
    if (mesh.userData.cornerTris) return mesh.userData;
    const pos = mesh.geometry.attributes.position;
    const idx = mesh.geometry.index.array;
    const keyOf = new Map();
    const corner = new Uint32Array(idx.length);
    for (let c = 0; c < idx.length; c += 1) {
      const v = idx[c];
      const key = `${Math.round(pos.getX(v) * 1e5)},${Math.round(pos.getY(v) * 1e5)},${Math.round(pos.getZ(v) * 1e5)}`;
      let id = keyOf.get(key);
      if (id === undefined) {
        id = keyOf.size;
        keyOf.set(key, id);
      }
      corner[c] = id;
    }
    const cornerTris = Array.from({ length: keyOf.size }, () => []);
    for (let c = 0; c < corner.length; c += 1) cornerTris[corner[c]].push((c / 3) | 0);
    mesh.userData.corner = corner;
    mesh.userData.cornerTris = cornerTris;
    return mesh.userData;
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
  // The flat face around triangle `t0`: connected triangles on the same plane (like the viewer's).
  function coplanarRegion(mesh, t0) {
    const { corner, cornerTris } = triangleAdjacency(mesh);
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
      const t = region[r];
      for (let k = 0; k < 3; k += 1) {
        for (const u of cornerTris[corner[3 * t + k]]) {
          if (seen[u]) continue;
          seen[u] = 1;
          triPlane(mesh, u, n, cen);
          if (n.dot(n0) > 0.999 && Math.abs(n0.dot(cen) - d0) < 0.0003) region.push(u);
        }
      }
    }
    return { tris: region, normal: n0 };
  }
  function highlightFor(mesh, region) {
    const pos = mesh.geometry.attributes.position;
    const idx = mesh.geometry.index.array;
    const out = new Float32Array(region.tris.length * 9);
    const lift = region.normal.clone().multiplyScalar(0.00025); // off the surface, no z-fight
    const v = new THREE.Vector3();
    region.tris.forEach((t, i) => {
      for (let k = 0; k < 3; k += 1) {
        v.fromBufferAttribute(pos, idx[3 * t + k]).add(lift);
        out.set([v.x, v.y, v.z], i * 9 + k * 3);
      }
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(out, 3));
    const hl = new THREE.Mesh(geometry, highlightMaterial.clone());
    hl.renderOrder = 2;
    return hl;
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
  function visitorWorld() {
    return visitor.mesh.localToWorld(visitor.local.clone());
  }
  function updateVisitorClick() {
    if (!visitor.mesh) return;
    const world = visitorWorld();
    placeMarker(visitor.marker, world);
    placeToast(visitor.toast, world);
  }
  function visitorClick(clientX, clientY) {
    const rect = ui.canvas.getBoundingClientRect();
    pickRay.setFromCamera(new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1), camera);
    arm.root.updateMatrixWorld(true);
    const hit = pickRay.intersectObjects(pickables, false).find((h) => (
      h.object.visible && !(h.object === connector && connectorProgress < 0.999) // skip the unbuilt part of the connector
    ));
    clearVisitorClick();
    if (!hit) return;
    const mesh = hit.object;
    const region = coplanarRegion(mesh, hit.faceIndex);
    const local = mesh.worldToLocal(hit.point.clone());
    const stl = mesh.userData.stlFromLink;
    const localMm = local.clone().applyMatrix4(stl).multiplyScalar(mesh.userData.stlUnit * 1000);
    const localNormal = region.normal.clone().transformDirection(stl);
    const nWorld = region.normal.clone().transformDirection(mesh.matrixWorld);
    const tag = formatTag(mesh.name, mesh.userData.link, hit.point, localMm, localNormal, nWorld, mesh.userData.triMap[hit.faceIndex]);

    visitor.mesh = mesh;
    visitor.local = local;
    visitor.highlight = highlightFor(mesh, region);
    mesh.add(visitor.highlight);
    const wr = winRect();
    ringAt({ x: clientX - wr.left, y: clientY - wr.top }, visitor.ring);
    visitor.marker.hidden = false;
    visitorTitle.textContent = "Copied to clipboard:";
    visitorBody.textContent = tag;
    visitor.toast.hidden = false;
    updateVisitorClick();
    dirty = true;
    try {
      navigator.clipboard.writeText(tag).catch(() => {
        visitorTitle.textContent = "Face reference:";
      });
    } catch {
      visitorTitle.textContent = "Face reference:";
    }
    // Show for 1 s, then fade the toast, marker and highlight out together.
    // (Timers, not frame hooks: stopping the replay clears frame hooks.)
    const gen = visitor.gen;
    const hl = visitor.highlight;
    visitor.toast.classList.remove("cac-fade");
    visitor.marker.classList.remove("cac-fade");
    setTimeout(() => {
      if (gen !== visitor.gen) return;
      visitor.toast.classList.add("cac-fade");
      visitor.marker.classList.add("cac-fade");
      tween(400, (e) => { hl.material.opacity = 0.85 * (1 - e); dirty = true; }, null).catch(() => {});
      setTimeout(() => {
        if (gen === visitor.gen) clearVisitorClick();
      }, 450);
    }, 1000);
  }

  // ---------------------------------------------------------------- the replay
  function resetScene() {
    dirty = true;
    setConnectorProgress(0);
    connector.material.emissiveIntensity = 0;
    highlights.forEach((_, i) => setHighlight(i, 0));
    ui.toast.hidden = true;
    ui.marker.hidden = true;
    markerWorld = null;
    ui.keycap.hidden = true;
    resetChips();
  }
  function builtState() {
    dirty = true;
    setConnectorProgress(1);
    connector.material.emissiveIntensity = 0;
    highlights.forEach((_, i) => setHighlight(i, 0));
    ui.toast.hidden = true;
    ui.marker.hidden = true;
    markerWorld = null;
    ui.keycap.hidden = true;
    hideCursor();
    resetChips();
    arm.setPose(HOME);
    finishPanel([tagFor(0), tagFor(1)]);
  }

  async function replayOnce(token) {
    if (connectorProgress > 0.01) {
      // Loop reset: un-grow the connector at the full view so the reset reads as intentional.
      const from = connectorProgress;
      await tween(520, (e) => setConnectorProgress(from * (1 - e)), token);
      await wait(250, token);
    }
    if (!pointerInside) userCamera = false;
    resetScene();
    resetPanel();
    arm.setPose(HOME);
    const tags = [tagFor(0), tagFor(1)];
    const shots = replayShots();
    cameraInReplayShot = true;
    showCursor(inputCaretPoint());
    await cameraTo(shots.face1, 1100, token);
    ui.input.classList.add("cac-input-active");
    await typeText("join the faces of ", token);
    await clickFace(0, token, tags);
    await glide(inputCaretPoint, token);
    await pasteTag(tags[0], token);
    await typeText(" with the face ", token);
    const toFace2 = cameraTo(shots.face2, 1000, token);
    await wait(250, token);
    await clickFace(1, token, tags);
    await toFace2;
    await glide(inputCaretPoint, token);
    await pasteTag(tags[1], token);
    await wait(260, token);
    // ⏎
    ui.inputText.innerHTML = "";
    ui.input.classList.remove("cac-input-active");
    logLine("user", `&gt; ${promptHtml(tags)}`);
    const thinking = logLine("thinking", `<span class="cac-spin">·</span> Thinking… <span class="cac-think-t">(0s · esc to interrupt)</span>`);
    const glyph = thinking.querySelector(".cac-spin");
    const started = performance.now();
    const spinner = (async () => {
      // Claude Code's spinner: the glyph pulses through these frames and back.
      const frames = ["·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢"];
      for (let f = 0; thinking.isConnected; f += 1) {
        glyph.textContent = frames[f % frames.length];
        await wait(110, token);
      }
    })().catch(() => {});
    await glide(() => ({ x: cursorPos.x + 30, y: cursorPos.y + 40 }), token, 400);
    const toGrow = cameraTo(shots.grow, 1300, token);
    await wait(1250, token);
    thinking.querySelector(".cac-think-t").textContent = `(${Math.max(1, Math.round((performance.now() - started) / 1000))}s · esc to interrupt)`;
    await wait(200, token);
    thinking.remove();
    await spinner;
    const tool = logLine("tool", `<span class="cac-bullet cac-ok">●</span> ${esc(TOOL_CALL)}<div class="cac-tool-out"></div>`);
    const out = tool.querySelector(".cac-tool-out");
    for (const [index, line] of TOOL_OUTPUT.entries()) {
      out.insertAdjacentHTML("beforeend", `<div>${index === 0 ? "⎿ " : "&nbsp;&nbsp;"}${esc(line)}</div>`);
      ui.log.scrollTop = ui.log.scrollHeight;
      await wait(140, token);
    }
    await toGrow;
    await tween(2200, (e, t) => {
      setConnectorProgress(e);
      // The base face is now inside the connector; fade its highlight so it can't show through the cut.
      setHighlight(0, 0.85 * Math.max(0, 1 - t / 0.12));
    }, token, easeInOut);
    await tween(700, (e, t) => {
      connector.material.emissiveIntensity = 0.35 * (1 - t);
      setHighlight(1, 0.85 * (1 - t));
    }, token);
    const reply = logLine("agent", `<span class="cac-bullet">●</span> `);
    for (const word of REPLY.split(" ")) {
      check(token);
      reply.insertAdjacentText("beforeend", `${word} `);
      ui.log.scrollTop = ui.log.scrollHeight;
      await wait(32, token);
    }
    hideCursor();
    await cameraTo(defaultView, 1400, token);
    cameraInReplayShot = false;
    // Hold on the finished arm; if the visitor's cursor is resting over the viewer, hand over early.
    for (let held = 0; held < 2600 && !(FOLLOW && pointerInside && held >= 700); held += 100) await wait(100, token);
  }

  let mode = "idle"; // idle | replay | user | returning
  let cameraInReplayShot = false; // true while the camera sits in a replay close-up
  async function runReplayLoop({ once = false } = {}) {
    if (replayToken) replayToken.aborted = true;
    const token = { aborted: false };
    replayToken = token;
    mode = "replay";
    setHint("replay");
    try {
      do {
        await replayOnce(token);
        if (FOLLOW && !once && pointerInside && lastPointerType !== "touch" && mode === "replay") {
          handOffToFollow();
          return;
        }
      } while (!once && !token.aborted && mode === "replay");
      if (once) {
        builtState();
        mode = "idle";
      }
    } catch (error) {
      if (!(error instanceof Aborted)) throw error;
    }
  }
  // The replay finished with the visitor's cursor over the viewer: the scene is already in its
  // built state, so the gripper simply starts following without a reset.
  function handOffToFollow() {
    replayToken = null;
    mode = "user";
    setHint("user");
    hasFollowTarget = false;
    if (lastClient) setFollowFromClient(lastClient.x, lastClient.y);
  }
  function stopReplay() {
    if (replayToken) replayToken.aborted = true;
    replayToken = null;
    frameHooks.clear();
    builtState();
  }

  // ---------------------------------------------------------------- TCP follow
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const followTarget = new THREE.Vector3();
  const smoothTarget = new THREE.Vector3();
  const smoothVel = new THREE.Vector3();
  let hasFollowTarget = false;
  const shoulder = () => arm.jointWorld("J2");
  const REACH = 0.42;
  function setFollowFromClient(clientX, clientY) {
    const rect = ui.canvas.getBoundingClientRect();
    pointer.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    // Work plane: faces the camera, through the arm's working area.
    const normal = camera.getWorldDirection(new THREE.Vector3()).negate();
    const anchor = armCenter.clone().add(new THREE.Vector3(0, 0, 0.02));
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, anchor);
    const hit = raycaster.ray.intersectPlane(plane, new THREE.Vector3());
    if (!hit) return;
    const s = shoulder();
    const offset = hit.clone().sub(s);
    if (offset.length() > REACH) offset.setLength(REACH);
    hit.copy(s).add(offset);
    hit.z = Math.max(0.05, hit.z);
    followTarget.copy(hit);
    if (!hasFollowTarget) {
      arm.tcpWorld(smoothTarget);
      smoothVel.set(0, 0, 0);
      hasFollowTarget = true;
    }
  }
  function updateFollow(dt) {
    if (mode !== "user" || !hasFollowTarget) return false;
    // critically damped spring toward the cursor target
    const omega = 1.75; // slow, so a visitor can catch a face to click while the arm follows
    const delta = new THREE.Vector3().subVectors(smoothTarget, followTarget);
    const accel = delta.multiplyScalar(-omega * omega).addScaledVector(smoothVel, -2 * omega);
    smoothVel.addScaledVector(accel, dt);
    smoothTarget.addScaledVector(smoothVel, dt);
    const before = arm.tcpWorld(new THREE.Vector3());
    arm.solveIK(smoothTarget, REST, { iterations: 3, maxStep: 0.6 * dt + 0.001 });
    // Still moving if the TCP moved this frame or the spring has not settled.
    return before.distanceTo(arm.tcpWorld(new THREE.Vector3())) > 0.00005 || smoothVel.lengthSq() > 1e-8;
  }

  // ---------------------------------------------------------------- interaction wiring
  let resumeTimer = 0;
  let pointerInside = false;
  let lastPointerType = "mouse";
  let lastClient = null;
  const HINTS = {
    replay: "Scripted replay · click any face to reference it · drag to orbit",
    "replay-hover": "Scripted replay · click any face to reference it · the gripper follows you when it ends",
    user: "The gripper follows your cursor · click any face to reference it · drag to orbit",
    idle: "Click any face to reference it · drag to orbit · click, then scroll to zoom"
  };
  function setHint(state) {
    if (ui.hint.textContent !== HINTS[state]) ui.hint.textContent = HINTS[state];
  }
  function enterUserMode() {
    clearTimeout(resumeTimer);
    if (mode === "replay" || mode === "returning") stopReplay();
    mode = "user";
    setHint("user");
    hideCursor();
    if (cameraInReplayShot) {
      // The visitor took over mid-replay: ease out of the close-up to the full arm.
      cameraInReplayShot = false;
      cameraTo(defaultView, 900, null).catch(() => {});
    }
  }
  controls.addEventListener("start", () => {
    userCamera = true;
    cameraInReplayShot = false; // the visitor is orbiting: the camera is theirs now
  });
  function scheduleResume(ms) {
    clearTimeout(resumeTimer);
    if (reduceMotion) return;
    resumeTimer = setTimeout(async () => {
      if (pointerInside || !visible || mode !== "user") return;
      mode = "returning";
      userCamera = false;
      hasFollowTarget = false;
      const from = arm.getPose();
      try {
        const token = { aborted: false };
        replayToken = token;
        await tween(900, (e) => {
          const pose = {};
          for (const name of Object.keys(HOME)) pose[name] = from[name] + (HOME[name] - from[name]) * e;
          arm.setPose(pose);
        }, token);
        if (mode === "returning") runReplayLoop();
      } catch (error) {
        if (!(error instanceof Aborted)) throw error;
      }
    }, ms);
  }
  ui.canvas.addEventListener("pointermove", (event) => {
    pointerInside = true;
    lastPointerType = event.pointerType;
    lastClient = { x: event.clientX, y: event.clientY };
    if (event.buttons) return; // orbiting/panning — the arm holds still
    if (event.pointerType === "touch" || !FOLLOW) return;
    // Hovering never interrupts a running replay (a resting cursor would otherwise kill it on
    // load); only a click/drag takes over. The gripper starts following when the replay ends.
    if (mode === "replay") {
      setHint("replay-hover");
      return;
    }
    if (mode !== "user") enterUserMode();
    setFollowFromClient(event.clientX, event.clientY);
  });
  // A click (not a drag) on a face = visitor click-to-reference, independent of the replay.
  let pressAt = null;
  ui.canvas.addEventListener("pointerdown", (event) => {
    if (event.button === 0) pressAt = { x: event.clientX, y: event.clientY, t: performance.now() };
  });
  ui.canvas.addEventListener("pointerup", (event) => {
    if (!pressAt || event.button !== 0) return;
    const moved = Math.hypot(event.clientX - pressAt.x, event.clientY - pressAt.y);
    const quick = performance.now() - pressAt.t < 650;
    pressAt = null;
    if (moved < 6 && quick) visitorClick(event.clientX, event.clientY);
  });
  ui.canvas.addEventListener("pointerdown", (event) => {
    pointerInside = true;
    lastPointerType = event.pointerType;
    controls.enableZoom = true;
    if (mode === "replay") {
      // Orbit/zoom alongside the replay instead of stopping it.
      ui.hint.classList.add("cac-hint-zoom");
      return;
    }
    if (!FOLLOW) return;
    enterUserMode();
    ui.hint.classList.add("cac-hint-zoom");
    if (event.pointerType === "touch") setFollowFromClient(event.clientX, event.clientY);
  });
  ui.canvas.addEventListener("pointerleave", () => {
    pointerInside = false;
    controls.enableZoom = false;
    ui.hint.classList.remove("cac-hint-zoom");
    if (mode === "replay") setHint("replay");
    // Touch "leaves" as soon as the finger lifts, so give taps a longer grace period.
    scheduleResume(lastPointerType === "touch" ? 6000 : 2500);
  });
  ui.play.addEventListener("click", () => {
    ui.play.hidden = true;
    runReplayLoop({ once: true }).then(() => {
      ui.play.hidden = false;
    });
  });

  // Pause when off-screen or hidden; restart the replay when it comes back.
  const io = new IntersectionObserver((entries) => {
    const nowVisible = entries.some((e) => e.isIntersecting);
    if (nowVisible === visible) return;
    visible = nowVisible;
    if (visible) {
      startLoop();
      if (!reduceMotion && mode !== "user") runReplayLoop();
    } else if (mode === "replay" || mode === "returning") {
      stopReplay();
      mode = "idle";
    }
  }, { threshold: 0.15 });
  io.observe(win);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      if (mode === "replay") {
        stopReplay();
        mode = "idle";
      }
    } else if (visible && !reduceMotion && mode === "idle") {
      runReplayLoop();
    }
  });

  if (options.debug) {
    window.__cac = { THREE, camera, controls, arm, faceWorld, faceNormalWorld, defaultView, replayShots, setHighlight, setConnectorProgress, box, armCenter, projectToWindow, get mode() { return mode; }, get progress() { return connectorProgress; }, clipPlane, connector };
  }
  startLoop();
  if (options.autoplay === false) {
    builtState();
    setConnectorProgress(0);
    mode = "idle";
  } else if (reduceMotion) {
    builtState();
    ui.play.hidden = false;
    mode = "idle";
  } else {
    resetPanel();
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
