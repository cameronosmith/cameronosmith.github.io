// claude-as-cad page — small intro viewer: the 6-DOF arm's gripper follows the cursor.
//
//   import { mountFollowViewer } from "./hero/follow_pane.js";
//   mountFollowViewer(document.querySelector("#cac-follow"));
//
// The same arm, IK and viewer setup as the click-to-reference pane, without the replay: the TCP
// follows the cursor anywhere on the page while the viewer is on screen (damped-least-squares IK on
// J1–J5 toward a camera-facing work plane), drag to orbit, click then scroll to zoom. The arm eases
// home when the cursor leaves the window.
import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js?v=68c77e28da";
import { loadArm } from "./arm.js?v=68c77e28da";

const BASE_URL = new URL(".", import.meta.url);
const REACH = 0.36; // a little inside full extension, so the arm never has to lock straight

export async function mountFollowViewer(root, options = {}) {
  if (!root) return null;
  const canvas = document.createElement("canvas");
  canvas.className = "cac-follow-canvas";
  canvas.setAttribute("aria-label", "Robot arm: the gripper follows your cursor. Drag to orbit; click, then scroll to zoom.");
  root.appendChild(canvas);
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "default" });
  } catch {
    root.remove();
    return null;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
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
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), new THREE.ShadowMaterial({ opacity: 0.12 }));
  ground.receiveShadow = true;
  scene.add(ground);

  let arm;
  try {
    arm = await loadArm(BASE_URL);
  } catch (error) {
    console.warn("claude-as-cad follow viewer: couldn't load the arm", error);
    renderer.dispose();
    root.remove();
    return null;
  }
  scene.add(arm.root);
  for (const m of arm.meshes.values()) if (m.userData.role === "highlight") m.visible = false;
  const HOME = arm.getPose();

  // framing: the whole arm, a little more of its reach above it
  const box = new THREE.Box3();
  arm.root.traverse((o) => { if (o.isMesh && o.visible) box.expandByObject(o); });
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  const dir = new THREE.Vector3(1.0, -1.35, 0.72).normalize();
  // aim a little left of and below the arm, so it sits right and up in the frame with room to reach
  // left (toward the text) without being cropped; pulled back so the whole reach fits
  const right = new THREE.Vector3().crossVectors(dir.clone().negate(), new THREE.Vector3(0, 0, 1)).normalize();
  const target = center.clone().addScaledVector(right, -size * 0.14).add(new THREE.Vector3(0, 0, -size * 0.06));
  camera.position.copy(target).addScaledVector(dir, size * 2.3);
  const controls = new OrbitControls(camera, canvas);
  controls.target.copy(target);
  controls.enableDamping = true;
  controls.dampingFactor = 0.09;
  controls.enableZoom = false; // turned on after a click, so page scrolling isn't hijacked
  controls.minDistance = 0.25;
  controls.maxDistance = 3;
  controls.update();

  // ---------------------------------------------------------------- cursor follow (as the click-to-reference pane)
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const followTarget = new THREE.Vector3();
  const smooth = new THREE.Vector3();
  const vel = new THREE.Vector3();
  let following = false;
  let returning = false;
  let orbiting = false; // IK is frozen while the visitor moves the view
  let stall = 0; // seconds the gripper has been stuck far from the cursor (pinned at a joint limit)
  let recoveredFor = null; // the target the last recovery was for
  let recovering = 0;
  function setTarget(clientX, clientY) {
    if (orbiting) return;
    const r = canvas.getBoundingClientRect();
    // clip the cursor to the viewer's box: far-away cursors aim at the nearest edge, not off into space
    const clip = (v) => Math.max(-0.92, Math.min(0.92, v));
    ndc.set(clip(((clientX - r.left) / r.width) * 2 - 1), clip(-((clientY - r.top) / r.height) * 2 + 1));
    raycaster.setFromCamera(ndc, camera);
    const normal = camera.getWorldDirection(new THREE.Vector3()).negate();
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, center.clone().add(new THREE.Vector3(0, 0, 0.02)));
    const hit = raycaster.ray.intersectPlane(plane, new THREE.Vector3());
    if (!hit) return;
    const s = arm.jointWorld("J2");
    const off = hit.clone().sub(s);
    if (off.length() > REACH) off.setLength(REACH);
    hit.copy(s).add(off);
    hit.z = Math.max(0.05, hit.z);
    followTarget.copy(hit);
    if (!following) {
      arm.tcpWorld(smooth);
      vel.set(0, 0, 0);
      following = true;
    }
    returning = false;
    kick();
  }
  function step(dt) {
    if (returning) {
      const pose = arm.getPose();
      let moving = false;
      const k = 1 - Math.exp(-dt * 3);
      const next = {};
      for (const [n, v] of Object.entries(HOME)) {
        next[n] = pose[n] + (v - pose[n]) * k;
        if (Math.abs(v - pose[n]) > 1e-4) moving = true;
      }
      arm.setPose(next);
      if (!moving) returning = false;
      return moving;
    }
    if (!following || orbiting) return false;
    const omega = 3.5;
    const accel = smooth.clone().sub(followTarget).multiplyScalar(-omega * omega).addScaledVector(vel, -2 * omega);
    vel.addScaledVector(accel, dt);
    smooth.addScaledVector(vel, dt);
    const before = arm.tcpWorld(new THREE.Vector3());
    const err = arm.solveIK(smooth, HOME, { iterations: 3, maxStep: 1.2 * dt + 0.002 });
    // Stuck: far from the target and not getting closer (a joint pinned at its limit, e.g. after the
    // view was rotated so the cursor maps behind the arm). Measured per second, not per frame — on a
    // fast display a converging arm moves < 0.4 mm/frame and must not count as stuck. Recovery
    // (unfold toward home, then re-approach) runs at most once per target.
    const speed = before.distanceTo(arm.tcpWorld(new THREE.Vector3())) / Math.max(dt, 1e-3); // m/s
    stall = err > 0.03 && speed < 0.004 ? stall + dt : 0;
    const sameTarget = recoveredFor && recoveredFor.distanceTo(followTarget) < 0.02;
    if (stall > 0.6 && !sameTarget) {
      recoveredFor = followTarget.clone();
      recovering = 0.8; // seconds of easing toward home
      stall = 0;
    }
    if (recovering > 0) {
      recovering -= dt;
      const pose = arm.getPose();
      const next = {};
      const k = 1 - Math.exp(-dt * 4);
      for (const [n, v] of Object.entries(HOME)) next[n] = pose[n] + (v - pose[n]) * k;
      arm.setPose(next);
      vel.set(0, 0, 0);
    }
    // a degenerate solve (NaN) would freeze the arm for good: drop back to home and start over
    if (Object.values(arm.getPose()).some((v) => !Number.isFinite(v))) {
      arm.setPose(HOME);
      following = false;
      return true;
    }
    return recovering > 0 || before.distanceTo(arm.tcpWorld(new THREE.Vector3())) > 0.00005 || vel.lengthSq() > 1e-8;
  }

  // ---------------------------------------------------------------- render on change
  let visible = false;
  let raf = 0;
  let last = performance.now();
  let dirty = true;
  function resize() {
    const r = root.getBoundingClientRect();
    renderer.setSize(Math.max(1, Math.round(r.width)), Math.max(1, Math.round(r.height)), false);
    camera.aspect = Math.max(1, r.width) / Math.max(1, r.height);
    camera.updateProjectionMatrix();
    dirty = true;
    kick();
  }
  function frame(now) {
    raf = 0;
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    const moving = step(dt);
    const orbiting = controls.update();
    if (moving || orbiting || dirty) {
      dirty = false;
      renderer.render(scene, camera);
      placeOverlays();
    }
    if (visible && (moving || orbiting)) raf = requestAnimationFrame(frame);
  }
  function kick() {
    if (!raf && visible) {
      last = performance.now();
      raf = requestAnimationFrame(frame);
    }
  }
  controls.addEventListener("change", kick);
  // freeze the IK while the view is being dragged; resume from wherever the gripper is afterwards
  controls.addEventListener("start", () => { orbiting = true; vel.set(0, 0, 0); });
  controls.addEventListener("end", () => {
    orbiting = false;
    following = false; // the next cursor move re-seeds the spring at the current TCP (new view, new work plane)
    if (lastClient) setTarget(lastClient.x, lastClient.y);
  });
  new ResizeObserver(resize).observe(root);
  new IntersectionObserver((entries) => {
    visible = entries.some((e) => e.isIntersecting);
    if (visible) { dirty = true; kick(); }
  }).observe(root);
  let lastClient = null;
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  if (!reduceMotion) {
    window.addEventListener("pointermove", (e) => {
      lastClient = { x: e.clientX, y: e.clientY };
      if (visible && e.pointerType !== "touch" && !e.buttons) setTarget(e.clientX, e.clientY);
    }, { passive: true });
    document.documentElement.addEventListener("pointerleave", () => {
      following = false;
      returning = true;
      kick();
    });
  }
  // ---------------------------------------------------------------- click-to-reference (as the viewer below)
  root.classList.add("cac-hero");
  const ring = Object.assign(document.createElement("div"), { className: "cac-ring" });
  const marker = Object.assign(document.createElement("div"), { className: "cac-marker", hidden: true });
  const toast = Object.assign(document.createElement("div"), { className: "cac-toast", hidden: true });
  toast.innerHTML = '<div class="cac-toast-title">Copied to clipboard:</div><div class="cac-toast-body"></div>';
  root.append(marker, toast, ring);
  const toastTitle = toast.querySelector(".cac-toast-title");
  const toastBody = toast.querySelector(".cac-toast-body");
  const baseLink = arm.links.get("base");
  const pickables = [...arm.meshes.values()].filter((m) => m.userData.triMap);
  const pick = new THREE.Raycaster();
  const click = { mesh: null, local: null, hl: null, gen: 0 };
  function rootPoint(world) {
    const v = world.clone().project(camera);
    const r = root.getBoundingClientRect();
    return { x: (v.x * 0.5 + 0.5) * r.width, y: (-v.y * 0.5 + 0.5) * r.height };
  }
  function placeOverlays() {
    if (!click.mesh) return;
    const p = rootPoint(click.local.clone().applyMatrix4(click.mesh.matrixWorld));
    marker.style.transform = `translate(${p.x}px, ${p.y}px)`;
    const r = root.getBoundingClientRect();
    const tw = toast.offsetWidth;
    const th = toast.offsetHeight;
    toast.style.left = `${Math.max(8, Math.min(r.width - tw - 8, p.x - tw * 0.6))}px`;
    toast.style.top = `${p.y + 18 + th < r.height - 8 ? p.y + 18 : Math.max(8, p.y - th - 18)}px`;
  }
  function triPlane(m, t, n, cen) {
    const pos = m.geometry.attributes.position;
    const idx = m.geometry.index.array;
    const a = new THREE.Vector3().fromBufferAttribute(pos, idx[3 * t]);
    const b = new THREE.Vector3().fromBufferAttribute(pos, idx[3 * t + 1]);
    const c = new THREE.Vector3().fromBufferAttribute(pos, idx[3 * t + 2]);
    n.subVectors(c, b).cross(new THREE.Vector3().subVectors(a, b)).normalize();
    cen.copy(a).add(b).add(c).multiplyScalar(1 / 3);
  }
  function coplanarRegion(m, t0) {
    if (!m.userData.cornerTris) {
      const pos = m.geometry.attributes.position;
      const idx = m.geometry.index.array;
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
      Object.assign(m.userData, { corner, cornerTris });
    }
    const { corner, cornerTris } = m.userData;
    const n0 = new THREE.Vector3();
    const c0 = new THREE.Vector3();
    triPlane(m, t0, n0, c0);
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
          triPlane(m, u, n, cen);
          if (n.dot(n0) > 0.999 && Math.abs(n0.dot(cen) - d0) < 0.0003) region.push(u);
        }
      }
    }
    return { tris: region, normal: n0 };
  }
  function clearClick() {
    click.gen += 1;
    if (click.hl) { click.hl.removeFromParent(); click.hl.geometry.dispose(); click.hl.material.dispose(); click.hl = null; }
    toast.hidden = marker.hidden = true;
    toast.classList.remove("cac-fade");
    marker.classList.remove("cac-fade");
    click.mesh = null;
    dirty = true;
    kick();
  }
  function referenceAt(clientX, clientY) {
    const r = canvas.getBoundingClientRect();
    pick.setFromCamera(new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1), camera);
    arm.root.updateMatrixWorld(true);
    const hit = pick.intersectObjects(pickables, false).find((h) => h.object.visible);
    clearClick();
    if (!hit) return;
    const m = hit.object;
    const region = coplanarRegion(m, hit.faceIndex);
    const local = m.worldToLocal(hit.point.clone());
    const stl = m.userData.stlFromLink;
    const lm = local.clone().applyMatrix4(stl).multiplyScalar(m.userData.stlUnit * 1000);
    const ln = region.normal.clone().transformDirection(stl);
    const nWorld = region.normal.clone().transformDirection(m.matrixWorld);
    const base = baseLink.worldToLocal(hit.point.clone()).multiplyScalar(1000);
    const qb = new THREE.Quaternion();
    baseLink.getWorldQuaternion(qb);
    const nb = nWorld.applyQuaternion(qb.invert());
    const comps = [nb.x, nb.y, nb.z];
    let k = 0;
    for (let j = 1; j < 3; j += 1) if (Math.abs(comps[j]) > Math.abs(comps[k])) k = j;
    const f1 = (v) => (Math.abs(v) < 0.05 ? 0 : v).toFixed(1);
    const f2 = (v) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2);
    const text = `[${m.name} in ${m.userData.link} @ (${f1(base.x)}, ${f1(base.y)}, ${f1(base.z)})mm | local (${f1(lm.x)}, ${f1(lm.y)}, ${f1(lm.z)})mm n(${[ln.x, ln.y, ln.z].map(f2).join(", ")}) tri ${m.userData.triMap[hit.faceIndex]}, ${comps[k] >= 0 ? "+" : "-"}${"XYZ"[k]} side face]`;
    const pos = m.geometry.attributes.position;
    const idx = m.geometry.index.array;
    const out = new Float32Array(region.tris.length * 9);
    const lift = region.normal.clone().multiplyScalar(0.00025);
    const v = new THREE.Vector3();
    region.tris.forEach((t, i) => {
      for (let c = 0; c < 3; c += 1) { v.fromBufferAttribute(pos, idx[3 * t + c]).add(lift); out.set([v.x, v.y, v.z], i * 9 + c * 3); }
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(out, 3));
    const hl = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: "#22c4f0", transparent: true, opacity: 0.85, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }));
    hl.renderOrder = 2;
    m.add(hl);
    Object.assign(click, { mesh: m, local, hl });
    const rr = root.getBoundingClientRect();
    ring.style.left = `${clientX - rr.left}px`;
    ring.style.top = `${clientY - rr.top}px`;
    ring.classList.remove("cac-ring-on");
    void ring.offsetWidth;
    ring.classList.add("cac-ring-on");
    toastTitle.textContent = "Copied to clipboard:";
    toastBody.textContent = text;
    toast.hidden = marker.hidden = false;
    placeOverlays();
    dirty = true;
    kick();
    try { navigator.clipboard.writeText(text).catch(() => { toastTitle.textContent = "Face reference:"; }); } catch { toastTitle.textContent = "Face reference:"; }
    const gen = click.gen;
    setTimeout(() => {
      if (gen !== click.gen) return;
      toast.classList.add("cac-fade");
      marker.classList.add("cac-fade");
      setTimeout(() => { if (gen === click.gen) clearClick(); }, 450);
    }, 1000);
  }
  let pressAt = null;
  canvas.addEventListener("pointerdown", (e) => {
    controls.enableZoom = true;
    if (e.button === 0) pressAt = { x: e.clientX, y: e.clientY, t: performance.now() };
  });
  canvas.addEventListener("pointerup", (e) => {
    if (!pressAt || e.button !== 0) return;
    const moved = Math.hypot(e.clientX - pressAt.x, e.clientY - pressAt.y);
    const quick = performance.now() - pressAt.t < 650;
    pressAt = null;
    if (moved < 6 && quick) referenceAt(e.clientX, e.clientY);
  });
  canvas.addEventListener("pointerleave", () => { controls.enableZoom = false; });
  resize();
  if (options.debug) window.__cacFollow = { arm, camera, controls, setTarget };
  return { destroy() { renderer.dispose(); } };
}
