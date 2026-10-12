// claude-as-cad page — table of contents with a robot-arm pointer.
//
//   import { mountTocArm } from "./hero/toc_arm.js";
//   mountTocArm(document.querySelector(".toc"));   // .toc holds <a href="#section"> links
//
// A small planar arm (two links and a level gripper, in the hero arm's colours) rides a vertical rail
// left of the list and reaches for the link of the section in view: the fingertip follows the active
// item on a fast spring, the rail carriage on a slow one, and two-link IK on the wrist bends the arm
// in between. Clicking a link scrolls there.
const SVG_NS = "http://www.w3.org/2000/svg";
const L1 = 34;
const L2 = 30;
const GRIP = 16; // wrist → fingertip
const WIDTH = 92; // the svg sits in the rail's left part, before the list
const RAIL_X = 10;

function node(tag, attrs, parent) {
  const n = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  parent?.appendChild(n);
  return n;
}

export function mountTocArm(toc) {
  if (!toc) return null;
  const links = [...toc.querySelectorAll("a[href^='#']")];
  const sections = links.map((a) => document.getElementById(a.getAttribute("href").slice(1))).filter(Boolean);
  if (!links.length) return null;
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

  const svg = node("svg", { class: "toc-arm", "aria-hidden": "true" }, null);
  toc.prepend(svg);
  const g = node("g", {}, svg);
  const rail = node("line", { x1: RAIL_X, x2: RAIL_X, class: "ta-rail" }, g);
  // drawn in arm-local coordinates; each part is placed by transform every frame
  const plate = node("rect", { width: 11, height: 18, rx: 2.5, class: "ta-plate" }, g);
  const link1 = node("g", {}, g);
  node("rect", { x: -4, y: -5, width: L1 + 8, height: 10, rx: 5, class: "ta-link" }, link1);
  const link2 = node("g", {}, g);
  node("rect", { x: -3.5, y: -4, width: L2 + 7, height: 8, rx: 4, class: "ta-link" }, link2);
  const hand = node("g", {}, g);
  node("rect", { x: -3, y: -5.5, width: 10, height: 11, rx: 2.5, class: "ta-servo" }, hand);
  node("path", { d: `M6 -5 L${GRIP} -2.2 L${GRIP} -0.7 L6 -1.3 Z`, class: "ta-fin" }, hand);
  node("path", { d: `M6 5 L${GRIP} 2.2 L${GRIP} 0.7 L6 1.3 Z`, class: "ta-fin" }, hand);
  const joints = [0, 1, 2].map(() => node("circle", { r: 5.2, class: "ta-servo" }, g));
  const pivots = [0, 1, 2].map(() => node("circle", { r: 2, class: "ta-pivot" }, g));

  let targetY = 0;
  let y = null; // fingertip
  let vy = 0;
  let by = 0; // rail carriage
  let vby = 0;
  let raf = 0;
  let last = 0;

  function layout() {
    const list = toc.getBoundingClientRect();
    svg.setAttribute("width", String(WIDTH));
    svg.setAttribute("height", String(list.height));
    rail.setAttribute("y1", String(linkY(links[0]) - 14));
    rail.setAttribute("y2", String(linkY(links[links.length - 1]) + 14));
  }
  function linkY(a) {
    const r = a.getBoundingClientRect();
    return r.top + r.height / 2 - toc.getBoundingClientRect().top;
  }
  function pose(ty, baseY) {
    // fingertip at (tipX, ty), gripper level → wrist at (tipX − GRIP, ty)
    const tipX = WIDTH - 4;
    const wx = tipX - GRIP - RAIL_X;
    const wy = ty - baseY;
    const d = Math.min(L1 + L2 - 0.5, Math.hypot(wx, wy));
    const cosE = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2);
    const e = -Math.acos(Math.max(-1, Math.min(1, cosE))); // elbow up
    const s = Math.atan2(wy, wx) - Math.atan2(L2 * Math.sin(e), L1 + L2 * Math.cos(e));
    const p0 = { x: RAIL_X, y: baseY };
    const p1 = { x: p0.x + L1 * Math.cos(s), y: p0.y + L1 * Math.sin(s) };
    const p2 = { x: p1.x + L2 * Math.cos(s + e), y: p1.y + L2 * Math.sin(s + e) };
    const deg = (r) => (r * 180) / Math.PI;
    plate.setAttribute("transform", `translate(${p0.x - 5.5} ${p0.y - 9})`);
    link1.setAttribute("transform", `translate(${p0.x} ${p0.y}) rotate(${deg(s)})`);
    link2.setAttribute("transform", `translate(${p1.x} ${p1.y}) rotate(${deg(s + e)})`);
    hand.setAttribute("transform", `translate(${p2.x} ${p2.y})`);
    [p0, p1, p2].forEach((p, i) => {
      joints[i].setAttribute("cx", p.x);
      joints[i].setAttribute("cy", p.y);
      pivots[i].setAttribute("cx", p.x);
      pivots[i].setAttribute("cy", p.y);
    });
  }
  function tick(now) {
    raf = 0;
    const dt = Math.min(0.05, (now - last) / 1000 || 0.016);
    last = now;
    const fast = 9;
    const slow = 3.2;
    vy += (-(fast * fast) * (y - targetY) - 2 * fast * vy) * dt;
    y += vy * dt;
    vby += (-(slow * slow) * (by - targetY) - 2 * slow * vby) * dt;
    by += vby * dt;
    pose(y, by);
    const moving = Math.abs(y - targetY) > 0.2 || Math.abs(vy) > 0.5 || Math.abs(by - targetY) > 0.2 || Math.abs(vby) > 0.5;
    if (moving) raf = requestAnimationFrame(tick);
  }
  function aim(index, immediate = false) {
    links.forEach((a, i) => a.classList.toggle("toc-on", i === index));
    targetY = linkY(links[index]);
    if (y === null || immediate || reduceMotion) {
      y = by = targetY;
      vy = vby = 0;
      pose(y, by);
      return;
    }
    if (!raf) {
      last = performance.now();
      raf = requestAnimationFrame(tick);
    }
  }

  // the active section: the last one whose top has passed 35% of the viewport
  let active = -1;
  function update() {
    const line = window.innerHeight * 0.35;
    let i = 0;
    sections.forEach((s, k) => { if (s.getBoundingClientRect().top <= line) i = k; });
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) i = sections.length - 1;
    if (i !== active) {
      active = i;
      aim(i);
    }
  }
  let scheduled = false;
  window.addEventListener("scroll", () => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => { scheduled = false; update(); });
  }, { passive: true });
  window.addEventListener("resize", () => { layout(); if (active >= 0) aim(active, true); });
  links.forEach((a, i) => a.addEventListener("click", (event) => {
    const target = sections[i];
    if (!target) return;
    event.preventDefault();
    target.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    history.replaceState(null, "", a.getAttribute("href"));
    active = i;
    aim(i);
  }));
  layout();
  update();
  aim(Math.max(0, active), true);
  return { update };
}
