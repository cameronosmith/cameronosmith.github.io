// claude-as-cad hero, pane 3 — "03 · Hierarchical notes".
//
//   import { mountNotesPane } from "./hero/notes_pane.js";
//   mountNotesPane(document.querySelector("#cac-notes"));
//
// A real slice of ~/cad (docs/, cadlib/, aruco_boards.py; scrubbed of hosts and absolute paths by
// tools/export_notes.py). Hover or click a folder to open it; hover a file to preview it, click to pin.
// While nobody is touching it, a short scripted tour walks the same beats as the clip:
// one folder/doc per CAD object → servos.md → cadlib/aruco.py → pitfalls.md.
import { el } from "./template.js?v=36b142ff4a";

const BASE_URL = new URL(".", import.meta.url);
const OPEN_AT_START = new Set(["cad/", "docs/", "cadlib/"]);
const OBJECT_DOCS = ["arm/", "camera_mount/", "gantry/", "home_robot/", "umi_gripper.md", "holders_yokes.md"];
const TOUR = [
  { group: OBJECT_DOCS, tag: "one folder / doc per CAD object", ms: 3200 },
  { file: "docs/servos.md", ms: 4200 },
  { file: "cadlib/aruco.py", also: ["docs/aruco_robot.md", "aruco_boards.py"], ms: 4200 },
  { file: "docs/pitfalls.md", also: ["docs/DESIGN_INDEX.md"], ms: 4600 }
];

const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function inline(s) {
  return esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
}
// Small, safe markdown subset: headings, lists, fenced code, tables (as preformatted), paragraphs.
function renderMarkdown(text) {
  const out = [];
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (line.startsWith("```")) {
      const code = [];
      for (i += 1; i < lines.length && !lines[i].startsWith("```"); i += 1) code.push(lines[i]);
      out.push(`<pre>${esc(code.join("\n"))}</pre>`);
    } else if (/^\s*\|/.test(line)) {
      const rows = [];
      for (; i < lines.length && /^\s*\|/.test(lines[i]); i += 1) if (!/^\s*\|[\s:|-]+\|\s*$/.test(lines[i])) rows.push(lines[i]);
      i -= 1;
      out.push(`<pre class="cac-nt-table">${esc(rows.join("\n"))}</pre>`);
    } else if (/^#{1,6}\s/.test(line)) {
      const level = Math.min(3, line.match(/^#+/)[0].length);
      out.push(`<h${level + 2}>${inline(line.replace(/^#+\s*/, ""))}</h${level + 2}>`);
    } else if (/^\s*([-*]|\d+\.)\s/.test(line)) {
      const indent = line.match(/^\s*/)[0].length;
      out.push(`<div class="cac-nt-li" style="padding-left:${12 + indent * 6}px">${inline(line.trim())}</div>`);
    } else if (line.trim()) {
      out.push(`<p>${inline(line)}</p>`);
    }
  }
  return out.join("");
}

export async function mountNotesPane(root, options = {}) {
  if (!root) return null;
  root.classList.add("cac-hero");
  const win = el("div", "cac-window cac-pane-notes");
  win.setAttribute("role", "region");
  win.setAttribute("aria-label", "Interactive demo: a real CAD notes folder. Open folders and preview files.");
  win.innerHTML = `
    <div class="cac-titlebar">
      <span class="cac-app">~/cad</span>
      <span class="cac-url">· real folders and notes, hover to open, click to pin</span>
      <span class="cac-tag" title="The files are real; machine names and absolute paths are scrubbed">real files</span>
    </div>
    <div class="cac-nt-body">
      <div class="cac-nt-tree" role="tree"></div>
      <div class="cac-nt-preview">
        <div class="cac-nt-path"></div>
        <div class="cac-nt-doc"></div>
      </div>
      <div class="cac-nt-tag" hidden></div>
    </div>`;
  root.innerHTML = "";
  root.appendChild(win);
  const treeEl = win.querySelector(".cac-nt-tree");
  const pathEl = win.querySelector(".cac-nt-path");
  const docEl = win.querySelector(".cac-nt-doc");
  const tagEl = win.querySelector(".cac-nt-tag");
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

  let data;
  try {
    data = await fetch(new URL("assets/notes.json", BASE_URL), { cache: "no-cache" }).then((r) => {
      if (!r.ok) throw new Error(`notes.json ${r.status}`);
      return r.json();
    });
  } catch (error) {
    console.warn("claude-as-cad notes pane: failed to load", error);
    treeEl.textContent = "Couldn't load the notes.";
    return null;
  }

  // ---------------------------------------------------------------- tree
  const rows = new Map(); // key (path, or folder name for folders) -> row element
  function build(node, depth, parentEl) {
    const isDir = !!node.children;
    const key = isDir ? node.name : node.path;
    const row = el("div", `cac-nt-row ${isDir ? "cac-nt-dir" : "cac-nt-file"}`);
    row.setAttribute("role", "treeitem");
    row.tabIndex = 0;
    row.dataset.key = key;
    row.style.paddingLeft = `${depth * 14}px`;
    row.innerHTML = `${depth ? '<span class="cac-nt-elbow">└</span>' : ""}<span class="cac-nt-name">${esc(node.name)}</span>`;
    parentEl.appendChild(row);
    rows.set(key, row);
    if (isDir) {
      const kids = el("div", "cac-nt-kids");
      parentEl.appendChild(kids);
      row._kids = kids;
      node.children.forEach((child) => build(child, depth + 1, kids));
      setOpen(row, OPEN_AT_START.has(node.name));
    }
  }
  function setOpen(row, open) {
    row.classList.toggle("cac-nt-open", open);
    row.setAttribute("aria-expanded", String(open));
    row._kids.hidden = !open;
  }
  build(data.tree, 0, treeEl);
  treeEl.appendChild(tagEl); // the group tag floats beside the rows it labels

  // ---------------------------------------------------------------- preview
  let pinned = null;
  let shown = null;
  function show(path) {
    if (shown === path) return;
    shown = path;
    for (const r of rows.values()) r.classList.toggle("cac-nt-shown", r.dataset.key === path);
    pathEl.textContent = path;
    const text = data.files[path] || "";
    docEl.innerHTML = path.endsWith(".py") ? `<pre class="cac-nt-code">${esc(text)}</pre>` : renderMarkdown(text);
    docEl.scrollTop = 0;
  }
  function mark(keys) {
    for (const r of rows.values()) r.classList.toggle("cac-nt-mark", keys.includes(r.dataset.key));
  }

  // ---------------------------------------------------------------- scripted tour (idle only)
  let touring = !reduceMotion;
  let tourTimer = 0;
  let step = 0;
  function tourStep() {
    if (!touring) return;
    const beat = TOUR[step % TOUR.length];
    step += 1;
    tagEl.hidden = !beat.tag;
    if (beat.group) {
      mark(beat.group);
      tagEl.textContent = beat.tag;
      const first = rows.get(beat.group[0]);
      tagEl.style.top = `${first.offsetTop}px`;
    } else {
      mark([beat.file, ...(beat.also || [])]);
      show(beat.file);
    }
    tourTimer = setTimeout(tourStep, beat.ms);
  }
  function stopTour() {
    touring = false;
    clearTimeout(tourTimer);
    tagEl.hidden = true;
    mark([]);
  }
  let resumeTimer = 0;

  // ---------------------------------------------------------------- interaction
  let hoverTimer = 0;
  treeEl.addEventListener("pointerover", (event) => {
    const row = event.target.closest(".cac-nt-row");
    if (!row) return;
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(() => {
      if (row.classList.contains("cac-nt-dir")) {
        if (!row.classList.contains("cac-nt-open")) setOpen(row, true);
      } else if (!pinned) {
        show(row.dataset.key);
      }
    }, 140);
  });
  treeEl.addEventListener("click", (event) => {
    const row = event.target.closest(".cac-nt-row");
    if (!row) return;
    clearTimeout(hoverTimer);
    if (row.classList.contains("cac-nt-dir")) {
      setOpen(row, !row.classList.contains("cac-nt-open"));
    } else {
      pinned = pinned === row.dataset.key ? null : row.dataset.key;
      for (const r of rows.values()) r.classList.toggle("cac-nt-pinned", r.dataset.key === pinned);
      show(row.dataset.key);
    }
  });
  treeEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      event.target.closest(".cac-nt-row")?.click();
    }
  });
  win.addEventListener("pointerenter", () => {
    clearTimeout(resumeTimer);
    stopTour();
  });
  win.addEventListener("focusin", stopTour);
  win.addEventListener("pointerleave", () => {
    if (reduceMotion || pinned) return;
    clearTimeout(resumeTimer);
    resumeTimer = setTimeout(() => {
      touring = true;
      tourStep();
    }, 8000);
  });

  show("docs/pitfalls.md");
  if (touring) tourTimer = setTimeout(tourStep, 1200);
  let interacted = false;
  win.addEventListener("pointerdown", () => { interacted = true; });
  new IntersectionObserver((entries) => {
    const visible = entries.some((e) => e.isIntersecting);
    if (!visible && touring) {
      clearTimeout(tourTimer);
    } else if (visible && touring && !interacted) {
      clearTimeout(tourTimer);
      tourTimer = setTimeout(tourStep, 900);
    }
  }, { threshold: 0.15 }).observe(win);
  return { destroy() { stopTour(); clearTimeout(resumeTimer); } };
}
