// claude-as-cad hero — DOM template (no three.js), shared by loader.js and hero.js.

// Text from the real session the reference clip re-enacts.
export const HISTORY = [
  { kind: "meta", text: "cad agent · ~/cad_double_to_single" },
  { kind: "user", text: "Actually undo it and set J4's yoke to 90 degree default again" },
  { kind: "agent", text: "Undone — the connector is removed and J4's yoke is at its 90° clocking (sq3r). Chain positions untouched." }
];
const CURSOR_SVG = `<svg viewBox="0 0 20 24" width="20" height="24" aria-hidden="true"><path d="M2 1.5v18.2l4.6-4.4 3 6.9 3.1-1.3-3-6.8h6.5z" fill="#111" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>`;

// horizontal-scrub cursor (the editor's number fields are dragged left/right, like Blender)
const SCRUB_SVG = `<svg class="cac-scrub-icon" viewBox="0 0 26 16" width="26" height="16" aria-hidden="true"><path d="M1.5 8l5-5v3h12.9V3l5 5-5 5v-3H6.5v3z" fill="#111" stroke="#fff" stroke-width="1.4" stroke-linejoin="round"/></svg>`;

export function el(tag, className, html) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

export function template() {
  const win = el("div", "cac-window");
  win.setAttribute("role", "region");
  win.setAttribute("aria-label", "Interactive demo: the claude-as-cad viewer showing a robot arm, next to a scripted replay of a Claude Code session");
  win.innerHTML = `
    <div class="cac-titlebar">
      <span class="cac-app">CAD viewer</span>
      <span class="cac-url">· localhost:4178/?file=double_to_single_compact_long.urdf</span>
      <span class="cac-tag" title="The agent session on the right is a scripted replay of a real one">scripted replay</span>
    </div>
    <div class="cac-body">
      <div class="cac-viewer">
        <canvas class="cac-canvas" aria-label="Robot arm. Move your pointer over it and the gripper follows. Drag to orbit."></canvas>
        <div class="cac-marker" hidden></div>
        <div class="cac-toast" hidden><div class="cac-toast-title">Copied to clipboard:</div><div class="cac-toast-body"></div></div>
        <div class="cac-chips" aria-hidden="true">
          <span data-chip="point">3D point</span><span data-chip="face">mesh face</span><span data-chip="id">object ID</span><span data-chip="chat" class="cac-chip-dark">→ clipboard → chat</span>
        </div>
        <div class="cac-hint" aria-hidden="true">Scripted replay · click any face to reference it · drag to orbit</div>
        <div class="cac-loading" aria-hidden="true">Loading arm…</div>
      </div>
      <div class="cac-panel" aria-hidden="true">
        <div class="cac-panel-head"><span class="cac-pdot"></span>cad — claude</div>
        <div class="cac-log"></div>
        <div class="cac-input"><span class="cac-prompt">&gt;</span><span class="cac-input-text"></span><span class="cac-caret"></span></div>
        <div class="cac-shortcuts">? for shortcuts</div>
        <div class="cac-keycap" hidden><span class="cac-keys"><kbd>⌘</kbd><kbd>V</kbd></span><span class="cac-keylabel">paste</span></div>
      </div>
    </div>
    <div class="cac-ring" aria-hidden="true"></div>
    <div class="cac-vcursor" aria-hidden="true" hidden>${CURSOR_SVG}</div>
    <button type="button" class="cac-play" hidden>▶ Play the replay</button>`;
  return win;
}


export function renderHistory(logEl) {
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  logEl.innerHTML = HISTORY.map((item) => {
    if (item.kind === "meta") return `<div class="cac-line cac-meta">${esc(item.text)}</div>`;
    if (item.kind === "user") return `<div class="cac-line cac-user cac-dim">&gt; ${esc(item.text)}</div>`;
    return `<div class="cac-line cac-agent cac-dim"><span class="cac-bullet">●</span> ${esc(item.text)}</div>`;
  }).join("");
}

// ---------------------------------------------------------------- pane 2: primitive editor
export const EDITOR_HISTORY = [{ kind: "meta", text: "cad agent · ~/cad_robot2" }];

export function editorPaneTemplate() {
  const win = el("div", "cac-window cac-pane-editor");
  win.setAttribute("role", "region");
  win.setAttribute("aria-label", "Interactive demo: the claude-as-cad primitive editor on a servo module, next to a scripted replay of a Claude Code session");
  win.innerHTML = `
    <div class="cac-titlebar">
      <span class="cac-app">Primitive editor</span>
      <span class="cac-url">· localhost:4180/?file=st3095_module.urdf</span>
      <span class="cac-tag" title="The session on the right replays a real one; the editor steps are a re-enactment">scripted replay</span>
    </div>
    <div class="cac-body">
      <div class="cac-viewer">
        <canvas class="cac-canvas" aria-label="Servo module. Drag to orbit; click a part to copy its reference."></canvas>
        <svg class="cac-leaders" aria-hidden="true"></svg>
        <div class="cac-callout cac-callout-good" hidden>✓ Yoke: good</div>
        <div class="cac-callout cac-callout-bad" hidden>✗ Holder: wrong orientation</div>
        <div class="cac-marker" hidden></div>
        <div class="cac-toast" hidden><div class="cac-toast-title">Copied to clipboard:</div><div class="cac-toast-body"></div></div>
        <div class="cac-pe" aria-hidden="true">
          <div class="cac-pe-head"><b>Primitive editor</b><span class="cac-pe-badge" hidden></span><span class="cac-pe-hide">Hide</span></div>
          <div class="cac-pe-frame"><span>FRAME</span><i class="on">World (Z up)</i><i>Base</i></div>
          <div class="cac-pe-empty">Click a part to select it, or add a box.</div>
          <div class="cac-pe-sel" hidden>
            <div class="cac-pe-name"><span class="cac-pe-name-text"></span><span class="cac-pe-caret" hidden></span></div>
            <div class="cac-pe-sub"></div>
            <div class="cac-pe-group" data-group="loc"><div class="cac-pe-label">LOCATION (MM, WORLD Z-UP, CENTER)</div>
              <div class="cac-pe-xyz"><span>X <b data-f="loc0"></b></span><span>Y <b data-f="loc1"></b></span><span>Z <b data-f="loc2"></b></span></div></div>
            <div class="cac-pe-group" data-group="rot"><div class="cac-pe-label">ROTATION (DEG)</div>
              <div class="cac-pe-xyz"><span>X <b data-f="rot0">0</b></span><span>Y <b data-f="rot1">0</b></span><span>Z <b data-f="rot2">0</b></span></div></div>
            <div class="cac-pe-group" data-group="size"><div class="cac-pe-label">SIZE (MM)</div>
              <div class="cac-pe-xyz"><span>X <b data-f="size0"></b></span><span>Y <b data-f="size1"></b></span><span>Z <b data-f="size2"></b></span></div></div>
            <div class="cac-pe-del"></div>
          </div>
          <div class="cac-pe-add">+ Add cube</div>
          <div class="cac-pe-label cac-pe-edits-label">EDITS (<span class="cac-pe-count">0</span>)</div>
          <div class="cac-pe-edits"></div>
          <div class="cac-pe-actions"><span class="cac-pe-save">Save to edit log</span><span>Copy edits</span><span class="cac-pe-revert">Revert all</span></div>
          <div class="cac-pe-note">Reference edits only — kept in the browser; the CAD is never changed.</div>
          <div class="cac-pe-saved" hidden>Saved to ~/cad_editor_edits/EDIT_LOG.md</div>
        </div>
        <div class="cac-hint" aria-hidden="true">Scripted replay · click any part to reference it · drag to orbit</div>
      </div>
      <div class="cac-panel" aria-hidden="true">
        <div class="cac-panel-head"><span class="cac-pdot"></span>cad — claude</div>
        <div class="cac-log"></div>
        <div class="cac-input"><span class="cac-prompt">&gt;</span><span class="cac-input-text"></span><span class="cac-caret"></span></div>
        <div class="cac-shortcuts">? for shortcuts</div>
      </div>
    </div>
    <div class="cac-ring" aria-hidden="true"></div>
    <div class="cac-vcursor" aria-hidden="true" hidden>${CURSOR_SVG}${SCRUB_SVG}</div>
    <button type="button" class="cac-play" hidden>▶ Play the replay</button>`;
  return win;
}

export function renderEditorHistory(logEl) {
  logEl.innerHTML = EDITOR_HISTORY.map((item) => `<div class="cac-line cac-meta">${item.text}</div>`).join("");
}

// ---------------------------------------------------------------- pane 5: whole-arm build
export const ARMBUILD_HISTORY = [{ kind: "meta", text: "cad agent · ~/cad_double_to_single" }];

export function armBuildTemplate() {
  const win = el("div", "cac-window cac-pane-armbuild");
  win.setAttribute("role", "region");
  win.setAttribute("aria-label", "Interactive demo: building the whole robot arm in the editor with Claude, a re-enactment of the build video");
  win.innerHTML = `
    <div class="cac-titlebar">
      <span class="cac-app">CAD editor</span>
      <span class="cac-url">· localhost:4180/?file=arm_build.urdf</span>
      <span class="cac-tag" title="Re-enactment: prompts and tool output from the real build sessions (Aug–Oct), compressed">re-enactment</span>
    </div>
    <div class="cac-ab-beats" role="tablist" aria-label="Build steps"></div>
    <div class="cac-body">
      <div class="cac-viewer">
        <canvas class="cac-canvas" aria-label="The robot arm being built. Drag to orbit; click a face to copy its reference."></canvas>
        <div class="cac-marker" hidden></div>
        <div class="cac-toast" hidden><div class="cac-toast-title">Copied to clipboard:</div><div class="cac-toast-body"></div></div>
        <div class="cac-ab-keys" hidden></div>
        <div class="cac-pe" aria-hidden="true">
          <div class="cac-pe-head"><b>Primitive editor</b><span class="cac-pe-badge" hidden></span><span class="cac-pe-hide">Hide</span></div>
          <div class="cac-pe-frame"><span>FRAME</span><i class="on">World (Z up)</i><i>Base</i></div>
          <div class="cac-pe-empty">Click a part to select it, or add a box.</div>
          <div class="cac-pe-sel" hidden>
            <div class="cac-pe-name"><span class="cac-pe-name-text"></span><span class="cac-pe-caret" hidden></span></div>
            <div class="cac-pe-sub"></div>
            <div class="cac-pe-group"><div class="cac-pe-label">LOCATION (MM, WORLD Z-UP, CENTER)</div>
              <div class="cac-pe-xyz"><span>X <b data-f="0"></b></span><span>Y <b data-f="1"></b></span><span>Z <b data-f="2"></b></span></div></div>
            <div class="cac-pe-group"><div class="cac-pe-label">ROTATION (DEG)</div>
              <div class="cac-pe-xyz"><span>X <b data-f="3"></b></span><span>Y <b data-f="4"></b></span><span>Z <b data-f="5"></b></span></div></div>
            <div class="cac-pe-group"><div class="cac-pe-label">SCALE</div>
              <div class="cac-pe-xyz"><span>X <b data-f="6">1</b></span><span>Y <b data-f="7">1</b></span><span>Z <b data-f="8">1</b></span></div></div>
          </div>
          <div class="cac-pe-label cac-pe-edits-label">EDITS (<span class="cac-pe-count">0</span>)</div>
          <div class="cac-pe-edits"></div>
          <div class="cac-pe-actions"><span class="cac-pe-save" data-btn="Save to edit log">Save to edit log</span><span>Copy edits</span><span class="cac-pe-revert" data-btn="Revert all">Revert all</span></div>
          <div class="cac-pe-saved" hidden>Saved to ~/cad_editor_edits/EDIT_LOG.md</div>
        </div>
        <div class="cac-hint" aria-hidden="true">Re-enactment · click any face to reference it · drag to orbit</div>
      </div>
      <div class="cac-panel" aria-hidden="true">
        <div class="cac-panel-head"><span class="cac-pdot"></span>cad — claude</div>
        <div class="cac-log"></div>
        <div class="cac-input"><span class="cac-prompt">&gt;</span><span class="cac-input-text"></span><span class="cac-caret"></span></div>
        <div class="cac-shortcuts">? for shortcuts</div>
        <div class="cac-keycap" hidden><span class="cac-keys"><kbd>⌘</kbd><kbd class="cac-key2">V</kbd></span><span class="cac-keylabel">paste</span></div>
      </div>
    </div>
    <div class="cac-ring" aria-hidden="true"></div>
    <div class="cac-vcursor" aria-hidden="true" hidden>${CURSOR_SVG}${SCRUB_SVG}</div>
    <button type="button" class="cac-play" hidden>▶ Play the build</button>`;
  return win;
}

export function renderArmBuildHistory(logEl) {
  logEl.innerHTML = ARMBUILD_HISTORY.map((item) => `<div class="cac-line cac-meta">${item.text}</div>`).join("");
}

// ---------------------------------------------------------------- gallery (CAD | real grid)
// order + photo focal points as in the opening video's 3×2 grid; `desc` is filled from the page
export const GALLERY_ITEMS = [
  { key: "arm", name: "Robot arm", fx: 0.2, fy: 0.5 },
  { key: "umi", name: "UMI gripper", fx: 0.5, fy: 0.35 },
  { key: "wheels", name: "Mobile base", fx: 0.5, fy: 0.6 },
  { key: "campost", name: "Camera post", fx: 0.5, fy: 0.0 },
  { key: "robot2", name: "Double-servo arm", fx: 0.5, fy: 0.5 },
  { key: "lift", name: "Lift", fx: 0.5, fy: 0.4 }
];

export function galleryGrid(descriptions = {}) {
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const grid = el("div", "cac-gal");
  for (const item of GALLERY_ITEMS) {
    const fig = el("figure", "cac-gal-card");
    fig.dataset.build = item.key;
    const photo = new URL(`assets/gallery/photo_${item.key}.webp`, import.meta.url).href;
    fig.innerHTML = `
      <div class="cac-gal-pair">
        <div class="cac-gal-cad"><canvas aria-label="${esc(item.name)} CAD model: drag to orbit, double-click to replay the assembly"></canvas><span class="cac-gal-tag">CAD</span></div>
        <div class="cac-gal-real"><img alt="The real ${esc(item.name.toLowerCase())}" loading="lazy" style="object-position:${item.fx * 100}% ${item.fy * 100}%" src="${photo}"><span class="cac-gal-tag cac-gal-tag-real">real</span></div>
      </div>
      <figcaption><span class="cac-gal-name">${esc(item.name)}</span><span class="cac-gal-desc">${esc(descriptions[item.key] || "")}</span></figcaption>`;
    grid.appendChild(fig);
  }
  return grid;
}
