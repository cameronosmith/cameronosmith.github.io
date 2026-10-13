// claude-as-cad hero — lightweight mount (no three.js on the critical path).
//
// Paints the window chrome, the agent panel's history and a poster of the viewer immediately,
// then imports the WebGL hero (three.js + the arm asset, ~3 MB) only when the hero is near the
// viewport. hero.js adopts the DOM painted here, so there is no layout shift.
import { armBuildTemplate, editorPaneTemplate, galleryGrid, renderArmBuildHistory, renderEditorHistory, renderHistory, template } from "./template.js?v=bdf89f74e3";

export function lazyMountClaudeCadHero(root, options = {}) {
  if (!root) return;
  root.classList.add("cac-hero");
  const win = template();
  root.innerHTML = "";
  root.appendChild(win);
  renderHistory(win.querySelector(".cac-log"));
  const viewer = win.querySelector(".cac-viewer");
  viewer.style.setProperty("--cac-poster", `url("${new URL("assets/poster.webp", import.meta.url)}")`);
  win.querySelector(".cac-loading")?.remove();

  let started = false;
  let io = null;
  const start = () => {
    if (started) return;
    started = true;
    io?.disconnect();
    import("./hero.js?v=bdf89f74e3")
      .then(({ mountClaudeCadHero }) => mountClaudeCadHero(root, options))
      .catch((error) => console.warn("claude-as-cad hero failed to load", error));
  };
  if ("IntersectionObserver" in window) {
    io = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) start();
    }, { rootMargin: "300px 0px" });
    io.observe(win);
  } else {
    start();
  }
}

// Pane 2 ("02 · Primitive editor"): same lazy pattern, its own module + 0.8 MB scene.
export function lazyMountEditorPane(root, options = {}) {
  if (!root) return;
  root.classList.add("cac-hero");
  const win = editorPaneTemplate();
  root.innerHTML = "";
  root.appendChild(win);
  renderEditorHistory(win.querySelector(".cac-log"));
  win.querySelector(".cac-viewer").style.setProperty("--cac-poster", `url("${new URL("assets/poster_editor.webp", import.meta.url)}")`);
  let started = false;
  let io = null;
  const start = () => {
    if (started) return;
    started = true;
    io?.disconnect();
    import("./editor_pane.js?v=bdf89f74e3")
      .then(({ mountEditorPane }) => mountEditorPane(root, options))
      .catch((error) => console.warn("claude-as-cad editor pane failed to load", error));
  };
  if ("IntersectionObserver" in window) {
    io = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) start();
    }, { rootMargin: "300px 0px" });
    io.observe(win);
  } else {
    start();
  }
}

// Pane 3 ("03 · Hierarchical notes"): DOM only; fetches the 80 KB notes bundle when near the viewport.
export function lazyMountNotesPane(root, options = {}) {
  // DOM-only and light: mount right away so the page doesn't shift under the reader later
  eagerImport(root, () => import("./notes_pane.js?v=bdf89f74e3").then(({ mountNotesPane }) => mountNotesPane(root, options)));
}

// Pane 4 (multi-agent): DOM + ~1.4 MB of real footage, fetched only when the pane is near.
export function lazyMountMultiAgentPane(root, options = {}) {
  eagerImport(root, () => import("./multiagent_pane.js?v=bdf89f74e3").then(({ mountMultiAgentPane }) => mountMultiAgentPane(root, options)));
}

// Pane 5 (whole-arm build): chrome painted now, WebGL + the shared arm asset when near.
export function lazyMountArmBuildPane(root, options = {}) {
  if (!root) return;
  root.classList.add("cac-hero");
  const win = armBuildTemplate();
  root.innerHTML = "";
  root.appendChild(win);
  renderArmBuildHistory(win.querySelector(".cac-log"));
  let started = false;
  const start = () => {
    if (started) return;
    started = true;
    io?.disconnect();
    import("./armbuild_pane.js?v=bdf89f74e3")
      .then(({ mountArmBuildPane }) => mountArmBuildPane(root, options))
      .catch((error) => console.warn("claude-as-cad arm-build pane failed to load", error));
  };
  const io = "IntersectionObserver" in window ? new IntersectionObserver((entries) => {
    if (entries.some((entry) => entry.isIntersecting)) start();
  }, { rootMargin: "300px 0px" }) : null;
  if (io) io.observe(win);
  else start();
}

// Gallery (CAD | real grid): cards + photos painted now; three.js + each build load when near.
export function lazyMountGallery(root, options = {}) {
  if (!root) return;
  root.classList.add("cac-hero");
  root.innerHTML = "";
  root.appendChild(galleryGrid(options.descriptions || {}));
  let started = false;
  const io = "IntersectionObserver" in window ? new IntersectionObserver((entries) => {
    if (!started && entries.some((e) => e.isIntersecting)) {
      started = true;
      io.disconnect();
      import("./gallery_pane.js?v=bdf89f74e3").then(({ mountGallery }) => mountGallery(root, options)).catch((error) => console.warn("claude-as-cad gallery failed to load", error));
    }
  }, { rootMargin: "400px 0px" }) : null;
  if (io) io.observe(root);
  else import("./gallery_pane.js?v=bdf89f74e3").then(({ mountGallery }) => mountGallery(root, options));
}

// Intro viewer: the arm's gripper follows the cursor (loads right away; it's at the top of the page).
export function lazyMountFollowViewer(root, options = {}) {
  if (!root) return;
  import("./follow_pane.js?v=bdf89f74e3").then(({ mountFollowViewer }) => mountFollowViewer(root, options)).catch((error) => console.warn("claude-as-cad follow viewer failed to load", error));
}

function eagerImport(root, start) {
  if (root) start().catch((error) => console.warn("claude-as-cad pane failed to load", error));
}
