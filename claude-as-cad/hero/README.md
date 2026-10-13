# claude-as-cad interactive hero

A self-contained static bundle (no build step, no backend) for the top of the claude-as-cad site:
the real arm in a light web fork of the viewer, next to a mock Claude Code panel.

- **Scripted replay** of click-to-reference loops: a virtual cursor clicks the two link_4 faces
  (red ring, "Copied to clipboard" toast, cyan face highlight), the panel types
  `join the faces of […] with the face […]` with ⌘V pastes, ⏎, and the trapezoidal connector grows.
  The UI labels it "scripted replay"; the panel text is the real session the reference clip re-enacts.
- Nothing the visitor does interrupts it: **orbit/pan/zoom during the replay** and its cursor, marker and
  toast track the faces wherever they are on screen (the replay stops driving the camera for that loop).
- **Click any face** (any time, also mid-replay, also on phones): the viewer's red ring, marker, cyan
  face highlight and "Copied to clipboard" toast, with a real reference — base-frame point, STL-local
  point + normal, the original STL triangle index (mapped from the web mesh at bake time) and side —
  copied to the clipboard.
- When a loop ends with the cursor over the viewer, the **gripper TCP follows the cursor**
  (damped-least-squares IK on J1–J5); leave the viewer and the replay resumes.
- Orbit/pan always work; **wheel zoom turns on after a click** into the viewer, so page scrolling
  isn't hijacked. On phones, vertical swipes scroll the page and taps aim the gripper.

## Mount

```html
<!-- fonts the hero is styled with (Inter + EB Garamond italic, both OFL via Google Fonts) -->
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@1,400;1,500&family=Inter:wght@400..600&display=swap">
<link rel="stylesheet" href="hero/hero.css?v=bdf89f74e3">

<div id="cac-hero"></div>
<p class="caption">Scripted replay of click-to-reference on the real arm — orbit and zoom while it runs; when it ends, the gripper follows your cursor.</p>

<script type="module">
  import { lazyMountClaudeCadHero } from "./hero/loader.js?v=bdf89f74e3";
  lazyMountClaudeCadHero(document.getElementById("cac-hero"), {
    fallbackVideo: "media/click_to_reference.mp4",   // shown if WebGL is unavailable
    fallbackPoster: "media/click_to_reference.jpg"
  });
</script>
```

`loader.js` paints the window chrome, the panel history and a 26 KB transparent viewer poster immediately
(no three.js), then imports the WebGL part (~2 MB) when the hero is within 300 px of the viewport.

Options: `fallbackVideo`, `fallbackPoster`, `autoplay` (default `true`), `debug` (exposes
`window.__cac`).

## Pane 2: "02 · Primitive editor"

A second interactive pane with the same viewer setup (embedded transparent viewer, orbit/zoom any
time, click any part to copy its reference): a scripted replay of the ST3095 holder session. Claude's
first holder comes out in the wrong orientation; a cursor deletes it in a mock of the primitive-editor
panel, adds a cube and drags its fields to Cameron's envelope (35.4 × 39.4 × 59.4 mm at (20, 0, −10)),
saves the edit log, and Claude carves the real holder from the box, showing the generator's real stages
(servo pocket → yoke-sweep relief → 7 counterbored screw holes). Assets: `assets/editor.{json,bin}`
(0.83 MB, from `tools/export_editor_scene.py`) + a 7 KB poster; lazy like pane 1.

```html
<div id="cac-editor"></div>
<script type="module">
  import { lazyMountEditorPane } from "./hero/loader.js?v=…";   // same stamped loader
  lazyMountEditorPane(document.getElementById("cac-editor"), {
    fallbackVideo: "media/primitive_editor.mp4", fallbackPoster: "media/primitive_editor.jpg"
  });
</script>
```

Honesty: the chat turns are the real session's (Sep 1, ~/cad_robot2); the editor steps are a re-enactment
with his exact box.

## Panes 3–5 and the page

- **03 · Hierarchical notes** (`notes_pane.js`, DOM only): a real slice of `~/cad` baked by
  `tools/export_notes.py` into `assets/notes.json` (scrubbed; the export fails if a host/IP/absolute path survives).
- **04 · Multi-agent** (`multiagent_pane.js`, DOM only): flowchart · footage (`assets/ma/`, ~1.4 MB, loaded when near) ·
  per-agent Claude tabs. Footage crops remove a Rerun title bar and a training-log header that named hosts/paths.
- **05 · A whole arm** (`armbuild_pane.js`, WebGL): clip 06 replayed 1:1 — the take's 47 stage files
  (`assets/armbuild2.*`, from `tools/export_armbuild_v2.py`), the director's camera moves, the 16 recorded clicks with the
  editor's own click-reference strings, ⌘C/⌘V copies + rename + typed/dragged fields in the primitive-editor panel, Save
  to edit log, live reloads and connector growth slices; dialogue from `arm_timeline4.py`
  (`tools/build_armbuild_script.py` → `assets/armbuild2_script.json`). A beat strip jumps to any step (fast-forwards state).
- **Table of contents** (`toc_arm.js`): `.toc` rail with a robot-arm pointer (≥1400 px).
- The whole page lives in `pages/index.html`; `python3 tools/build_pages.py [--site]` renders it to the preview
  (and the live site), stamps versions and syncs `hero/`.

## Look

Styled after skills.wheresryan.sh/anatomy: Inter 450 / 14 px, ink `#171717`, text-2 `#525252`,
muted `#8f8f8f`, faint `#bdbdbd`, rules `#ebebeb`, wells `#f1f1f1`, cards `#fff` with 12 px radius,
EB Garamond italic accents, page ground `#fafafa`. The viewer is **transparent with feathered edges**
(no frame), so the arm and its shadow sit on whatever the page background is; if the page isn't a
flat colour, set `--cac-ground` on `.cac-hero`. Red `#e0312b` is used only for the click ring/marker,
cyan for the face highlight. All tokens are CSS variables on `.cac-hero`.

## Behaviour and budgets

| | |
| --- | --- |
| Weight | ~2.2 MB, all lazy: three.js r160 min (656 KB), arm mesh 1.4 MB (44k triangles + STL triangle map), JS+CSS ~50 KB, poster ~26 KB (transparent WebP) |
| Rendering | renders only when something changes (replay tween, orbit damping, arm moving) |
| Off-screen / hidden tab | render loop and replay pause; replay restarts when visible again |
| `prefers-reduced-motion` | no autoplay or camera moves; shows the finished arm + transcript and a "Play the replay" button |
| No WebGL / load failure | swaps to `fallbackVideo` (or `fallbackPoster`) |
| Mobile (≤760 px) | panel stacks under the viewer; touch-action `pan-y` on coarse pointers |

## Files

```
hero/
  loader.js        lazy mount (no three.js)
  template.js      window chrome + panel history DOM
  hero.js          scene, replay timeline, TCP-follow, interaction
  arm.js           asset loader, URDF joint chain, IK
  hero.css
  vendor/          three.js r160 (three.module.min.js, OrbitControls.js, THREE_LICENSE — MIT)
  assets/          arm.json + arm.bin (baked arm), poster.webp
```

## Regenerating the arm asset

`arm.json`/`arm.bin` are baked from the URDF by `tools/export_hero_arm.py` (in the source tree,
not this bundle): decimated meshes with crease normals, the joint chain, the two connector faces
(exact coplanar triangles + real triangle indices) and the gripper TCP.

```bash
python export_hero_arm.py --urdf double_to_single_compact_long.urdf --out hero_assets
```

## Credits

- Arm: Cameron Smith's `double_to_single_compact_long` robot arm.
- Rendering: [three.js](https://threejs.org/) (MIT), including OrbitControls.
- Interaction and replay recreate the click-to-reference workflow of the claude-as-cad viewer.
