// claude-as-cad hero, pane 4 — "Multi-agent: CAD is only the first step".
//
//   import { mountMultiAgentPane } from "./hero/multiagent_pane.js";
//   mountMultiAgentPane(document.querySelector("#cac-agents"));
//
// Same layout as the video's multi-agent part: the agent flowchart on the left, each stage's real
// footage in the middle (screen recordings are videos, not replays), and the agents' Claude Code
// panels on the right. Click a stage to jump to it, click a tab to read any agent's transcript.
// The chats are re-enacted from the real sessions (strings as in the video); it auto-plays
// cad → calib_and_data_recording → model_training → deploy_model and loops.
import { el } from "./template.js?v=36b142ff4a";

const BASE_URL = new URL(".", import.meta.url);
const media = (name) => new URL(`assets/ma/${name}`, BASE_URL).href;

const AGENTS = [
  { id: "cad", sub: "CAD agent · URDF, ArUco boards", tab: "cad — claude", cwd: "~/cad_umi" },
  { id: "calib_and_data_recording", sub: "calibration + UMI data recording", tab: "calib_and_data_recording", cwd: "~/refactor_claude_feetech_controller" },
  { id: "model_training", sub: "model code + BIND training", tab: "model_training", cwd: "~/refactor_claude_feetech_controller" },
  { id: "deploy_model", sub: "runs policies on the arm", tab: "deploy_model", cwd: "~/refactor_claude_feetech_controller" }
];

// Each stage: title, media, and the agent's turns. Kinds: user | agent | tool (title + output lines) |
// counter (a tool line whose number ticks up) | handoff (sends the next agent's first message).
const STAGES = [
  {
    title: "CAD is only the first step",
    // ma_cad.mp4 = the editor side of the original render, its four prompt overlays cut out, at 2×.
    // Source segments [0,4.7] [9.0,27.2] [31.1,43.5] [48.6,58.8] s; `src(t)` maps an original-render
    // time to this clip, so every line lands when it did in the render. Prompts are typed while the
    // video holds at the cut where their overlay used to be.
    media: [{ kind: "video", src: "ma_cad.mp4", label: "EDITOR · 2× speed", ar: 832 / 720, sync: true }],
    sync: [
      { kind: "playUntil", t: "A" },
      { kind: "prompt", text: "lets build umi, open the eef in a new window so i can place a box for the arucos" },
      { kind: "tool", title: "Bash(Write umi_build_start.urdf: eef only)", out: ["grip: holder, brstub, servo, fixed fin, wrist cam + mount", "jaw: yoke, brstub, moving fin (q = 0)"], at: [9.0, 9.2, 9.3] },
      { kind: "tool", title: "Bash(open http://localhost:4180/?file=umi_build_start.urdf)", out: ["opened in a new tab"], at: [9.6, 9.8] },
      { kind: "agent", text: "EEF-only model is open in a new editor tab: localhost:4180/?file=umi_build_start.urdf. Add a box where the ArUco block should go, hit Save to edit log, and tell me.", at: 10.1 },
      { kind: "playUntil", t: "B" },
      { kind: "prompt", text: "i added a box in the editor, can you make it flush with the servo holder" },
      { kind: "tool", title: "Read(~/cad_editor_edits/EDIT_LOG.md)", out: ["## 2026-10-06 13:59:26 — umi_build_start.urdf", 'CUBE "cube1" (grip): center (6.0, -28.0, 270.0)mm | rot (0.0, 0.0, 0.0)deg | size 85.0 x 85.0 x 85.1mm'], at: [31.4, 31.7, 32.0] },
      { kind: "tool", title: "Bash(Editor cube -> grip frame; swap in real umi cube + snap flush)", out: ["cube1 in grip frame: center (6.0, 110.0, 28.0) mm, 85.05 mm box", "its bottom sits 1.06 mm into the holder top (y 68.53)", "real cube 66.6 mm: drop 8.16 mm -> flush, statics 0.0 mm³", "jaw sweep: moving brace + fin hit it past -70° (1866 mm³)", "slide +15 mm z (camera side) -> jaw sweep 0.0 mm³"], at: [32.6, 32.9, 33.3, 33.8, 35.2, 38.6] },
      { kind: "agent", text: "Swapped your box for the real UMI block and snapped it flush to the holder face (yours sat 1.1 mm into it). One catch the static scene can't show: when the gripper opens past 70°, the moving fin swings through the cube, so I slid it 15 mm toward the camera side. Full jaw sweep is clear now.", at: 40.0 },
      { kind: "playUntil", t: "C" },
      { kind: "prompt", text: "put aruco boards on the 5 visible sides of the umi block and send me the metric aruco sheet to print" },
      { kind: "tool", title: "Bash(Generate UMI cube boards + sheet)", out: ["umi_fr_top    +y  ids 79-82  2x2, 26.0 mm markers", "umi_fr_right  +x  ids 87-90", "umi_fr_front  +z  ids 91-94  (camera side)", "umi_fr_left   -x  ids 83-86", "umi_fr_rear   -z  ids 95-98", "-> fid_boards_umi_cube.pdf  (true scale, 66.6 mm stickers)"], at: [48.7, 49.0, 49.4, 49.8, 50.2, 50.6, 51.6] },
      { kind: "agent", text: "Boards are on all five exposed faces (the bottom is the mount). Sheet's ready: print at 100%, cut on the gray outlines, each sticker covers its face edge to edge. Opening it.", at: 52.4 },
      { kind: "playUntil", t: "end" },
      { kind: "prompt", text: "looks good, send the aruco boards and new umi model to the calib_and_data_recording agent to deploy the recording script" }
    ],
    turns: [
      { kind: "user", text: "lets build umi, open the eef in a new window so i can place a box for the arucos" },
      { kind: "tool", title: "Bash(Write umi_build_start.urdf: eef only)", out: ["grip: holder, brstub, servo, fixed fin, wrist cam + mount", "jaw: yoke, brstub, moving fin (q = 0)"] },
      { kind: "tool", title: "Bash(open http://localhost:4180/?file=umi_build_start.urdf)", out: ["opened in a new tab"] },
      { kind: "agent", text: "EEF-only model is open in a new editor tab: localhost:4180/?file=umi_build_start.urdf. Add a box where the ArUco block should go, hit Save to edit log, and tell me." },
      { kind: "user", text: "i added a box in the editor, can you make it flush with the servo holder" },
      { kind: "tool", title: "Read(~/cad_editor_edits/EDIT_LOG.md)", out: ["## 2026-10-06 13:59:26 — umi_build_start.urdf", 'CUBE "cube1" (grip): center (6.0, -28.0, 270.0)mm | rot (0.0, 0.0, 0.0)deg | size 85.0 x 85.0 x 85.1mm'] },
      { kind: "tool", title: "Bash(Editor cube -> grip frame; swap in real umi cube + snap flush)", out: ["cube1 in grip frame: center (6.0, 110.0, 28.0) mm, 85.05 mm box", "its bottom sits 1.06 mm into the holder top (y 68.53)", "real cube 66.6 mm: drop 8.16 mm -> flush, statics 0.0 mm³", "jaw sweep: moving brace + fin hit it past -70° (1866 mm³)", "slide +15 mm z (camera side) -> jaw sweep 0.0 mm³"] },
      { kind: "agent", text: "Swapped your box for the real UMI block and snapped it flush to the holder face (yours sat 1.1 mm into it). One catch the static scene can't show: when the gripper opens past 70°, the moving fin swings through the cube, so I slid it 15 mm toward the camera side. Full jaw sweep is clear now." },
      { kind: "user", text: "put aruco boards on the 5 visible sides of the umi block and send me the metric aruco sheet to print" },
      { kind: "tool", title: "Bash(Generate UMI cube boards + sheet)", out: ["umi_fr_top    +y  ids 79-82  2x2, 26.0 mm markers", "umi_fr_right  +x  ids 87-90", "umi_fr_front  +z  ids 91-94  (camera side)", "umi_fr_left   -x  ids 83-86", "umi_fr_rear   -z  ids 95-98", "-> fid_boards_umi_cube.pdf  (true scale, 66.6 mm stickers)"] },
      { kind: "agent", text: "Boards are on all five exposed faces (the bottom is the mount). Sheet's ready: print at 100%, cut on the gray outlines, each sticker covers its face edge to edge. Opening it." },
      { kind: "user", text: "looks good, send the aruco boards and new umi model to the calib_and_data_recording agent to deploy the recording script" },
      { kind: "handoff", to: 1, text: "check your inbox. cad: UMI handheld ready (urdf + meshes, 5 cube boards ids 79-98, print sheet). deploy the recording script on cameron's mac." }
    ]
  },
  {
    title: "Data: 50 UMI demos, pick cube → bowl",
    media: [
      { kind: "video", src: "ma_calib_rerun.mp4", label: "RERUN · live UMI tracking", note: "screen 1.9× speed", ar: 880 / 464 },
      { kind: "video", src: "ma_calib_live.mp4", label: "LIVE", dark: true, ar: 880 / 496 }
    ],
    turns: [
      { kind: "agent", text: "Handoff received: UMI URDF + cube boards (ids 79–98). Launching the Rerun recording script on Cameron’s Mac." },
      { kind: "tool", title: "Bash(python dataset/collect.py pick_cube_oct_9 --scene-discover)", out: [] },
      { kind: "counter", format: (n) => `recording on cameron's mac · episode ${n}/50`, to: 50, ms: 5200 },
      { kind: "handoff", to: 2, text: "calib_and_data_recording: Dataset ready: pick_cube_oct_9 (50 UMI episodes, pick cube → bowl). Please train BIND on it." }
    ]
  },
  {
    title: "Training: BIND action head",
    media: [{ kind: "image", src: "ma_training.webp", label: "/runs · bind_pick_cube_oct_9 · loss + val predictions", ar: 1100 / 730 }],
    turns: [
      { kind: "agent", text: "Got pick_cube_oct_9. Launching the same vanilla BIND config as the last run, logging to /runs/bind_pick_cube_oct_9." },
      { kind: "tool", title: "Bash(vanilla BIND training · dataset pick_cube_oct_9 → /runs/bind_pick_cube_oct_9)", out: [] },
      { kind: "counter", format: (n) => `training · step ${n.toLocaleString("en-US")} / 38,000`, to: 38000, ms: 4800 },
      { kind: "handoff", to: 3, text: "model_training: bind_pick_cube_oct_9 finished training (38k steps). Checkpoint: /runs/bind_pick_cube_oct_9/latest.pth. Ready to deploy on the arm." }
    ]
  },
  {
    title: "Deploy: the policy on the real arm",
    media: [
      { kind: "video", src: "ma_deploy_rerun.mp4", label: "RERUN · policy predictions", note: "screen 6.2× speed", ar: 880 / 464 },
      { kind: "video", src: "ma_deploy_live.mp4", label: "LIVE", dark: true, ar: 880 / 496 }
    ],
    turns: [
      { kind: "agent", text: "Pulling latest.pth from bind_pick_cube_oct_9. Launching the deploy script + Rerun viewer on Cameron’s Mac." },
      { kind: "tool", title: "Bash(deploy bind_pick_cube_oct_9 + rerun viewer on cameron's mac)", out: [] },
      { kind: "line", text: "policy running · pick cube → bowl", hold: 9000 },
      { kind: "line", text: "episode done · cube in the bowl", hold: 3000 }
    ]
  }
];

// original-render time (s) -> time in ma_cad.mp4 (four kept segments, played at 2×)
const CAD_SEGMENTS = [[0, 4.7], [9.0, 27.2], [31.1, 43.5], [48.6, 58.8]];
function cadTime(src) {
  let t = 0;
  for (const [a, b] of CAD_SEGMENTS) {
    if (src <= b) return (t + Math.max(0, src - a)) / 2;
    t += b - a;
  }
  return t / 2;
}
const CAD_MARKS = { A: cadTime(4.7), B: cadTime(27.2), C: cadTime(43.5), end: cadTime(58.8) };

const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
class Aborted extends Error {}

export function mountMultiAgentPane(root, options = {}) {
  if (!root) return null;
  root.classList.add("cac-hero");
  const win = el("div", "cac-window cac-pane-agents");
  win.setAttribute("role", "region");
  win.setAttribute("aria-label", "Interactive demo: four Claude Code agents handing a robot project from CAD to data, training and deployment");
  win.innerHTML = `
    <div class="cac-titlebar">
      <span class="cac-app cac-ma-title"></span>
      <span class="cac-url">· click a stage or an agent's tab</span>
      <span class="cac-tag" title="Footage is real; the agent chats re-enact the real sessions">agent chats re-enacted</span>
    </div>
    <div class="cac-ma-body">
      <ol class="cac-ma-flow"></ol>
      <div class="cac-ma-media"></div>
      <div class="cac-panel cac-ma-chat">
        <div class="cac-ma-tabs" role="tablist"></div>
        <div class="cac-panel-head cac-ma-cwd"></div>
        <div class="cac-log"></div>
        <div class="cac-input"><span class="cac-prompt">&gt;</span><span class="cac-input-text"></span><span class="cac-caret"></span></div>
      </div>
    </div>`;
  root.innerHTML = "";
  root.appendChild(win);
  const $ = (sel) => win.querySelector(sel);
  const ui = { title: $(".cac-ma-title"), flow: $(".cac-ma-flow"), media: $(".cac-ma-media"), tabs: $(".cac-ma-tabs"), cwd: $(".cac-ma-cwd"), log: $(".cac-log") };
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

  // ---------------------------------------------------------------- flowchart + tabs
  const stageEls = AGENTS.map((agent, i) => {
    const li = el("li", "cac-ma-stage", `<span class="cac-ma-n">${String(i + 1).padStart(2, "0")}</span><span class="cac-ma-state"></span><b>${agent.id}</b><small>${esc(agent.sub)}</small>`);
    li.tabIndex = 0;
    li.setAttribute("role", "button");
    li.addEventListener("click", () => jumpTo(i));
    li.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); jumpTo(i); } });
    ui.flow.appendChild(li);
    if (i < AGENTS.length - 1) ui.flow.appendChild(el("li", "cac-ma-arrow", "<i></i>"));
    return li;
  });
  const arrows = [...ui.flow.querySelectorAll(".cac-ma-arrow")];
  const tabEls = AGENTS.map((agent, i) => {
    const tab = el("button", "cac-ma-tab", `<span class="cac-ma-dot"></span>${esc(agent.tab)}`);
    tab.type = "button";
    tab.setAttribute("role", "tab");
    tab.addEventListener("click", () => viewAgent(i, true));
    ui.tabs.appendChild(tab);
    return tab;
  });

  // ---------------------------------------------------------------- transcripts (one log per agent)
  const logs = AGENTS.map(() => []); // rendered html lines, so tabs can be re-read any time
  const incoming = AGENTS.map(() => null); // a handoff waiting to be typed into an agent's chat
  let viewing = 0;
  function renderLog() {
    ui.log.innerHTML = logs[viewing].join("") || `<div class="cac-line cac-meta">no messages yet</div>`;
    ui.log.scrollTop = ui.log.scrollHeight;
  }
  function viewAgent(i, byUser = false) {
    viewing = i;
    tabEls.forEach((t, k) => {
      t.classList.toggle("cac-ma-tab-on", k === i);
      t.setAttribute("aria-selected", String(k === i));
    });
    tabEls[i].classList.remove("cac-ma-ping");
    ui.cwd.innerHTML = `<span class="cac-pdot"></span>${esc(AGENTS[i].id)} · ${esc(AGENTS[i].cwd)}`;
    renderLog();
    if (byUser) userViewed = performance.now();
  }
  function push(agent, html) {
    logs[agent].push(html);
    if (agent === viewing) {
      ui.log.insertAdjacentHTML("beforeend", html);
      ui.log.scrollTop = ui.log.scrollHeight;
    }
    return logs[agent].length - 1;
  }
  function replaceLast(agent, index, html) {
    logs[agent][index] = html;
    if (agent === viewing) renderLog();
  }
  const userLine = (text) => `<div class="cac-line cac-user">&gt; ${esc(text)}</div>`;
  const agentLine = (text) => `<div class="cac-line cac-agent"><span class="cac-bullet">●</span> ${esc(text)}</div>`;
  const toolLine = (title, out) => `<div class="cac-line cac-tool"><span class="cac-bullet cac-ok">●</span> ${esc(title)}<div class="cac-tool-out">${out.map((l, i) => `<div>${i === 0 ? "⎿ " : "&nbsp;&nbsp;"}${esc(l)}</div>`).join("")}</div></div>`;

  // ---------------------------------------------------------------- media
  let pendingMedia = null;
  let syncVideo = null;
  function showMedia(stage) {
    if (!visible) {
      pendingMedia = stage; // don't fetch footage until the pane is near the viewport
      return;
    }
    pendingMedia = null;
    syncVideo = null;
    for (const v of ui.media.querySelectorAll("video")) v.pause();
    ui.media.innerHTML = "";
    ui.media.classList.toggle("cac-ma-media-2", stage.media.length > 1);
    for (const m of stage.media) {
      const box = el("div", `cac-ma-shot${m.dark ? " cac-ma-dark" : ""}`);
      box.style.aspectRatio = String(m.ar || 16 / 9);
      box.dataset.ar = String(m.ar || 16 / 9);
      if (m.kind === "video") {
        const v = el("video");
        Object.assign(v, { src: media(m.src), muted: true, loop: !m.sync, playsInline: true, preload: "auto" });
        if (m.sync) syncVideo = v;
        v.setAttribute("muted", "");
        v.setAttribute("playsinline", "");
        box.appendChild(v);
        if (!reduceMotion && visible) v.play().catch(() => {});
      } else {
        const img = el("img");
        Object.assign(img, { src: media(m.src), alt: m.label, loading: "lazy" });
        box.appendChild(img);
      }
      box.appendChild(el("span", "cac-ma-label", esc(m.label)));
      if (m.note) box.appendChild(el("span", "cac-ma-note", esc(m.note)));
      ui.media.appendChild(box);
    }
    fitMedia();
  }
  // every box keeps its media's aspect ratio; a stack shares one width that fits the column
  function fitMedia() {
    const shots = [...ui.media.querySelectorAll(".cac-ma-shot")];
    if (!shots.length) return;
    const W = ui.media.clientWidth;
    const H = ui.media.clientHeight;
    const gap = 12;
    const inv = shots.reduce((sum, b) => sum + 1 / Number(b.dataset.ar), 0);
    const w = Math.max(80, Math.min(W, (H - gap * (shots.length - 1)) / inv));
    for (const b of shots) b.style.width = `${Math.floor(w)}px`;
  }
  new ResizeObserver(fitMedia).observe(ui.media);

  // ---------------------------------------------------------------- the run
  let token = { aborted: false };
  let current = -1;
  let userViewed = -1e9;
  const wait = (ms, tk) => new Promise((resolve, reject) => {
    const t = setTimeout(() => (tk.aborted ? reject(new Aborted()) : resolve()), ms);
    tk.timers = tk.timers || [];
    tk.timers.push(t);
  });
  const check = (tk) => { if (tk.aborted) throw new Aborted(); };

  function setStage(i) {
    current = i;
    ui.title.textContent = STAGES[i].title;
    stageEls.forEach((s, k) => {
      s.classList.toggle("cac-ma-on", k === i);
      s.classList.toggle("cac-ma-done", k < i);
      s.querySelector(".cac-ma-state").textContent = k < i ? "✓ done" : "";
    });
    arrows.forEach((a, k) => a.classList.toggle("cac-ma-arrow-done", k < i));
    showMedia(STAGES[i]);
    // follow the active agent unless the visitor just picked a tab to read
    if (performance.now() - userViewed > 12000) viewAgent(i);
  }
  // fill an agent's transcript instantly (used for stages that are already done when jumping)
  function fillStage(i, deferTo = -1) {
    for (const turn of STAGES[i].turns) {
      if (turn.kind === "user") push(i, userLine(turn.text));
      else if (turn.kind === "agent") push(i, agentLine(turn.text));
      else if (turn.kind === "tool") push(i, toolLine(turn.title, turn.out));
      else if (turn.kind === "counter") push(i, `<div class="cac-line cac-tool-out">⎿ ${esc(turn.format(turn.to))}</div>`);
      else if (turn.kind === "line") push(i, `<div class="cac-line cac-tool-out">⎿ ${esc(turn.text)}</div>`);
      else if (turn.kind === "handoff") {
        push(i, `<div class="cac-line cac-ma-sent">→ ${esc(AGENTS[turn.to].id)}: ${esc(turn.text)}</div>`);
        if (turn.to === deferTo) incoming[turn.to] = turn.text;
        else push(turn.to, userLine(turn.text));
      }
    }
  }
  // video clock for synced stages; falls back to wall time if the video stalls or can't play
  function videoClock(tk) {
    const v = syncVideo;
    const t0 = performance.now();
    let base = 0;
    let lastT = -1;
    let lastMove = performance.now();
    let wall = false;
    return {
      now() {
        if (!v || wall) return base + (performance.now() - t0) / 1000;
        if (v.currentTime !== lastT) {
          lastT = v.currentTime;
          lastMove = performance.now();
        } else if (!v.paused && performance.now() - lastMove > 3000) {
          wall = true; // stalled: keep the story moving on the wall clock
          base = v.currentTime - (performance.now() - t0) / 1000;
        }
        return v.currentTime;
      },
      async until(t) {
        if (v && !wall && v.paused && !v.ended) v.play().catch(() => { wall = true; });
        while (this.now() < t - 0.03 && !(v && v.ended)) {
          check(tk);
          await wait(40, tk);
        }
      },
      hold() { if (v) v.pause(); },
      resume() { if (v && !wall) v.play().catch(() => { wall = true; }); }
    };
  }
  async function typePrompt(i, text, tk) {
    const input = win.querySelector(".cac-ma-chat .cac-input-text");
    win.querySelector(".cac-ma-chat .cac-input").classList.add("cac-input-active");
    for (const ch of text) {
      check(tk);
      input.textContent += ch;
      await wait(ch === " " ? 12 : 20 + Math.random() * 18, tk);
    }
    await wait(250, tk);
    input.textContent = "";
    win.querySelector(".cac-ma-chat .cac-input").classList.remove("cac-input-active");
    push(i, userLine(text));
  }
  async function playSynced(i, tk) {
    const stage = STAGES[i];
    const clock = videoClock(tk);
    if (syncVideo) syncVideo.currentTime = 0;
    for (const step of stage.sync) {
      check(tk);
      if (step.kind === "playUntil") {
        clock.resume();
        await clock.until(CAD_MARKS[step.t]);
        clock.hold();
      } else if (step.kind === "prompt") {
        await typePrompt(i, step.text, tk);
        await wait(250, tk);
        clock.resume();
      } else if (step.kind === "tool") {
        await clock.until(cadTime(step.at[0]));
        const idx = push(i, toolLine(step.title, []));
        for (let k = 1; k <= step.out.length; k += 1) {
          await clock.until(cadTime(step.at[k]));
          replaceLast(i, idx, toolLine(step.title, step.out.slice(0, k)));
        }
      } else if (step.kind === "agent") {
        await clock.until(cadTime(step.at));
        const idx = push(i, agentLine(""));
        let text = "";
        for (const word of step.text.split(" ")) {
          text += `${word} `;
          replaceLast(i, idx, agentLine(text));
          await wait(18, tk);
        }
      }
    }
    const handoff = stage.turns[stage.turns.length - 1];
    push(i, `<div class="cac-line cac-ma-sent">→ ${esc(AGENTS[handoff.to].id)}: ${esc(handoff.text)}</div>`);
    arrows[i]?.classList.add("cac-ma-arrow-pulse");
    tabEls[handoff.to].classList.add("cac-ma-ping");
    await wait(1300, tk);
    arrows[i]?.classList.remove("cac-ma-arrow-pulse");
    incoming[handoff.to] = handoff.text;
    await wait(1000, tk);
  }
  async function playStage(i, tk) {
    setStage(i);
    if (incoming[i]) {
      const text = incoming[i];
      incoming[i] = null;
      await wait(450, tk);
      await typePrompt(i, text, tk); // the message arrives in this agent's chat as live typing
      await wait(300, tk);
    }
    if (STAGES[i].sync && syncVideo) {
      await playSynced(i, tk);
      return;
    }
    await wait(500, tk);
    for (const turn of STAGES[i].turns) {
      check(tk);
      if (turn.kind === "user") {
        push(i, userLine(turn.text));
        await wait(700, tk);
      } else if (turn.kind === "agent") {
        const idx = push(i, agentLine(""));
        let text = "";
        for (const word of turn.text.split(" ")) {
          text += `${word} `;
          replaceLast(i, idx, agentLine(text));
          await wait(28, tk);
        }
        await wait(500, tk);
      } else if (turn.kind === "tool") {
        const idx = push(i, toolLine(turn.title, []));
        for (let k = 1; k <= turn.out.length; k += 1) {
          await wait(160, tk);
          replaceLast(i, idx, toolLine(turn.title, turn.out.slice(0, k)));
        }
        await wait(450, tk);
      } else if (turn.kind === "counter") {
        const idx = push(i, `<div class="cac-line cac-tool-out">⎿ ${esc(turn.format(1))}</div>`);
        const steps = 40;
        for (let k = 1; k <= steps; k += 1) {
          await wait(turn.ms / steps, tk);
          const n = Math.max(1, Math.round((turn.to * k) / steps));
          replaceLast(i, idx, `<div class="cac-line cac-tool-out">⎿ ${esc(turn.format(n))}</div>`);
        }
        await wait(400, tk);
      } else if (turn.kind === "line") {
        push(i, `<div class="cac-line cac-tool-out">⎿ ${esc(turn.text)}</div>`);
        await wait(turn.hold, tk);
      } else if (turn.kind === "handoff") {
        push(i, `<div class="cac-line cac-ma-sent">→ ${esc(AGENTS[turn.to].id)}: ${esc(turn.text)}</div>`);
        arrows[i]?.classList.add("cac-ma-arrow-pulse");
        tabEls[turn.to].classList.add("cac-ma-ping");
        await wait(1300, tk);
        arrows[i]?.classList.remove("cac-ma-arrow-pulse");
        incoming[turn.to] = turn.text; // typed live into that agent's chat when its stage starts
      }
    }
    await wait(1200, tk);
  }
  async function run(from, tk) {
    try {
      for (;;) {
        for (let i = from; i < STAGES.length; i += 1) await playStage(i, tk);
        await wait(3000, tk);
        logs.forEach((l) => l.splice(0));
        incoming.fill(null);
        from = 0;
      }
    } catch (error) {
      if (!(error instanceof Aborted)) throw error;
    }
  }
  function stop() {
    token.aborted = true;
    for (const t of token.timers || []) clearTimeout(t);
  }
  // Jump: earlier stages complete instantly, then the chosen stage plays live and the run continues.
  function jumpTo(i) {
    stop();
    logs.forEach((l) => l.splice(0));
    incoming.fill(null);
    for (let k = 0; k < i; k += 1) fillStage(k, reduceMotion ? -1 : i);
    userViewed = -1e9;
    token = { aborted: false };
    if (reduceMotion) {
      fillStage(i);
      setStage(i);
      return;
    }
    run(i, token);
  }

  let visible = false;
  let started = false;
  new IntersectionObserver((entries) => {
    visible = entries.some((e) => e.isIntersecting);
    if (visible && pendingMedia) showMedia(pendingMedia);
    for (const v of ui.media.querySelectorAll("video")) {
      if (visible && !reduceMotion) v.play().catch(() => {});
      else v.pause();
    }
    if (visible && !started) {
      started = true;
      jumpTo(0);
    } else if (!visible && started && !reduceMotion) {
      stop(); // pause off-screen; pick the current stage up again when it comes back
    } else if (visible && started && !reduceMotion && token.aborted) {
      jumpTo(Math.max(0, current));
    }
  }, { threshold: 0.2 }).observe(win);
  if (reduceMotion) {
    for (let k = 0; k < STAGES.length; k += 1) fillStage(k);
    setStage(0);
  } else {
    setStage(0);
  }
  if (options.debug) window.__cacAgents = { jumpTo, get stage() { return current; }, get viewing() { return viewing; } };
  return { destroy: stop };
}
