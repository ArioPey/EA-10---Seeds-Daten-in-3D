"use strict";
/* Seeds-Daten in 3D: Daten laden (CSV), mit t-SNE (tSNEJS) auf 3 Dimensionen reduzieren
   und als interaktive 3D-Punktwolke darstellen. Reines WebGL. */

(function () {
  const DATA_URL = "seeds.csv";            // Originaldaten (7 Merkmale + Klasse)
  const SAVED_URL = "seeds_tsne3d.csv";    // gespeicherte, beste t-SNE-Lösung (3D)
  const CLASS_NAMES = { 1: "Kama", 2: "Rosa", 3: "Canadian" };
  const CLASS_COLORS = { 1: [0.90, 0.62, 0.00], 2: [0.80, 0.47, 0.65], 3: [0.00, 0.62, 0.45] };

  const $ = id => document.getElementById(id);
  const canvas = $("gl"), errBox = $("error");

  function showError(msg) {
    console.error(msg);
    errBox.hidden = false;
    errBox.textContent = msg;
  }

  /* ---------- CSV ---------- */
  // Zeilen mit # sind Metadaten (key=value); eine nicht-numerische erste Zeile ist die Kopfzeile.
  function parseCSV(text) {
    const rows = [], meta = {};
    text.split(/\r?\n/).forEach(line => {
      line = line.trim();
      if (!line) return;
      if (line[0] === "#") {
        const m = line.slice(1).trim().match(/^(\w+)\s*=\s*(.+)$/);
        if (m) meta[m[1]] = m[2];
        return;
      }
      const cells = line.split(",");
      const nums = cells.map(Number);
      if (nums.some(Number.isNaN)) return;       // Kopfzeile überspringen
      rows.push(nums);
    });
    return { rows, meta };
  }

  async function loadCSV(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error("Datei " + url + " konnte nicht geladen werden (HTTP " + res.status + ").");
    return parseCSV(await res.text());
  }

  // z-Standardisierung je Merkmal (Mittelwert 0, Standardabweichung 1)
  function standardize(feats) {
    const N = feats.length, D = feats[0].length;
    const out = feats.map(r => r.slice());
    for (let d = 0; d < D; d++) {
      let m = 0; for (let i = 0; i < N; i++) m += feats[i][d]; m /= N;
      let v = 0; for (let i = 0; i < N; i++) v += (feats[i][d] - m) ** 2; v = Math.sqrt(v / N) || 1;
      for (let i = 0; i < N; i++) out[i][d] = (feats[i][d] - m) / v;
    }
    return out;
  }

  /* ---------- Zustand ---------- */
  const state = {
    X: null,            // standardisierte Merkmale
    labels: null,       // Klassen 1..3
    saved: null,        // {Y, iter, cost, meta}
    params: { perplexity: 20, epsilon: 10 },
    tsne: null,         // laufende tSNE-Instanz (null = nur gespeicherte Lösung angezeigt)
    Y: null,            // aktuell angezeigte Positionen
    iter: 0,
    cost: null,
    source: "",
  };

  function refreshUI() {
    $("step-count").textContent = String(state.iter);
    $("cost").textContent = state.cost === null ? "–" : state.cost.toFixed(4);
    $("source").textContent = state.source;
  }

  function newTsne() {
    const t = new tsnejs.tSNE({ dim: 3, perplexity: state.params.perplexity, epsilon: state.params.epsilon });
    t.initDataRaw(state.X);
    return t;
  }

  // Restart: Step 0 mit neuen Zufallszahlen auf den Originaldaten
  function restart() {
    if (!state.X) return;
    state.tsne = newTsne();
    state.Y = state.tsne.getSolution();
    state.iter = 0;
    state.cost = null;
    state.source = "neu berechnet (Zufallsstart)";
    view.snap = true;
    refreshUI();
  }

  // Zeigt vor dem ersten Restart die gespeicherte Lösung. Der erste Step setzt von dort aus fort.
  function ensureTsne() {
    if (state.tsne) return;
    state.tsne = newTsne();
    if (state.saved) {
      state.tsne.Y = state.saved.Y.map(r => r.slice());
      state.tsne.iter = state.saved.iter;
      state.source = "gespeicherte Lösung, wird fortgesetzt";
    }
    state.Y = state.tsne.getSolution();
  }

  function doSteps(n) {
    if (!state.X) return;
    ensureTsne();
    let cost = state.cost;
    for (let i = 0; i < n; i++) cost = state.tsne.step();
    state.Y = state.tsne.getSolution();
    state.iter = state.tsne.iter;
    state.cost = cost;
    if (state.source.indexOf("fortgesetzt") < 0) state.source = "neu berechnet (Zufallsstart)";
    refreshUI();
  }

  /* ---------- WebGL ---------- */
  const gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
  if (!gl) { showError("WebGL wird von diesem Browser nicht unterstützt."); return; }

  function makeProgram(vsSrc, fsSrc) {
    function compile(type, src) {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error("Shader-Fehler:\n" + gl.getShaderInfoLog(sh));
      return sh;
    }
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error("Link-Fehler:\n" + gl.getProgramInfoLog(p));
    return p;
  }

  const POINT_VS = `
    attribute vec3 aPos;
    attribute vec3 aColor;
    uniform mat4 uView, uProj;
    uniform float uRadius;      // Kugelradius im Weltraum
    uniform float uHeight;      // Viewport-Höhe in Pixeln
    varying vec3 vColor;
    void main() {
      vec4 p = uProj * uView * vec4(aPos, 1.0);
      gl_Position = p;
      gl_PointSize = max(2.0, uRadius * uProj[1][1] * uHeight / p.w);
      vColor = aColor;
    }`;
  const POINT_FS = `
    precision mediump float;
    varying vec3 vColor;
    void main() {
      vec2 c = gl_PointCoord * 2.0 - 1.0;
      float r2 = dot(c, c);
      if (r2 > 1.0) discard;
      vec3 n = vec3(c.x, -c.y, sqrt(1.0 - r2));        // Kugelnormale
      vec3 L = normalize(vec3(-0.4, 0.6, 0.7));
      float diff = max(dot(n, L), 0.0);
      float spec = pow(max(dot(reflect(-L, n), vec3(0.0, 0.0, 1.0)), 0.0), 24.0);
      vec3 col = vColor * (0.35 + 0.75 * diff) + vec3(0.35) * spec;
      gl_FragColor = vec4(min(col, vec3(1.0)), 1.0);
    }`;
  const LINE_VS = `
    attribute vec3 aPos;
    uniform mat4 uView, uProj;
    void main() { gl_Position = uProj * uView * vec4(aPos, 1.0); }`;
  const LINE_FS = `
    precision mediump float;
    uniform vec3 uColor;
    void main() { gl_FragColor = vec4(uColor, 1.0); }`;

  let pointProg, lineProg;
  try {
    pointProg = makeProgram(POINT_VS, POINT_FS);
    lineProg = makeProgram(LINE_VS, LINE_FS);
  } catch (e) { showError(e.message); return; }

  const P = {
    aPos: gl.getAttribLocation(pointProg, "aPos"), aColor: gl.getAttribLocation(pointProg, "aColor"),
    uView: gl.getUniformLocation(pointProg, "uView"), uProj: gl.getUniformLocation(pointProg, "uProj"),
    uRadius: gl.getUniformLocation(pointProg, "uRadius"), uHeight: gl.getUniformLocation(pointProg, "uHeight"),
  };
  const Lc = {
    aPos: gl.getAttribLocation(lineProg, "aPos"),
    uView: gl.getUniformLocation(lineProg, "uView"), uProj: gl.getUniformLocation(lineProg, "uProj"),
    uColor: gl.getUniformLocation(lineProg, "uColor"),
  };

  const posBuf = gl.createBuffer(), colBuf = gl.createBuffer(), boxBuf = gl.createBuffer();

  // Würfelrahmen als Orientierungshilfe
  (function () {
    const s = 1.1, v = [];
    const c = [[-s,-s,-s],[s,-s,-s],[s,s,-s],[-s,s,-s],[-s,-s,s],[s,-s,s],[s,s,s],[-s,s,s]];
    [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]].forEach(e => v.push(...c[e[0]], ...c[e[1]]));
    gl.bindBuffer(gl.ARRAY_BUFFER, boxBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(v), gl.STATIC_DRAW);
  })();

  /* ---------- Matrizen ---------- */
  function perspective(fovy, aspect, n, f) {
    const t = 1 / Math.tan(fovy / 2), m = new Float32Array(16);
    m[0] = t / aspect; m[5] = t; m[10] = (f + n) / (n - f); m[11] = -1; m[14] = 2 * f * n / (n - f);
    return m;
  }
  function lookAt(e, c, u) {
    let zx = e[0]-c[0], zy = e[1]-c[1], zz = e[2]-c[2];
    let l = Math.hypot(zx, zy, zz); zx/=l; zy/=l; zz/=l;
    let xx = u[1]*zz - u[2]*zy, xy = u[2]*zx - u[0]*zz, xz = u[0]*zy - u[1]*zx;
    l = Math.hypot(xx, xy, xz); xx/=l; xy/=l; xz/=l;
    const yx = zy*xz - zz*xy, yy = zz*xx - zx*xz, yz = zx*xy - zy*xx;
    return new Float32Array([xx,yx,zx,0, xy,yy,zy,0, xz,yz,zz,0,
      -(xx*e[0]+xy*e[1]+xz*e[2]), -(yx*e[0]+yy*e[1]+yz*e[2]), -(zx*e[0]+zy*e[1]+zz*e[2]), 1]);
  }

  /* ---------- Kamera / Navigation ---------- */
  const cam = { yaw: 0.6, pitch: 0.35, dist: 4.4 };
  const camDefault = Object.assign({}, cam);
  const view = { scale: 1, snap: true };       // Skalierung der Punktwolke auf Einheitsgröße
  const clampPitch = p => Math.min(1.5, Math.max(-1.5, p));
  const clampDist = d => Math.min(14, Math.max(2, d));

  const keys = new Set();
  const NAV = ["arrowleft", "arrowright", "arrowup", "arrowdown", "w", "a", "s", "d", "q", "e", "+", "-", "="];
  window.addEventListener("keydown", e => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "T") { doSteps(10); e.preventDefault(); return; }        // Shift+T: Step10
    const k = e.key.toLowerCase();
    if (k === "t") { doSteps(1); e.preventDefault(); }
    else if (k === "r") { restart(); e.preventDefault(); }
    else if (k === "c") { Object.assign(cam, camDefault); }
    else if (NAV.includes(k)) { keys.add(k); e.preventDefault(); }
  });
  window.addEventListener("keyup", e => keys.delete(e.key.toLowerCase()));
  window.addEventListener("blur", () => keys.clear());

  $("btn-restart").addEventListener("click", restart);
  $("btn-step").addEventListener("click", () => doSteps(1));
  $("btn-step10").addEventListener("click", () => doSteps(10));
  $("btn-cam").addEventListener("click", () => Object.assign(cam, camDefault));

  let drag = null;
  canvas.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", e => {
    if (!drag) return;
    cam.yaw -= (e.clientX - drag.x) * 0.008;
    cam.pitch = clampPitch(cam.pitch + (e.clientY - drag.y) * 0.008);
    drag.x = e.clientX; drag.y = e.clientY;
  });
  const endDrag = () => { drag = null; };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    cam.dist = clampDist(cam.dist * (1 + Math.sign(e.deltaY) * 0.08));
  }, { passive: false });

  function updateCamera(dt) {
    const rot = 1.6 * dt, zoom = 1.0 * dt;
    if (keys.has("arrowleft")  || keys.has("a")) cam.yaw -= rot;
    if (keys.has("arrowright") || keys.has("d")) cam.yaw += rot;
    if (keys.has("arrowup")    || keys.has("w")) cam.pitch = clampPitch(cam.pitch + rot);
    if (keys.has("arrowdown")  || keys.has("s")) cam.pitch = clampPitch(cam.pitch - rot);
    if (keys.has("q") || keys.has("-")) cam.dist = clampDist(cam.dist * (1 + zoom));
    if (keys.has("e") || keys.has("+") || keys.has("=")) cam.dist = clampDist(cam.dist * (1 - zoom));
  }

  /* ---------- Rendern ---------- */
  let N = 0, posArr = null, colArr = null;

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }

  function uploadPoints() {
    // Punktwolke auf Radius 1 normieren (t-SNE-Werte wachsen während der Iterationen)
    let maxR = 1e-12;
    for (let i = 0; i < N; i++) {
      const y = state.Y[i];
      maxR = Math.max(maxR, Math.hypot(y[0], y[1], y[2]));
    }
    const target = 1 / maxR;
    if (view.snap) { view.scale = target; view.snap = false; }
    else view.scale = Math.exp(Math.log(view.scale) + (Math.log(target) - Math.log(view.scale)) * 0.3);
    for (let i = 0; i < N; i++) {
      const y = state.Y[i];
      posArr[i*3] = y[0] * view.scale; posArr[i*3+1] = y[1] * view.scale; posArr[i*3+2] = y[2] * view.scale;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, posArr, gl.DYNAMIC_DRAW);
  }

  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(0.06, 0.07, 0.10, 1);

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    updateCamera(dt);
    resize();
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    if (state.Y) {
      const eye = [
        cam.dist * Math.cos(cam.pitch) * Math.sin(cam.yaw),
        cam.dist * Math.sin(cam.pitch),
        cam.dist * Math.cos(cam.pitch) * Math.cos(cam.yaw)
      ];
      const V = lookAt(eye, [0, 0, 0], [0, 1, 0]);
      const Pm = perspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 100);

      gl.useProgram(lineProg);
      gl.uniformMatrix4fv(Lc.uView, false, V);
      gl.uniformMatrix4fv(Lc.uProj, false, Pm);
      gl.uniform3f(Lc.uColor, 0.28, 0.32, 0.40);
      gl.bindBuffer(gl.ARRAY_BUFFER, boxBuf);
      gl.enableVertexAttribArray(Lc.aPos);
      gl.vertexAttribPointer(Lc.aPos, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.LINES, 0, 24);

      uploadPoints();
      gl.useProgram(pointProg);
      gl.uniformMatrix4fv(P.uView, false, V);
      gl.uniformMatrix4fv(P.uProj, false, Pm);
      gl.uniform1f(P.uRadius, 0.035);
      gl.uniform1f(P.uHeight, canvas.height);
      gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
      gl.enableVertexAttribArray(P.aPos);
      gl.vertexAttribPointer(P.aPos, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
      gl.enableVertexAttribArray(P.aColor);
      gl.vertexAttribPointer(P.aColor, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.POINTS, 0, N);
    }
    requestAnimationFrame(frame);
  }

  /* ---------- Start ---------- */
  function fillLegend(labels) {
    const counts = {};
    labels.forEach(l => { counts[l] = (counts[l] || 0) + 1; });
    const ul = $("legend");
    ul.innerHTML = "";
    [1, 2, 3].forEach(c => {
      const li = document.createElement("li");
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.style.background = "rgb(" + CLASS_COLORS[c].map(v => Math.round(v * 255)).join(",") + ")";
      li.appendChild(chip);
      li.appendChild(document.createTextNode(CLASS_NAMES[c] + " (" + (counts[c] || 0) + ")"));
      ul.appendChild(li);
    });
  }

  function fillParams() {
    const m = state.saved ? state.saved.meta : {};
    $("p-perp").textContent = state.params.perplexity;
    $("p-eps").textContent = state.params.epsilon;
    $("p-iter").textContent = state.saved ? state.saved.iter : "–";
    $("p-sil").textContent = m.silhouette || "–";
    $("p-knn").textContent = m.knn5 ? (Number(m.knn5) * 100).toFixed(1) + " %" : "–";
    $("p-seed").textContent = m.seed || "–";
  }

  async function init() {
    try {
      const [data, saved] = await Promise.all([loadCSV(DATA_URL), loadCSV(SAVED_URL)]);
      if (data.rows.length === 0 || data.rows[0].length !== 8) throw new Error("Die Datei " + DATA_URL + " hat nicht das erwartete Format (7 Merkmale + Klasse).");
      if (saved.rows.length !== data.rows.length) throw new Error("Anzahl der Zeilen in " + SAVED_URL + " und " + DATA_URL + " stimmt nicht überein.");

      state.labels = data.rows.map(r => r[7]);
      state.X = standardize(data.rows.map(r => r.slice(0, 7)));
      const meta = saved.meta;
      if (Number(meta.perplexity) > 0) state.params.perplexity = Number(meta.perplexity);
      if (Number(meta.epsilon) > 0) state.params.epsilon = Number(meta.epsilon);
      state.saved = {
        Y: saved.rows.map(r => r.slice(0, 3)),
        iter: Number(meta.iterations) || 0,
        cost: meta.cost !== undefined ? Number(meta.cost) : null,
        meta
      };

      N = state.labels.length;
      posArr = new Float32Array(N * 3);
      colArr = new Float32Array(N * 3);
      state.labels.forEach((l, i) => colArr.set(CLASS_COLORS[l] || [1, 1, 1], i * 3));
      gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
      gl.bufferData(gl.ARRAY_BUFFER, colArr, gl.STATIC_DRAW);

      // Beim Start: gespeicherte beste Lösung anzeigen
      state.Y = state.saved.Y;
      state.iter = state.saved.iter;
      state.cost = state.saved.cost;
      state.source = "gespeicherte Lösung";
      fillLegend(state.labels);
      fillParams();
      refreshUI();
    } catch (e) {
      showError(e.message + "\nTipp: Seite über einen Webserver öffnen (nicht über file://), z. B. mit python3 -m http.server.");
    }
  }

  init();
  requestAnimationFrame(frame);
})();
