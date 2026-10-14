/*
 * Parameter-Suche für t-SNE (3D) auf dem Seeds-Datensatz.
 * Aufruf (im Projektordner):  node tools/optimize.js
 *
 * - nutzt dieselbe tsne.js wie die Webanwendung
 * - ersetzt Math.random durch einen Zufallsgenerator mit festem Startwert (mulberry32),
 *   damit jeder Lauf reproduzierbar ist
 * - bewertet die Läufe mit Silhouette-Koeffizient (Klassenlabels) und 5-NN-Genauigkeit im 3D-Raum
 * - schreibt die beste Lösung nach data/seeds_tsne3d.csv
 */
const fs = require("fs");
const path = require("path");
const root = path.join(__dirname, "..");

// ---------- Daten ----------
const lines = fs.readFileSync(path.join(root, "data/seeds.csv"), "utf8").trim().split(/\r?\n/).slice(1);
const raw = lines.map(l => l.split(",").map(Number));
const labels = raw.map(r => r[7]);
const feats = raw.map(r => r.slice(0, 7));
// z-Standardisierung je Merkmal (gleiche Vorverarbeitung wie in der Webanwendung)
const D = 7, N = feats.length;
const X = feats.map(r => r.slice());
for (let d = 0; d < D; d++) {
  let m = 0; for (const r of feats) m += r[d]; m /= N;
  let v = 0; for (const r of feats) v += (r[d] - m) ** 2; v = Math.sqrt(v / N) || 1;
  for (let i = 0; i < N; i++) X[i][d] = (feats[i][d] - m) / v;
}

// ---------- Hilfen ----------
function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function dist(a, b) { let s = 0; for (let k = 0; k < a.length; k++) s += (a[k] - b[k]) ** 2; return Math.sqrt(s); }

function silhouette(Y) {
  let sum = 0;
  for (let i = 0; i < N; i++) {
    const acc = {}, cnt = {};
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const c = labels[j];
      acc[c] = (acc[c] || 0) + dist(Y[i], Y[j]); cnt[c] = (cnt[c] || 0) + 1;
    }
    const a = acc[labels[i]] / cnt[labels[i]];
    let b = Infinity;
    for (const c in acc) if (+c !== labels[i]) b = Math.min(b, acc[c] / cnt[c]);
    sum += (b - a) / Math.max(a, b);
  }
  return sum / N;
}
function knnAcc(Y, k) {
  let ok = 0;
  for (let i = 0; i < N; i++) {
    const d = [];
    for (let j = 0; j < N; j++) if (j !== i) d.push([dist(Y[i], Y[j]), labels[j]]);
    d.sort((p, q) => p[0] - q[0]);
    const votes = {};
    for (let t = 0; t < k; t++) votes[d[t][1]] = (votes[d[t][1]] || 0) + 1;
    let best = null; for (const c in votes) if (best === null || votes[c] > votes[best]) best = c;
    if (+best === labels[i]) ok++;
  }
  return ok / N;
}
function freshTsne() {
  const p = require.resolve("../tsne.js");
  delete require.cache[p];             // frischer interner Zustand pro Lauf => reproduzierbar
  return require("../tsne.js");
}

// ---------- Suche ----------
const perplexities = [5, 10, 15, 20, 30, 40];
const epsilons = [5, 10, 20];
const checkpoints = [250, 500, 1000, 1500];
const seeds = [1, 2, 3, 4, 5];
// Als gespeicherte Lösung kommen nur eingeschwungene Läufe in Frage (nach 250/500 Iterationen ist t-SNE noch nicht konvergiert)
const MIN_ITER_FOR_BEST = 1000;

const agg = {};      // Schlüssel: perplexity|epsilon|iter  ->  Liste von {sil, knn, cost, seed}
let best = null;
const t0 = Date.now();

for (const perp of perplexities) {
  for (const eps of epsilons) {
    for (const seed of seeds) {
      Math.random = mulberry32(seed * 7919 + perp * 31 + eps);
      const tsnejs = freshTsne();
      const t = new tsnejs.tSNE({ dim: 3, perplexity: perp, epsilon: eps });
      t.initDataRaw(X);
      let cost = 0;
      for (let it = 1; it <= checkpoints[checkpoints.length - 1]; it++) {
        cost = t.step();
        if (checkpoints.includes(it)) {
          const Y = t.getSolution();
          const sil = silhouette(Y), knn = knnAcc(Y, 5);
          const key = perp + "|" + eps + "|" + it;
          (agg[key] = agg[key] || []).push({ sil, knn, cost, seed });
          if (it >= MIN_ITER_FOR_BEST && (!best || sil > best.sil)) {
            best = { sil, knn, cost, perp, eps, iter: it, seed, Y: Y.map(r => r.slice()) };
          }
        }
      }
    }
    console.error("fertig: perplexity", perp, "epsilon", eps, ((Date.now() - t0) / 1000).toFixed(0) + " s");
  }
}

// ---------- Ausgabe ----------
const rows = Object.keys(agg).map(k => {
  const [p, e, it] = k.split("|").map(Number);
  const L = agg[k], m = f => L.reduce((s, x) => s + x[f], 0) / L.length;
  return { perplexity: p, epsilon: e, iterations: it, silhouette: m("sil"), knn5: m("knn"), cost: m("cost"),
           bestSil: Math.max(...L.map(x => x.sil)) };
});
fs.writeFileSync(path.join(root, "tools/results.json"), JSON.stringify(rows, null, 1));

const out = ["# t-SNE 3D, Seeds-Datensatz (z-standardisiert), tsne.js (tSNEJS)",
  `# perplexity=${best.perp}`, `# epsilon=${best.eps}`, `# iterations=${best.iter}`,
  `# seed=${best.seed * 7919 + best.perp * 31 + best.eps}`, `# cost=${best.cost.toFixed(5)}`,
  `# silhouette=${best.sil.toFixed(4)}`, `# knn5=${best.knn.toFixed(4)}`, "x,y,z,class"];
best.Y.forEach((y, i) => out.push(y.map(v => v.toFixed(5)).join(",") + "," + labels[i]));
fs.writeFileSync(path.join(root, "data/seeds_tsne3d.csv"), out.join("\n") + "\n");
console.error("BESTE LÖSUNG:", JSON.stringify({ perp: best.perp, eps: best.eps, iter: best.iter, seed: best.seed,
  sil: best.sil, knn: best.knn, cost: best.cost }));
