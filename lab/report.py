"""Visual error report: every train pair with saved predictions drawn on its images, worst pairs first.

    python report.py ~/Downloads/oof_dinov2_l_reg_840_mask0.0.json        # its 'fine-tuned + lab' predictions
    python report.py "lab_out/preds.json:zoom 1.0" --out lab_out/zoom      # any lab.load_preds spec
Writes OUT/report.html. The page links the images in --data instead of copying them, so open it where the data is."""
import argparse
import html
import json
import os
from pathlib import Path

import lab as L
import semcorr as sc

SIZES = ("small", "medium", "large")


def build(data, spec, out):
    pairs, preds, out = sc.load_split(data, "train"), L.load_preds(spec), Path(out)
    kind = L.error_kinds(preds, pairs)
    out.mkdir(parents=True, exist_ok=True)
    r1 = lambda v: round(float(v), 1)
    # per query: source x, y, predicted x, y, error kind, size bucket, scale, accepted targets
    rows = [{"id": p.pair_id, "ss": p.source_size, "ts": p.target_size,
             "src": os.path.relpath(Path(data) / p.source_image, out),
             "tgt": os.path.relpath(Path(data) / p.target_image, out),
             "q": [[*map(r1, q.src_xy), *map(r1, preds[q.id]), L.KINDS.index(kind[q.id]),
                    SIZES.index(L.size_bucket(p, q)), r1(q.scale), [[r1(x), r1(y)] for x, y in q.targets]]
                   for q in p.queries]} for p in pairs]
    page = PAGE.replace("__SPEC__", html.escape(Path(spec).name)).replace("__DATA__", json.dumps(rows).replace("</", "<\\/"))
    (out / "report.html").write_text(page)
    print(L.taxonomy(preds, pairs))
    return out / "report.html"


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Matching error report</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --border: rgba(11, 11, 11, .10);
  --k0: #2a78d6; --k1: #eb6834; --k2: #1baf7a; --k3: #eda100; --k4: #e87ba4;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7;
    --grid: #2c2c2a; --border: rgba(255, 255, 255, .10);
    --k0: #3987e5; --k1: #d95926; --k2: #199e70; --k3: #c98500; --k4: #d55181;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 32px 0 4px; }
.sub { color: var(--ink-2); margin: 0 0 12px; max-width: 80ch; }
.tiles { display: flex; flex-wrap: wrap; gap: 12px; }
code { overflow-wrap: anywhere; }
.tile { flex: 1 1 0; min-width: 96px; max-width: 220px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px 16px; }
.tile b { display: block; font-size: 26px; font-weight: 600; }
.tile span { color: var(--ink-2); font-size: 12px; }
.panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; color: var(--ink-2); font-size: 12px; margin: 0 0 10px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 10px; height: 10px; border-radius: 2px; }
.row { display: grid; grid-template-columns: 200px 1fr 56px; gap: 4px 12px; align-items: center; margin: 8px 0; }
@media (max-width: 640px) { .row { grid-template-columns: 1fr 56px; } .row .lab { grid-column: 1 / -1; } }
.row .lab { color: var(--ink-2); font-size: 12px; }
.row .n { color: var(--muted); font-size: 12px; text-align: right; font-variant-numeric: tabular-nums; }
.bar { display: flex; gap: 2px; height: 20px; }
.seg { flex-basis: 0; min-width: 2px; cursor: pointer; color: #fff; font-size: 12px; font-weight: 600; line-height: 20px; text-indent: 6px; white-space: nowrap; }
.seg:last-child { border-radius: 0 4px 4px 0; }
.seg:hover, .seg:focus-visible { filter: brightness(1.15); outline: none; }
details { margin-top: 12px; color: var(--ink-2); font-size: 12px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; margin-top: 8px; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 10px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
#tip { position: fixed; z-index: 10; pointer-events: none; display: none; max-width: 280px; background: var(--surface); color: var(--ink);
  border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; font-size: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, .18); }
#tip b { font-size: 14px; }
.filters { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; margin: 12px 0; }
select { font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 5px 8px; margin-left: 6px; }
#count { color: var(--ink-2); }
#g { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(520px, 100%), 1fr)); gap: 12px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 8px; cursor: zoom-in; }
.card.big { grid-column: 1 / -1; cursor: zoom-out; }
.card h3 { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 2px 12px; margin: 0 0 6px; font-size: 12px; font-weight: 500; color: var(--ink-2); }
.imgs { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; align-items: start; }
figure { position: relative; margin: 0; }
figure img { display: block; width: 100%; height: auto; border-radius: 4px; background: var(--grid); }
figure svg { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; }
.k0 { fill: var(--k0); } .k1 { fill: var(--k1); } .k2 { fill: var(--k2); } .k3 { fill: var(--k3); } .k4 { fill: var(--k4); }
.dot { stroke: #fff; stroke-width: 1.5; vector-effect: non-scaling-stroke; }
.gt { fill: none; stroke: #fff; stroke-width: 1.5; vector-effect: non-scaling-stroke; }
.halo { fill: none; stroke: rgba(0, 0, 0, .55); stroke-width: 3.5; vector-effect: non-scaling-stroke; }
.ln { stroke: #fff; stroke-width: 1.5; vector-effect: non-scaling-stroke; }
.num { fill: #fff; stroke: rgba(0, 0, 0, .8); paint-order: stroke; font-weight: 700; }
.tol { display: none; fill: rgba(255, 255, 255, .15); stroke: #fff; stroke-width: 1; vector-effect: non-scaling-stroke; }
.hit { fill: transparent; }
.m { transition: opacity .12s; }
#g[data-focus="1"] .m:not([data-k="1"]), #g[data-focus="2"] .m:not([data-k="2"]),
#g[data-focus="3"] .m:not([data-k="3"]), #g[data-focus="4"] .m:not([data-k="4"]) { opacity: .25; }
#g .card.hl .m:not(.on) { opacity: .12; }
#g .card.hl .m.on { opacity: 1; }
#g .card.hl .m.on .tol { display: inline; }
</style>
</head>
<body>
<main>
<h1>Where the matcher misses</h1>
<p class="sub">Predictions <code>__SPEC__</code> on the labelled train pairs (the test set has no labels). With an out-of-fold file, each pair was
predicted by a model that never saw its images. A query counts as correct when the prediction lands within 10% of the object's
bounding-box long side from an accepted answer.</p>
<div class="tiles" id="tiles"></div>

<h2>Outcome by object size and by repeated parts</h2>
<p class="sub">Share of queries per outcome. Click a segment to list those pairs below.</p>
<div class="panel">
  <div class="legend kinds"></div>
  <div id="chart"></div>
  <details><summary>Table view</summary><div class="scroll"><table id="tbl"></table></div></details>
</div>

<h2>Every pair, worst first</h2>
<p class="sub">Left: source image with the query points. Right: target image with our predictions. Hover a point to isolate it
and see its tolerance circle; click a card to enlarge it.</p>
<div class="legend kinds"></div>
<div class="legend">
  <span><svg width="16" height="16"><circle class="dot k0" cx="8" cy="8" r="4.5"/></svg>prediction, colored by outcome</span>
  <span><svg width="16" height="16"><circle class="halo" cx="8" cy="8" r="6"/><circle class="gt" cx="8" cy="8" r="6"/></svg>accepted answer</span>
  <span><svg width="24" height="14"><line class="halo" x1="2" y1="7" x2="22" y2="7"/><line class="ln" x1="2" y1="7" x2="22" y2="7"/></svg>miss to its nearest accepted answer</span>
</div>
<div class="filters">
  <label>Pairs with<select id="kind">
    <option value="">any outcome</option><option value="1">a near miss</option><option value="2">a hit on another part</option>
    <option value="3">a hit elsewhere on the object</option><option value="4">a hit on the background</option><option value="ok">no errors</option>
  </select></label>
  <label>Object size<select id="size">
    <option value="">any</option><option value="small">small (under 25% of frame)</option><option value="medium">medium</option><option value="large">large (60%+)</option>
  </select></label>
  <span id="count"></span>
</div>
<div id="g"></div>
</main>
<div id="tip" role="tooltip"></div>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const KINDS = ['correct', 'near miss (0.1–0.2)', 'another part', 'on object, elsewhere', 'background / other instance'];
const SHORT = ['correct', 'near miss', 'another part', 'elsewhere on object', 'background'];
const SIZES = ['small', 'medium', 'large'];
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => Object.assign(document.createElement(tag), {className: cls, textContent: text});
const esc = s => String(s).replace(/[&<>"]/g, c => `&#${c.charCodeAt(0)};`);
const pct = (a, b) => (100 * a / Math.max(1, b)).toFixed(1) + '%';
const Q = D.flatMap(p => p.q);  // [sx, sy, px, py, kind, size, scale, targets]

const nOk = Q.filter(q => q[4] === 0).length;
for (const [v, l] of [[pct(nOk, Q.length), 'PCK@0.10'], [D.length, 'pairs'], [Q.length.toLocaleString(), 'queries']]) {
  const t = el('div', 'tile'); t.append(el('b', '', v), el('span', '', l)); $('tiles').append(t);
}
for (const box of document.querySelectorAll('.kinds')) KINDS.forEach((k, i) => {
  const s = el('span', ''), sw = el('i', 'sw'); sw.style.background = `var(--k${i})`; s.append(sw, k); box.append(s);
});

const tip = $('tip');
function showTip(s, x, y) {
  tip.replaceChildren(el('b', '', s.dataset.pct), ' ' + s.dataset.kind, document.createElement('br'), s.dataset.ctx);
  tip.style.display = 'block';
  tip.style.left = Math.max(8, Math.min(x + 12, innerWidth - tip.offsetWidth - 8)) + 'px';
  tip.style.top = (y + 14) + 'px';
}
const GROUPS = [
  ['All queries', () => true, ''],
  ['Small objects (under 25% of frame)', q => q[5] === 0, 'small'],
  ['Medium objects (25–60%)', q => q[5] === 1, 'medium'],
  ['Large objects (60%+)', q => q[5] === 2, 'large'],
  ['One accepted answer', q => q[7].length === 1, ''],
  ['2+ accepted answers (repeated parts)', q => q[7].length > 1, ''],
];
const head = $('tbl').createTHead().insertRow();
for (const t of ['', 'queries', ...SHORT]) head.append(el('th', '', t));
for (const [label, f, size] of GROUPS) {
  const sel = Q.filter(f), n = KINDS.map((_, k) => sel.filter(q => q[4] === k).length);
  const bar = el('div', 'bar');
  n.forEach((c, k) => {
    if (!c) return;
    const s = el('div', 'seg', k ? '' : pct(c, sel.length));  // direct label only on the correct share
    s.tabIndex = 0; s.style.flexGrow = c; s.style.background = `var(--k${k})`;
    Object.assign(s.dataset, {pct: pct(c, sel.length), kind: KINDS[k], ctx: `${label} · ${c} of ${sel.length} queries`});
    s.onpointermove = e => showTip(s, e.clientX, e.clientY);
    s.onpointerleave = s.onblur = () => { tip.style.display = 'none'; };
    s.onfocus = () => { const r = s.getBoundingClientRect(); showTip(s, r.left, r.bottom); };
    s.onclick = () => { $('kind').value = k ? String(k) : ''; $('size').value = size; render(); $('g').scrollIntoView({behavior: 'smooth'}); };
    bar.append(s);
  });
  const row = el('div', 'row'); row.append(el('div', 'lab', label), bar, el('div', 'n', `n=${sel.length}`)); $('chart').append(row);
  const tr = $('tbl').insertRow();
  for (const t of [label, sel.length, ...n.map(c => pct(c, sel.length))]) tr.insertCell().textContent = t;
}

const ORDER = D.map(p => ({p, ok: p.q.filter(q => q[4] === 0).length}))
  .sort((a, b) => a.ok / a.p.q.length - b.ok / b.p.q.length || b.p.q.length - a.p.q.length);
const fig = (src, [w, h], marks) =>
  `<figure><img src="${esc(src)}" width="${w}" height="${h}" loading="lazy" decoding="async" alt=""><svg viewBox="0 0 ${w} ${h}">${marks}</svg></figure>`;
const dot = (x, y, k, u, i) => `<circle class="dot k${k}" cx="${x}" cy="${y}" r="${1.8 * u}"/>` +
  `<text class="num" x="${x + 2.2 * u}" y="${y - 1.8 * u}" font-size="${4.2 * u}" stroke-width="${0.9 * u}">${i + 1}</text>` +
  `<circle class="hit" cx="${x}" cy="${y}" r="${4.5 * u}"/>`;
function card({p, ok}) {
  const us = Math.max(...p.ss) / 100, ut = Math.max(...p.ts) / 100;  // marker unit: 1% of the image's long side
  const src = p.q.map((q, i) => `<g class="m" data-q="${i}" data-k="${q[4]}">${dot(q[0], q[1], q[4], us, i)}</g>`).join('');
  const tgt = p.q.map((q, i) => {
    const [, , x, y, k, , s, T] = q;
    const near = T.reduce((a, t) => Math.hypot(t[0] - x, t[1] - y) < Math.hypot(a[0] - x, a[1] - y) ? t : a);
    const off = Math.hypot(near[0] - x, near[1] - y) / (0.1 * s);
    return `<g class="m" data-q="${i}" data-k="${k}"><title>query ${i + 1}: ${KINDS[k]}, ${off.toFixed(1)}× the tolerance from the nearest accepted answer</title>` +
      T.map(t => `<circle class="tol" cx="${t[0]}" cy="${t[1]}" r="${0.1 * s}"/>`).join('') +
      T.map(t => `<circle class="halo" cx="${t[0]}" cy="${t[1]}" r="${2.8 * ut}"/><circle class="gt" cx="${t[0]}" cy="${t[1]}" r="${2.8 * ut}"/>`).join('') +
      (k ? `<line class="halo" x1="${x}" y1="${y}" x2="${near[0]}" y2="${near[1]}"/><line class="ln" x1="${x}" y1="${y}" x2="${near[0]}" y2="${near[1]}"/>` : '') +
      dot(x, y, k, ut, i) + '</g>';
  }).join('');
  const errs = [1, 2, 3, 4].map(k => [k, p.q.filter(q => q[4] === k).length]).filter(([, c]) => c).map(([k, c]) => `${c} ${SHORT[k]}`);
  const size = [...new Set(p.q.map(q => SIZES[q[5]]))].join('/');
  return `<article class="card"><h3><span>${esc(p.id)} · ${size} object</span><span>${[`${ok}/${p.q.length} correct`, ...errs].join(' · ')}</span></h3>` +
    `<div class="imgs">${fig(p.src, p.ss, src)}${fig(p.tgt, p.ts, tgt)}</div></article>`;
}
function render() {
  const k = $('kind').value, s = $('size').value;
  const list = ORDER.filter(({p, ok}) => (!s || p.q.some(q => SIZES[q[5]] === s)) &&
    (!k || (k === 'ok' ? ok === p.q.length : p.q.some(q => q[4] === +k))));
  $('g').dataset.focus = k === 'ok' ? '' : k;
  $('count').textContent = `${list.length} of ${D.length} pairs`;
  $('g').innerHTML = list.map(card).join('');
}
$('kind').onchange = $('size').onchange = render;
const g = $('g');
g.addEventListener('pointerover', e => {  // hovering a query isolates it in both images
  const c = e.target.closest('.card'), m = e.target.closest('.m');
  if (!c) return;
  c.classList.toggle('hl', !!m);
  for (const x of c.querySelectorAll('.m')) x.classList.toggle('on', !!m && x.dataset.q === m.dataset.q);
});
g.addEventListener('pointerout', e => { const c = e.target.closest('.card'); if (c && !c.contains(e.relatedTarget)) c.classList.remove('hl'); });
g.addEventListener('click', e => e.target.closest('.card')?.classList.toggle('big'));
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", help="lab.load_preds spec: file.json[:key]")
    ap.add_argument("--data", default="../aml-challenge-2026-27")
    ap.add_argument("--out", default="lab_out")
    a = ap.parse_args()
    print("wrote", build(a.data, a.preds, a.out))
