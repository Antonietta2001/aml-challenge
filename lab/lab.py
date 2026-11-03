"""Idea lab for the AML semantic-correspondence challenge, built on the colleague's pipeline (semcorr.py).

Every inference idea is a switch of one matcher, so a single cached feature extraction serves all tests:
  context=queries  flip choice / RANSAC prior use only the queried source points
                   (the non-queried source points are mostly invisible in the target)
  cycle=b          forward-backward consistency: penalise target cells whose best source match is not the query
                   (look-alike background, other instances, wrong part)
  zoom=w           re-match inside square crops framed on the keypoints of both images, fused with the
                   full-image map (small objects get more patches, and both crops show the parts at a similar scale)
  geo_lambda=l     the colleague's RANSAC-affine prior

Training idea (BERT/MAE principle): --mask-ratio r hides a random r of the tokens entering the fine-tuned blocks, so
their features must be inferred from context (semcorr.TailModel._mask).

    python lab.py selftest
    python lab.py ablate  --data D        # zero-shot rows (ABLATION): PCK, paired-bootstrap CI vs baseline, taxonomy
    python lab.py ft-cv   --data D --mask-ratio 0.3 --cfg '{"cycle": 1, "zoom": 0.5}'   # CV: zero-shot / FT / FT+ideas
    python lab.py compare --data D "lab_out/oof_A.json:fine-tuned" "lab_out/oof_B.json:fine-tuned"   # paired test
    python lab.py predict --data D --finetune --mask-ratio 0.3 --cfg '{"cycle": 1, "zoom": 0.5}'
"""
import argparse
import dataclasses
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import semcorr as sc


@dataclasses.dataclass
class LabConfig(sc.MatchConfig):
    cycle: float = 0.0     # weight of the forward-backward term (0 = off)
    zoom: float = 0.0      # weight of the zoomed-crop similarity inside the crop (0 = off)
    zoom_pad: float = 0.5  # crop margin on each side, as a fraction of the keypoint-hull long side
    zoom_max: float = 4.0  # max magnification: crop long side >= image long side / zoom_max


def queries_as_context(p: sc.Pair) -> sc.Pair:
    qs = [dataclasses.replace(q, query_index=i) for i, q in enumerate(p.queries)]
    return dataclasses.replace(p, source_xy=np.array([q.src_xy for q in p.queries], np.float32), queries=qs)


def crop_box(pts, size_wh, pad, zoom_max):
    """Square (x0,y0,x1,y1) around `pts`, enlarged by `pad`, at least long side / zoom_max, kept inside the image."""
    W, H = size_wh
    lo, hi = pts.min(0), pts.max(0)
    side = min(max((hi - lo).max() * (1 + 2 * pad), max(W, H) / zoom_max), max(W, H))
    x0 = np.clip((lo[0] + hi[0] - side) / 2, 0, max(0.0, W - side))
    y0 = np.clip((lo[1] + hi[1] - side) / 2, 0, max(0.0, H - side))
    return int(x0), int(y0), int(min(W, x0 + side)), int(min(H, y0 + side))


def trim(pts):
    """Drop gross outliers (a match on a look-alike region) before framing the target crop."""
    d = np.linalg.norm(pts - np.median(pts, 0), axis=1)
    return pts[d <= 3 * np.median(d) + 1e-6]


@torch.no_grad()
def crop_features(bank, rel, box, flip):
    """Features of an image crop: FeatureBank -> its extractor; TunedBank -> frozen trunk tokens + (tuned) tail."""
    if hasattr(bank, "tail"):
        t, a, kw, hw, s = bank.tokens.get(rel, flip, img=sc.load_image(bank.tokens.root, rel).crop(box))
        bank.tail.eval()
        return bank.tail(t, a, kw, hw).float().cpu(), s
    return bank.ex(sc.load_image(bank.root, rel).crop(box), flip=flip)


class LabMatcher(sc.Matcher):
    """semcorr.Matcher + forward-backward term + zoom-in refinement, on frozen (FeatureBank) or tuned (TunedBank) banks."""

    def _low(self, fs, ss, ft, pts, patch):
        """(N,h,w) cosine similarity of the source points against the target map, plus the cycle term."""
        if torch.cuda.is_available():  # the cycle term is a (h*w)^2 matmul per pair
            fs, ft = fs.cuda(), ft.cuda()
        sim = torch.einsum("nc,chw->nhw", sc.sample_points(fs, pts, ss, patch), ft)
        if self.cfg.cycle > 0:  # sim(q,c) - max_p sim(p,c): 0 where the query is c's best source match, < 0 elsewhere
            back = (ft.flatten(1).T @ fs.flatten(1)).max(1).values.view(ft.shape[1:])
            sim = sim + self.cfg.cycle * (sim - back)
        return sim.cpu()

    def _full(self, pair, src_pts, flip):
        Ww, Hw = sc.work_size(pair.target_size, self.cfg.work_res)
        pts = sc.flip_points(src_pts, pair.source_size[0]) if flip else np.asarray(src_pts, np.float32)
        tot = 0
        for bank, w in self.banks:
            fs, ss = bank.get(pair.source_image, flip)
            ft, _ = bank.get(pair.target_image, False)
            low = self._low(fs, ss, ft, pts, bank.ex.patch)
            tot = tot + w * F.interpolate(low[None], size=(Hw, Ww), mode="bilinear", align_corners=False)[0]
        return tot / sum(w for _, w in self.banks)

    def similarity(self, pair, src_pts, flip):
        sim = self._full(pair, src_pts, flip)
        return self._zoom(pair, src_pts, flip, sim) if self.cfg.zoom > 0 else sim

    def _zoom(self, pair, src_pts, flip, sim):
        c, (W, H), (Ws, Hs) = self.cfg, pair.target_size, pair.source_size
        Hw, Ww = sim.shape[1:]
        q_src = np.array([q.src_xy for q in pair.queries], np.float32)
        flat = self._full(pair, q_src, flip).flatten(1).argmax(1)  # coarse matches of the queries frame the target crop
        coarse = (torch.stack([flat % Ww, flat // Ww], 1).float().numpy() + 0.5) * np.array([W / Ww, H / Hw], np.float32)
        bs = crop_box(q_src, (Ws, Hs), c.zoom_pad, c.zoom_max)
        bt = crop_box(trim(coarse), (W, H), c.zoom_pad, c.zoom_max)
        full = lambda b, wh: max(b[2] - b[0], b[3] - b[1]) >= 0.9 * max(wh)
        if full(bs, (Ws, Hs)) and full(bt, (W, H)):
            return sim  # both objects already fill their image: nothing to zoom
        loc = np.asarray(src_pts, np.float32) - np.array(bs[:2], np.float32)
        if flip:
            loc = sc.flip_points(loc, bs[2] - bs[0])
        xs = (np.arange(Ww) + 0.5) * W / Ww - bt[0]  # target work-grid cell centres in crop pixels
        ys = (np.arange(Hw) + 0.5) * H / Hw - bt[1]
        tot = 0
        for bank, w in self.banks:
            fs, ss = crop_features(bank, pair.source_image, bs, flip)
            ft, st = crop_features(bank, pair.target_image, bt, False)
            low = self._low(fs, ss, ft, loc, bank.ex.patch)
            h, w_ = ft.shape[1:]
            gx = torch.as_tensor(xs * st[0] / (w_ * bank.ex.patch) * 2 - 1, dtype=torch.float32)
            gy = torch.as_tensor(ys * st[1] / (h * bank.ex.patch) * 2 - 1, dtype=torch.float32)
            GY, GX = torch.meshgrid(gy, gx, indexing="ij")
            tot = tot + w * F.grid_sample(low[None], torch.stack([GX, GY], -1)[None], mode="bilinear",
                                          align_corners=False, padding_mode="border")[0]
        crop = tot / sum(w for _, w in self.banks)
        inside = (torch.as_tensor((ys >= 0) & (ys <= bt[3] - bt[1]))[:, None]
                  & torch.as_tensor((xs >= 0) & (xs <= bt[2] - bt[0]))[None])
        return torch.where(inside, (1 - c.zoom) * sim + c.zoom * crop, sim)


# --------------------------------------------------------------------------- tests / diagnostics
def correct(preds, pairs, alpha=0.10):
    return {q.id: int(np.linalg.norm(q.targets - np.asarray(preds[q.id]), axis=1).min() <= alpha * q.scale)
            for p in pairs for q in p.queries}


def size_bucket(p, q):
    r = q.scale / max(p.target_size)  # object size relative to the frame
    return "small" if r < 0.25 else ("medium" if r < 0.6 else "large")


def by_bucket(preds, pairs):
    ok = correct(preds, pairs)
    out = {}
    for b in ("small", "medium", "large"):
        v = [ok[q.id] for p in pairs for q in p.queries if size_bucket(p, q) == b]
        out[b] = (float(np.mean(v)) if v else float("nan"), len(v))
    return out


def boot_delta(pa, pb, pairs, n=2000, seed=0):
    """Mean and 95% CI of PCK(b) - PCK(a), resampling whole pairs (the queries of a pair are correlated)."""
    oa, ob = correct(pa, pairs), correct(pb, pairs)
    d = np.array([sum(ob[q.id] - oa[q.id] for q in p.queries) for p in pairs], float)
    m = np.array([len(p.queries) for p in pairs], float)
    idx = np.random.default_rng(seed).integers(0, len(pairs), (n, len(pairs)))
    s = d[idx].sum(1) / m[idx].sum(1)
    return d.sum() / m.sum(), np.percentile(s, 2.5), np.percentile(s, 97.5)


def taxonomy(preds, pairs):
    """Where do the failures land? near miss / another annotated part / elsewhere on the object / background."""
    rows = []
    for p in pairs:
        allt = np.concatenate([q.targets for q in p.queries])
        lo, hi = allt.min(0), allt.max(0)
        for q in p.queries:
            x, s = np.asarray(preds[q.id]), q.scale
            d = np.linalg.norm(q.targets - x, axis=1).min() / s
            if d <= 0.1:
                k = "correct"
            elif d <= 0.2:
                k = "near miss (0.1-0.2)"
            elif any(np.linalg.norm(o.targets - x, axis=1).min() <= 0.1 * s for o in p.queries if o is not q):
                k = "another part"
            elif ((x >= lo - 0.1 * s) & (x <= hi + 0.1 * s)).all():
                k = "on object, elsewhere"
            else:
                k = "background / other instance"
            rows.append((size_bucket(p, q), k))
    kinds = ["correct", "near miss (0.1-0.2)", "another part", "on object, elsewhere", "background / other instance"]
    lines = [f"{'':28s}" + "".join(f"{b:>9s}" for b in ("all", "small", "medium", "large"))]
    for k in kinds:
        cells = []
        for b in ("all", "small", "medium", "large"):
            sel = [r for r in rows if b == "all" or r[0] == b]
            cells.append(f"{sum(r[1] == k for r in sel) / max(1, len(sel)):9.3f}")
        lines.append(f"{k:28s}" + "".join(cells))
    return "\n".join(lines)


def topk_oracle(matcher, pairs, ks=(1, 2, 3, 5)):
    """PCK if the best of the top-k similarity peaks (5% NMS) were picked: the headroom of any re-ranking idea."""
    hits = {k: [] for k in ks}
    for p in pairs:
        flip = matcher.choose_flip(p)
        sim = matcher.similarity(p, np.array([q.src_xy for q in p.queries], np.float32), flip)
        _, Hw, Ww = sim.shape
        r = max(1, round(0.05 * max(Hw, Ww)))
        # separable max-pool (columns, then rows): same result as one (2r+1)^2 window, ~40x faster on CPU
        peak = sim == F.max_pool2d(F.max_pool2d(sim[None], (2 * r + 1, 1), 1, (r, 0)), (1, 2 * r + 1), 1, (0, r))[0]
        top = torch.where(peak, sim, torch.full_like(sim, -1e9)).flatten(1).topk(max(ks), 1).indices
        xy = torch.stack([top % Ww, top // Ww], -1).float().numpy() + 0.5
        xy = xy * np.array([p.target_size[0] / Ww, p.target_size[1] / Hw])
        for i, q in enumerate(p.queries):
            d = np.linalg.norm(xy[i][:, None] - q.targets[None], axis=2).min(1) / q.scale
            for k in ks:
                hits[k].append(d[:k].min() <= 0.1)
    return {k: float(np.mean(v)) for k, v in hits.items()}


# --------------------------------------------------------------------------- experiments
# (name, spec): spec = LabConfig overrides, plus "context": "queries". Rows with zoom re-run the backbone on crops.
ABLATION = [
    ("baseline (colleague zero-shot)", {}),
    ("flip none", {"flip": "none"}),
    ("context=queries", {"context": "queries"}),
    ("geo 1", {"geo_lambda": 1.0}),
    ("geo 1 + context=queries", {"geo_lambda": 1.0, "context": "queries"}),
    ("cycle 0.5", {"cycle": 0.5}),
    ("cycle 1", {"cycle": 1.0}),
    ("cycle 2", {"cycle": 2.0}),
    ("zoom 0.5", {"zoom": 0.5}),
    ("zoom 1.0", {"zoom": 1.0}),
    ("zoom 0.5 pad 1.0", {"zoom": 0.5, "zoom_pad": 1.0}),
    ("zoom 0.5 max 8", {"zoom": 0.5, "zoom_max": 8.0}),
    ("cycle 1 + zoom 0.5", {"cycle": 1.0, "zoom": 0.5}),
    ("cycle 1 + zoom 0.5 + geo 1 + queries", {"cycle": 1.0, "zoom": 0.5, "geo_lambda": 1.0, "context": "queries"}),
]


def load_banks(a, pairs):
    banks = []
    for ex, w in sc.build_extractors(a.backbone, a.long_side):
        bank = sc.FeatureBank(ex, a.data)
        path = a.cache and Path(a.cache) / f"{ex.name}_{a.long_side}.pt"
        if path:
            bank.load(path)
        n0 = len(bank.store)
        for p in sc.tqdm(pairs, desc=f"features[{ex.name}]"):
            bank.get(p.source_image, False), bank.get(p.source_image, True), bank.get(p.target_image, False)
        if path and len(bank.store) > n0:
            path.parent.mkdir(parents=True, exist_ok=True)
            bank.save(path)
        banks.append((bank, w))
    return banks


def ft_args(a):
    """The colleague's fine-tuning settings (v2.1 notebook CFG), plus the lab's token-masking ratio."""
    return SimpleNamespace(data=a.data, backbones=a.backbone, long_side=a.long_side, layers="-1", work_res=512,
                           radius=6, tau=0.02, flip="auto", adapter_weight=1.0, geo_lambda=0.0, geo_thresh=0.08,
                           cache_dir=None, limit_pairs=a.limit, adapter=False, finetune_blocks=a.blocks,
                           block_lr=2e-5, lr=3e-4, epochs=a.epochs, train_tau=0.05, sigma_rel=0.04, swap_aug=True,
                           raw_weight=0.0, seed=0, mask_ratio=a.mask_ratio)


def lab_matcher(banks, spec):
    """LabMatcher from a JSON spec of LabConfig overrides plus "context": "queries"; returns (matcher, pairs_fn)."""
    over = json.loads(spec)
    qctx = over.pop("context", "all") == "queries"
    return LabMatcher(banks, LabConfig(**over)), (lambda ps: [queries_as_context(p) for p in ps] if qctx else ps)


def ablate(a):
    """Zero-shot rows on the whole train set (no training, so no folds needed); writes preds.json and ablation.json."""
    pairs = sc.load_split(a.data, "train")[: a.limit or None]
    banks = load_banks(a, pairs)
    q = sc.queries_of(pairs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results, table = {}, []
    print(f"{'config':38s} {'PCK':>6s} {'delta [95% CI]':>24s} {'small':>6s} {'medium':>6s} {'large':>6s} {'sec':>5s}", flush=True)
    for name, spec in (json.loads(a.rows) if a.rows else ABLATION):
        t = time.time()
        m, prep = lab_matcher(banks, json.dumps(spec))
        results[name] = preds = m.predict(prep(pairs), progress=False)
        d, lo, hi = boot_delta(results[table[0]["name"]] if table else preds, preds, pairs)
        bb = by_bucket(preds, pairs)
        table.append({"name": name, "cfg": spec, "pck": sc.pck(preds, q), "delta": d, "ci": [lo, hi]})
        print(f"{name:38s} {table[-1]['pck']:6.4f} {d:+8.4f} [{lo:+.4f},{hi:+.4f}] "
              f"{bb['small'][0]:6.3f} {bb['medium'][0]:6.3f} {bb['large'][0]:6.3f} {time.time() - t:5.0f}", flush=True)
        (out / "preds.json").write_text(json.dumps(results))
        (out / "ablation.json").write_text(json.dumps(table, indent=1))
    print(f"bucket sizes (object long side / frame long side: small <0.25, large >=0.6): "
          f"{ {b: v[1] for b, v in by_bucket(preds, pairs).items()} }")
    best = max(table, key=lambda r: r["pck"])["name"]
    for name in dict.fromkeys([table[0]["name"], best]):
        print(f"\n--- error taxonomy: {name}\n{taxonomy(results[name], pairs)}")
    print("\n--- top-k peak oracle (baseline maps): PCK if the right peak among the top k were picked")
    print(topk_oracle(LabMatcher(banks, LabConfig()), pairs))


def ft_cv(a):
    """K-fold CV on the colleague's leakage-free folds: fine-tune the last blocks (optionally with BERT-style token
    masking) and score the same out-of-fold queries three ways: frozen zero-shot, fine-tuned, fine-tuned + lab ideas."""
    args = ft_args(a)
    banks, cfg, data = sc.setup(args, ["train"])
    pairs = data["train"]
    folds = sc.make_folds(pairs, a.folds, 0)
    oof = {"zero-shot": {}, "fine-tuned": {}, "fine-tuned + lab": {}}
    for k in range(a.folds):
        tr = [p for p in pairs if folds[p.pair_id] != k]
        va = [p for p in pairs if folds[p.pair_id] == k]
        make, _, _ = sc.fit(args, banks, cfg, tr, va)
        oof["zero-shot"].update(sc.Matcher(banks, cfg).predict(va, False))
        oof["fine-tuned"].update(make(cfg).predict(va, False))
        m, prep = lab_matcher(make(cfg).banks, a.cfg)
        oof["fine-tuned + lab"].update(m.predict(prep(va), False))
        print(f"fold {k}: " + "  ".join(f"{n} {sc.pck(o, sc.queries_of(va)):.4f}" for n, o in oof.items()), flush=True)
        del make, m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    q = sc.queries_of(pairs)
    print(f"=== {a.folds}-fold CV, {a.backbone} @ {a.long_side}, last {a.blocks} blocks, {a.epochs} epochs, "
          f"mask ratio {a.mask_ratio}, lab ideas {a.cfg}")
    for n, o in oof.items():
        d, lo, hi = boot_delta(oof["zero-shot"], o, pairs)
        print(f"{n:18s} PCK {sc.pck(o, q):.4f}   vs zero-shot {d:+.4f} [{lo:+.4f}, {hi:+.4f}]   {by_bucket(o, pairs)}")
    print(f"--- error taxonomy: fine-tuned + lab\n{taxonomy(oof['fine-tuned + lab'], pairs)}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    path = Path(a.out) / f"oof_{a.backbone.replace(':', '-').replace(',', '+')}_{a.long_side}_mask{a.mask_ratio}.json"
    path.write_text(json.dumps(oof))
    print("saved", path)


def load_preds(spec):
    """'file.json:key', e.g. 'lab_out/preds.json:cycle 1' or 'lab_out/oof_..._mask0.3.json:fine-tuned'.
    Without ':key' an ft-cv file gives its 'fine-tuned + lab' entry."""
    path, _, key = spec.partition(":")
    return json.loads(Path(path).read_text())[key or "fine-tuned + lab"]


def compare(a):
    """Paired bootstrap between two saved prediction sets (e.g. ft-cv with and without masking)."""
    pairs = sc.load_split(a.data, "train")
    pa, pb = (load_preds(f) for f in a.files)
    pairs = [p for p in pairs if all(q.id in pa and q.id in pb for q in p.queries)]
    q = sc.queries_of(pairs)
    d, lo, hi = boot_delta(pa, pb, pairs)
    print(f"A {sc.pck(pa, q):.4f}  B {sc.pck(pb, q):.4f}  B-A {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  (n={len(q)})")


def predict(a):
    """Test submission: zero-shot, or (--finetune) fine-tuned on the whole train set, both with the lab ideas."""
    if a.finetune:
        args = ft_args(a)
        banks, cfg, data = sc.setup(args, ["train", "test"])
        make, _, _ = sc.fit(args, banks, cfg, data["train"])
        banks, pairs = make(cfg).banks, data["test"]
    else:
        pairs = sc.load_split(a.data, "test")[: a.limit or None]
        banks = load_banks(a, pairs)
    m, prep = lab_matcher(banks, a.cfg)
    preds = m.predict(prep(pairs))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    sc.write_submission(preds, Path(a.data) / "test.csv", Path(a.out) / "submission.csv")
    print("wrote", Path(a.out) / "submission.csv")


def selftest():
    """Crop/zoom/flip coordinate check. The fake features encode the coordinates painted into the pixels, so matching
    an image with itself must send every point onto itself through every code path (full, zoom, flip, cycle)."""

    class PixelCoords:
        patch, name = 14, "pixel-coords"

        def __call__(self, img, flip=False):
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            W, H = img.size
            s = 448 / max(W, H)
            w, h = max(1, round(W * s / 14)), max(1, round(H * s / 14))
            a = torch.from_numpy(np.asarray(img.resize((w, h), Image.BOX), np.float32) / 255).permute(2, 0, 1)
            fr = torch.linspace(1, 12, 16)[:, None, None] * np.pi
            f = torch.cat([torch.sin(a[0] * fr), torch.cos(a[0] * fr), torch.sin(a[1] * fr), torch.cos(a[1] * fr)])
            return F.normalize(f, dim=0), (w * 14 / W, h * 14 / H)

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        for W, H in [(500, 375), (333, 499), (877, 640)]:
            xx, yy = np.meshgrid(np.arange(W) * 255 / W, np.arange(H) * 255 / H)
            Image.fromarray(np.stack([xx, yy, 0 * xx], -1).astype(np.uint8)).save(f"{tmp}/im.png")
            c = rng.uniform(0.3, 0.7, 2) * (W, H)  # a small object: keypoints within ~15% of the frame
            pts = (c + rng.uniform(-0.07, 0.07, (8, 2)) * max(W, H)).astype(np.float32)
            pr = sc.Pair("p", "im.png", "im.png", (W, H), (W, H), pts,
                         [sc.Query(f"q{i}", "p", i, tuple(pts[i])) for i in range(8)])
            bank = sc.FeatureBank(PixelCoords(), tmp)
            for over, flip in [({}, False), ({"zoom": 1.0}, False), ({"zoom": 1.0}, True), ({"zoom": 0.5, "cycle": 1.0}, True)]:
                m = LabMatcher([(bank, 1.0)], LabConfig(flip="none", tau=0.01, **over))
                m.choose_flip = lambda p, f=flip: f
                pred = m.predict_pair(pr)
                err = max(np.hypot(*(np.asarray(pred[f"q{i}"]) - pts[i])) for i in range(8)) / max(W, H)
                print(f"{W}x{H} {over} flip={flip}: max error {err:.4f} of long side")
                assert err < 0.02, err
    print("lab coordinate pipeline OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "ablate", "ft-cv", "compare", "predict"])
    ap.add_argument("--data", default="../aml-challenge-2026-27")
    ap.add_argument("--backbone", default="dinov2_l_reg", help="semcorr spec, e.g. dinov2_l_reg:1,dinov3_l:1")
    ap.add_argument("--long-side", type=int, default=840)
    ap.add_argument("--cache", default="", help="ablate/predict: folder to keep extracted features between runs")
    ap.add_argument("--out", default="lab_out")
    ap.add_argument("--limit", type=int, default=0, help="first N pairs only (quick debug)")
    ap.add_argument("--rows", default="", help="JSON list of [name, spec] rows replacing ABLATION")
    ap.add_argument("--cfg", default="{}", help="ft-cv/predict: LabConfig overrides as JSON, plus \"context\": \"queries\"")
    ap.add_argument("--finetune", action="store_true", help="predict: fine-tune on all of train first")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--mask-ratio", type=float, default=0.0)
    ap.add_argument("files", nargs="*", help="compare: two OOF prediction json files")
    a = ap.parse_args()
    {"selftest": lambda: selftest(), "ablate": lambda: ablate(a), "ft-cv": lambda: ft_cv(a),
     "compare": lambda: compare(a), "predict": lambda: predict(a)}[a.cmd]()
