"""Colleague's v2.1 pipeline (github.com/Antonietta2001/aml-challenge, aml_semcorr_v2_1.ipynb),
extracted verbatim from notebook cells 7-21 so it can be imported. Lab additions (defaults keep the original
behaviour): optional BERT-style token masking while fine-tuning (TailModel.mask_ratio, plumbed through
train_tail/fit) and TokenBank.get(img=...) for in-memory zoom crops.
`python semcorr.py` runs the notebook's coordinate sanity check."""
import copy, gc, itertools, json, math, os, random, time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

# ==================== notebook cell 7
@dataclass
class Query:
    id: str
    pair_id: str
    query_index: int
    src_xy: tuple[float, float]
    targets: np.ndarray | None = None  # (K,2) accepted target points (train only)
    scale: float | None = None  # longest side of target bbox (train only)


@dataclass
class Pair:
    pair_id: str
    source_image: str
    target_image: str
    source_size: tuple[int, int]  # (W,H)
    target_size: tuple[int, int]
    source_xy: np.ndarray  # (N,2) all source points of the pair (context)
    queries: list[Query] = field(default_factory=list)


def load_split(root: str | Path, split: str) -> list[Pair]:
    """Load `train` or `test` into a list of Pair objects with their queries attached."""
    root = Path(root)
    meta = json.loads((root / f"{split}_pairs.json").read_text())
    pairs = {}
    for p in meta["pairs"]:
        pairs[p["pair_id"]] = Pair(
            pair_id=p["pair_id"],
            source_image=p["source_image"],
            target_image=p["target_image"],
            source_size=tuple(p["source_size"]),
            target_size=tuple(p["target_size"]),
            source_xy=np.asarray(p["source_xy"], dtype=np.float32).reshape(-1, 2),
        )
    df = pd.read_csv(root / f"{split}.csv")
    has_gt = "target_points" in df.columns
    for r in df.itertuples(index=False):
        q = Query(
            id=r.id,
            pair_id=r.pair_id,
            query_index=int(r.query_index),
            src_xy=(float(r.source_x), float(r.source_y)),
        )
        if has_gt:
            q.targets = np.asarray(json.loads(r.target_points), dtype=np.float32).reshape(-1, 2)
            q.scale = float(r.scale)
        pairs[r.pair_id].queries.append(q)
    # keep only pairs that actually have queries
    return [p for p in pairs.values() if p.queries]


def load_image(root: str | Path, rel: str) -> Image.Image:
    return Image.open(Path(root) / rel).convert("RGB")


def image_components(pairs: list[Pair]) -> dict[str, int]:
    """Union-find over the pair graph: images linked by a pair get the same group id.

    Splitting by component guarantees that no image is shared between folds,
    mirroring the official train/test separation.
    """
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        a, b = find(p.source_image), find(p.target_image)
        if a != b:
            parent[a] = b
    roots = sorted({find(x) for x in list(parent)})
    rid = {r: i for i, r in enumerate(roots)}
    return {x: rid[find(x)] for x in parent}


def make_folds(pairs: list[Pair], n_folds: int = 5, seed: int = 0) -> dict[str, int]:
    """Assign each pair to a fold, grouping by connected image component.

    Greedy balancing on number of queries so folds have similar size.
    """
    comp = image_components(pairs)
    sizes: dict[int, int] = {}
    for p in pairs:
        c = comp[p.source_image]
        sizes[c] = sizes.get(c, 0) + len(p.queries)
    rng = np.random.default_rng(seed)
    order = sorted(sizes, key=lambda c: (-sizes[c], rng.random()))
    load = np.zeros(n_folds)
    comp_fold = {}
    for c in order:
        f = int(np.argmin(load))
        comp_fold[c] = f
        load[f] += sizes[c]
    return {p.pair_id: comp_fold[comp[p.source_image]] for p in pairs}


def reversed_pairs(pairs: list[Pair]) -> list[Pair]:
    """Source/target swap augmentation for training.

    For every query with exactly ONE accepted target, the target point becomes a source query whose
    accepted answer is the original source point. The object scale in the (new) target image is not
    given, so it is estimated from the ratio of the keypoint spreads in the two images (fallback:
    ratio of image long sides).
    """
    out = []
    for p in pairs:
        single = [q for q in p.queries if q.targets is not None and len(q.targets) == 1]
        if not single:
            continue
        src = np.array([q.src_xy for q in p.queries], np.float32)
        tgt = np.array([q.targets[0] for q in p.queries], np.float32)
        ext_s, ext_t = np.ptp(src, 0).max(), np.ptp(tgt, 0).max()
        if len(p.queries) >= 2 and ext_s > 1 and ext_t > 1:
            ratio = float(np.clip(ext_s / ext_t, 0.25, 4.0))
        else:
            ratio = max(p.source_size) / max(p.target_size)
        qs = [Query(id=q.id + "_rev", pair_id=p.pair_id + "_rev", query_index=i,
                    src_xy=(float(q.targets[0][0]), float(q.targets[0][1])),
                    targets=np.array([q.src_xy], np.float32), scale=q.scale * ratio)
              for i, q in enumerate(single)]
        out.append(Pair(pair_id=p.pair_id + "_rev", source_image=p.target_image, target_image=p.source_image,
                        source_size=p.target_size, target_size=p.source_size,
                        source_xy=np.array([q.src_xy for q in qs], np.float32), queries=qs))
    return out


# ==================== notebook cell 9
def normalized_errors(preds: dict[str, tuple[float, float]], queries) -> np.ndarray:
    errs = []
    for q in queries:
        p = np.asarray(preds[q.id], dtype=np.float64)
        d = np.linalg.norm(q.targets - p[None], axis=1).min()
        errs.append(d / q.scale)
    return np.asarray(errs)


def pck(preds: dict[str, tuple[float, float]], queries, alpha: float = 0.10) -> float:
    return float((normalized_errors(preds, queries) <= alpha).mean())


def pck_report(preds, queries, alpha: float = 0.10) -> str:
    """Overall PCK plus breakdown by object scale (small objects are the hard part)."""
    e = normalized_errors(preds, queries)
    s = np.array([q.scale for q in queries])
    lines = [f"PCK@{alpha:.2f} = {(e <= alpha).mean():.4f}  (n={len(e)})"]
    for lo, hi in [(0, 100), (100, 250), (250, 1e9)]:
        m = (s >= lo) & (s < hi)
        if m.any():
            lines.append(f"  scale [{lo:>4.0f},{hi:>6.0f}): {(e[m] <= alpha).mean():.4f}  (n={m.sum()})")
    for a in (0.05, 0.15, 0.20):
        lines.append(f"  PCK@{a:.2f} = {(e <= a).mean():.4f}")
    return "\n".join(lines)


# ==================== notebook cell 11
# Short aliases -> HuggingFace model ids. Any HF id or local directory also works.
ALIASES = {
    "dinov2_s": "facebook/dinov2-small",
    "dinov2_b": "facebook/dinov2-base",
    "dinov2_l": "facebook/dinov2-large",
    "dinov2_g": "facebook/dinov2-giant",
    "dinov2_b_reg": "facebook/dinov2-with-registers-base",
    "dinov2_l_reg": "facebook/dinov2-with-registers-large",
    "dinov2_g_reg": "facebook/dinov2-with-registers-giant",
    # DINOv3 checkpoints are gated on HF: accept the license and set HF_TOKEN
    # (Kaggle: Add-ons > Secrets), or point to a local copy / Kaggle Model.
    "dinov3_s": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_b": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_l": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_hplus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "dinov3_7b": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class DinoExtractor:
    """Wraps an HF DINOv2/DINOv3 model and returns patch-token feature maps.

    Args:
        name: alias from ALIASES, HF id, or local path. "random_dinov2" builds a tiny
            randomly initialised model (for offline smoke tests only).
        long_side: the image is resized so its longest side is ~long_side
            (rounded to a multiple of the patch size, aspect ratio preserved).
        layers: which hidden states to use. -1 = final normalised output; other ints
            index `hidden_states` (0 = embeddings). Multiple layers are L2-normalised
            and concatenated.
    """

    def __init__(self, name: str = "dinov2_b_reg", long_side: int = 644, layers=(-1,),
                 device: str | None = None, dtype=None):
        from transformers import AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            # DINOv3 has very large intermediate activations that overflow in fp16 (-> inf/NaN):
            # run it in fp32. DINOv2 is safe (and faster) in fp16.
            dtype = torch.float32 if "dinov3" in name.lower() else torch.float16
        self.dtype = dtype if self.device.startswith("cuda") else torch.float32
        if name == "random_dinov2":
            from transformers import Dinov2Config, Dinov2Model
            cfg = Dinov2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                               intermediate_size=128, patch_size=14, image_size=224)
            self.model = Dinov2Model(cfg)
        elif name == "random_dinov3":  # offline smoke test of the DINOv3 code path (RoPE, registers)
            from transformers import DINOv3ViTConfig, DINOv3ViTModel
            cfg = DINOv3ViTConfig(hidden_size=64, num_hidden_layers=3, num_attention_heads=2,
                                  intermediate_size=128, patch_size=16, num_register_tokens=4)
            self.model = DINOv3ViTModel(cfg)
        else:
            path = ALIASES.get(name, name)
            token = os.environ.get("HF_TOKEN")
            self.model = AutoModel.from_pretrained(path, token=token, dtype=self.dtype)
        self.model.eval().to(self.device, self.dtype)
        cfg = self.model.config
        self.patch = int(cfg.patch_size)
        self.n_special = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.long_side = long_side
        self.layers = tuple(layers)
        self.name = name

    def _resize(self, img: Image.Image):
        W, H = img.size
        s = self.long_side / max(W, H)
        w = max(self.patch, int(round(W * s / self.patch)) * self.patch)
        h = max(self.patch, int(round(H * s / self.patch)) * self.patch)
        return img.resize((w, h), Image.BICUBIC), (w / W, h / H)

    def preprocess(self, img: Image.Image, flip: bool = False):
        """-> (pixel tensor (1,3,H',W') on device, (sx, sy))."""
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        im, (sx, sy) = self._resize(img)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return ((x - MEAN) / STD)[None].to(self.device, self.dtype), (sx, sy)

    @torch.no_grad()
    def __call__(self, img: Image.Image, flip: bool = False):
        """Returns (feat (C,h,w) float32 L2-normalised on CPU, (sx, sy))."""
        x, (sx, sy) = self.preprocess(img, flip)
        need_hidden = any(l != -1 for l in self.layers)
        out = self.model(pixel_values=x, output_hidden_states=need_hidden)
        h, w = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        feats = []
        for l in self.layers:
            t = out.last_hidden_state if l == -1 else out.hidden_states[l]
            t = t[:, self.n_special:, :]
            assert t.shape[1] == h * w, (t.shape, h, w)
            t = t[0].float().T.reshape(-1, h, w)
            feats.append(F.normalize(t, dim=0))
        f = torch.cat(feats, 0)
        f = F.normalize(f, dim=0)
        return f.cpu(), (sx, sy)


def build_extractors(spec: str, long_side: int, layers=(-1,), device=None):
    """`spec` is a comma-separated list, optionally with weights: "dinov2_l_reg:1,dinov3_l:1"."""
    out = []
    for item in spec.split(","):
        name, _, w = item.partition(":")
        out.append((DinoExtractor(name.strip(), long_side=long_side, layers=layers, device=device),
                    float(w) if w else 1.0))
    return out


# ==================== notebook cell 13
# --------------------------------------------------------------------------- cache
class FeatureBank:
    """Caches (feature map, (sx,sy)) per (image, flip) for one extractor.

    Features are stored in fp16 on CPU. Optionally persisted to `cache_dir`
    so Kaggle sessions can reuse them (`save()` / `load()`).
    """

    def __init__(self, extractor, root):
        self.ex = extractor
        self.root = root
        self.store: dict[tuple[str, bool], tuple[torch.Tensor, tuple[float, float]]] = {}

    def get(self, rel: str, flip: bool = False):
        key = (rel, bool(flip))
        if key not in self.store:
            f, s = self.ex(load_image(self.root, rel), flip=flip)
            self.store[key] = (f.half(), s)
        f, s = self.store[key]
        return f.float(), s

    def precompute(self, images, flips=(False, True)):
        for rel in tqdm(sorted(set(images)), desc=f"features[{self.ex.name}]"):
            for fl in flips:
                self.get(rel, fl)

    def save(self, path):
        torch.save(self.store, path)

    def load(self, path):
        if Path(path).exists():
            self.store.update(torch.load(path))
        return self


# --------------------------------------------------------------------------- geometry
def flip_points(xy: np.ndarray, width: int) -> np.ndarray:
    out = np.array(xy, dtype=np.float32).copy()
    out[:, 0] = width - out[:, 0]
    return out


def sample_points(feat: torch.Tensor, xy: np.ndarray, sxsy, patch: int) -> torch.Tensor:
    """Bilinearly sample an L2-normalised (C,h,w) map at original-pixel points -> (N,C)."""
    C, h, w = feat.shape
    sx, sy = sxsy
    xy = torch.as_tensor(np.asarray(xy, dtype=np.float32), device=feat.device)
    gx = xy[:, 0] * sx / (w * patch) * 2 - 1
    gy = xy[:, 1] * sy / (h * patch) * 2 - 1
    grid = torch.stack([gx, gy], -1).view(1, 1, -1, 2).to(feat.dtype)
    v = F.grid_sample(feat[None], grid, mode="bilinear", align_corners=False, padding_mode="border")
    return F.normalize(v[0, :, 0].T, dim=1)


def work_size(size_wh, work_res: int):
    W, H = size_wh
    s = work_res / max(W, H)
    return max(1, int(round(W * s))), max(1, int(round(H * s)))


def sim_on_grid(src_vecs: torch.Tensor, tgt_feat: torch.Tensor, out_wh) -> torch.Tensor:
    """Cosine similarity of (N,C) vectors with a (C,h,w) map, upsampled to (N,Hw,Ww)."""
    sim = torch.einsum("nc,chw->nhw", src_vecs, tgt_feat)
    Ww, Hw = out_wh
    return F.interpolate(sim[None], size=(Hw, Ww), mode="bilinear", align_corners=False)[0]


def soft_argmax(sim: torch.Tensor, radius: int, tau: float) -> np.ndarray:
    """Argmax followed by a local softmax-weighted mean in a (2r+1)^2 window.

    sim: (N,H,W). Returns (N,2) [x,y] in work-grid pixel-center coordinates.
    """
    N, H, W = sim.shape
    flat = sim.view(N, -1).argmax(1)
    ys, xs = (flat // W).tolist(), (flat % W).tolist()
    out = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        y, x = ys[i], xs[i]
        if radius <= 0:
            out[i] = (x, y)
            continue
        y0, y1, x0, x1 = max(0, y - radius), min(H, y + radius + 1), max(0, x - radius), min(W, x + radius + 1)
        win = sim[i, y0:y1, x0:x1]
        wgt = torch.softmax((win - win.max()).flatten() / tau, 0).view_as(win)
        gy, gx = torch.meshgrid(torch.arange(y0, y1, device=sim.device),
                                torch.arange(x0, x1, device=sim.device), indexing="ij")
        out[i] = ((wgt * gx).sum().item(), (wgt * gy).sum().item())
    return out


# --------------------------------------------------------------------------- geometry prior
def fit_affine_ransac(P: np.ndarray, Q: np.ndarray, thresh: float, iters: int = 300, seed: int = 0):
    """Robust 2D affine Q ~ [P,1] @ A (A: 3x2). Returns (A, inlier mask) or (None, None)."""
    n = len(P)
    if n < 4:
        return None, None
    X = np.hstack([P, np.ones((n, 1), np.float32)])
    rng = np.random.default_rng(seed)
    best_in = None
    combos = [rng.choice(n, 3, replace=False) for _ in range(iters)]
    for idx in combos:
        A, *_ = np.linalg.lstsq(X[idx], Q[idx], rcond=None)
        inl = np.linalg.norm(X @ A - Q, axis=1) < thresh
        if best_in is None or inl.sum() > best_in.sum():
            best_in = inl
    if best_in.sum() < 4:
        return None, None
    A, *_ = np.linalg.lstsq(X[best_in], Q[best_in], rcond=None)
    return A, best_in


# --------------------------------------------------------------------------- predictor
@dataclass
class MatchConfig:
    work_res: int = 512        # resolution of the similarity grid (long side)
    radius: int = 6            # soft-argmax window radius (work-grid pixels); 0 = hard argmax
    tau: float = 0.02          # soft-argmax temperature (cosine-sim units)
    flip: str = "auto"         # "none" | "auto" (pick src orientation that best matches target)
    adapter_weight: float = 1.0  # mix: (1-w)*raw + w*adapted, when an adapter is given
    geo_lambda: float = 0.0    # strength of the geometric-consistency prior (0 = off)
    geo_thresh: float = 0.08   # RANSAC inlier threshold, fraction of the target work-grid long side


class Matcher:
    """Combines one or more FeatureBanks (weighted) and an optional learned adapter."""

    def __init__(self, banks_weights, cfg: MatchConfig | None = None, adapter=None, adapter_bank_idx=0):
        self.banks = banks_weights  # list of (FeatureBank-like, weight)
        self.cfg = cfg or MatchConfig()
        self.adapter = adapter
        self.adapter_idx = adapter_bank_idx

    def _adapted(self, feat):
        dev = next(self.adapter.parameters()).device
        with torch.no_grad():
            return self.adapter(feat.to(dev)[None])[0].float().cpu()

    def _orientation_score(self, bank, pair: Pair, flip: bool) -> float:
        """Mean best-match similarity of ALL source points of the pair (context points included)."""
        fs, ss = bank.get(pair.source_image, flip)
        ft, _ = bank.get(pair.target_image, False)
        pts = flip_points(pair.source_xy, pair.source_size[0]) if flip else pair.source_xy
        v = sample_points(fs, pts, ss, bank.ex.patch)
        sim = torch.einsum("nc,chw->nhw", v, ft).flatten(1)
        return sim.max(1).values.mean().item()

    def choose_flip(self, pair: Pair) -> bool:
        if self.cfg.flip != "auto":
            return False
        tot = [0.0, 0.0]
        for bank, w in self.banks:
            for k, fl in enumerate((False, True)):
                tot[k] += w * self._orientation_score(bank, pair, fl)
        return tot[1] > tot[0]

    def similarity(self, pair: Pair, src_pts: np.ndarray, flip: bool) -> torch.Tensor:
        """Weighted similarity maps (N, Hw, Ww) on the target work grid."""
        out_wh = work_size(pair.target_size, self.cfg.work_res)
        pts = flip_points(src_pts, pair.source_size[0]) if flip else np.asarray(src_pts, np.float32)
        total, wsum = None, 0.0
        for i, (bank, w) in enumerate(self.banks):
            fs, ss = bank.get(pair.source_image, flip)
            ft, _ = bank.get(pair.target_image, False)
            sim = sim_on_grid(sample_points(fs, pts, ss, bank.ex.patch), ft, out_wh)
            if self.adapter is not None and i == self.adapter_idx:
                aw = self.cfg.adapter_weight
                fa_s, fa_t = self._adapted(fs), self._adapted(ft)
                sim_a = sim_on_grid(sample_points(fa_s, pts, ss, bank.ex.patch), fa_t, out_wh)
                sim = (1 - aw) * sim + aw * sim_a
            total = w * sim if total is None else total + w * sim
            wsum += w
        return total / wsum

    def _geo_penalty(self, pair: Pair, sim_all: torch.Tensor, flip: bool, n_q: int):
        """Squared distance (normalised by the work-grid long side) of every target cell to the point
        predicted for each query by a RANSAC affine fit of ALL source keypoints -> their best matches.
        Returns (n_q, Hw, Ww) or None when no reliable fit exists."""
        N, Hw, Ww = sim_all.shape
        L = float(max(Hw, Ww))
        flat = sim_all.view(N, -1).argmax(1)
        Q = torch.stack([flat % Ww, flat // Ww], 1).float().cpu().numpy() / L
        P = np.asarray(pair.source_xy, np.float32)
        if flip:
            P = flip_points(P, pair.source_size[0])
        P = P / float(max(pair.source_size))
        A, inl = fit_affine_ransac(P, Q, self.cfg.geo_thresh)
        if A is None:
            return None
        qi = [q.query_index for q in pair.queries][:n_q]
        pred = np.hstack([P[qi], np.ones((len(qi), 1), np.float32)]) @ A  # (n_q,2) normalised work coords
        gy, gx = torch.meshgrid(torch.arange(Hw), torch.arange(Ww), indexing="ij")
        pr = torch.as_tensor(pred, dtype=torch.float32)
        d2 = (gx[None] / L - pr[:, 0, None, None]) ** 2 + (gy[None] / L - pr[:, 1, None, None]) ** 2
        return d2.to(sim_all.device)

    def predict_pair_multi(self, pair: Pair, configs) -> list[dict]:
        """Predictions for several (radius, tau, geo_lambda) settings re-using one similarity
        computation. work_res and flip are taken from self.cfg."""
        flip = self.choose_flip(pair)
        n_q = len(pair.queries)
        need_geo = any(c.geo_lambda > 0 for c in configs)
        if need_geo:
            # queries are a subset of source_xy (query_index); compute all rows once
            sim_all = self.similarity(pair, pair.source_xy, flip)
            sim = sim_all[[q.query_index for q in pair.queries]]
            d2 = self._geo_penalty(pair, sim_all, flip, n_q)
        else:
            src = np.array([q.src_xy for q in pair.queries], dtype=np.float32)
            sim, d2 = self.similarity(pair, src, flip), None
        W, H = pair.target_size
        Ww, Hw = work_size(pair.target_size, self.cfg.work_res)
        outs = []
        for c in configs:
            s = sim - c.geo_lambda * d2 if (d2 is not None and c.geo_lambda > 0) else sim
            uv = soft_argmax(s, c.radius, c.tau)
            x = np.clip((uv[:, 0] + 0.5) * W / Ww, 0, W - 1)
            y = np.clip((uv[:, 1] + 0.5) * H / Hw, 0, H - 1)
            outs.append({q.id: (float(x[i]), float(y[i])) for i, q in enumerate(pair.queries)})
        return outs

    def predict_pair(self, pair: Pair) -> dict[str, tuple[float, float]]:
        return self.predict_pair_multi(pair, [self.cfg])[0]

    def predict(self, pairs, progress=True) -> dict[str, tuple[float, float]]:
        preds = {}
        for p in tqdm(pairs, desc="matching", disable=not progress):
            preds.update(self.predict_pair(p))
        return preds

    def predict_multi(self, pairs, configs, progress=False) -> list[dict]:
        outs = [{} for _ in configs]
        for p in tqdm(pairs, desc="matching (sweep)", disable=not progress):
            for o, r in zip(outs, self.predict_pair_multi(p, configs)):
                o.update(r)
        return outs


# ==================== notebook cell 15
class Adapter(nn.Module):
    def __init__(self, c_in: int, hidden: int = 512, c_out: int = 384, dropout: float = 0.1):
        super().__init__()
        self.skip = nn.Conv2d(c_in, c_out, 1)
        self.body = nn.Sequential(
            nn.Conv2d(c_in, hidden, 1), nn.GELU(), nn.Dropout2d(dropout),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=8), nn.GELU(),
            nn.Conv2d(hidden, c_out, 1),
        )

    def forward(self, f):  # f: (B,C,h,w), already L2-normalised per position
        return F.normalize(self.skip(f) + self.body(f), dim=1)


def _target_distribution(targets_xy, scale, tgt_shape, sxsy, patch, sigma_rel, device):
    """Soft label over target patches: Gaussian around the NEAREST accepted target."""
    _, h, w = tgt_shape
    sx, sy = sxsy
    xs = (torch.arange(w, device=device) + 0.5) * patch / sx
    ys = (torch.arange(h, device=device) + 0.5) * patch / sy
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    t = torch.as_tensor(targets_xy, device=device, dtype=torch.float32)
    d2 = ((gx[None] - t[:, 0, None, None]) ** 2 + (gy[None] - t[:, 1, None, None]) ** 2).min(0).values
    sigma = max(sigma_rel * scale, 0.5 * patch / min(sx, sy))
    p = torch.exp(-d2 / (2 * sigma ** 2)).flatten()
    return p / p.sum().clamp_min(1e-8)


def _samples(pairs: list[Pair], use_flip_aug: bool):
    out = []
    for p in pairs:
        out.append((p, False))
        if use_flip_aug:
            out.append((p, True))
    return out


def pair_loss_from_feats(a_s, ss, a_t, st, patch, pair: Pair, flip: bool, tau: float, sigma_rel: float, device):
    """Dense contrastive loss for all queries of a pair, given (D,h,w) source/target feature maps.
    flip=True means BOTH images were mirrored (a semantics-preserving augmentation)."""
    src = np.array([q.src_xy for q in pair.queries], np.float32)
    if flip:
        src = flip_points(src, pair.source_size[0])
    q = sample_points(a_s.float(), src, ss, patch)  # (N,D)
    logits = torch.einsum("nd,dhw->nhw", q, a_t.float()).flatten(1) / tau
    logp = F.log_softmax(logits, 1)
    loss = 0.0
    for i, qq in enumerate(pair.queries):
        tg = flip_points(qq.targets, pair.target_size[0]) if flip else qq.targets
        p = _target_distribution(tg, qq.scale, a_t.shape, st, patch, sigma_rel, device)
        loss = loss + -(p * logp[i]).sum()
    return loss / len(pair.queries)


def pair_loss(adapter, bank: FeatureBank, pair: Pair, flip: bool, tau: float, sigma_rel: float, device):
    fs, ss = bank.get(pair.source_image, flip)
    ft, st = bank.get(pair.target_image, flip)
    if fs.shape == ft.shape:
        a = adapter(torch.stack([fs, ft]).to(device))
        a_s, a_t = a[0], a[1]
    else:
        a_s, a_t = adapter(fs.to(device)[None])[0], adapter(ft.to(device)[None])[0]
    return pair_loss_from_feats(a_s, ss, a_t, st, bank.ex.patch, pair, flip, tau, sigma_rel, device)


def cosine_schedule(opt, steps: int, warm_frac: float = 0.1):
    """Linear warm-up + cosine decay; robust for any number of steps."""
    warm = max(1, int(warm_frac * steps))
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))


def run_training(model: nn.Module, opt, loss_fn, train_pairs: list[Pair], val_fn=None, epochs: int = 5,
                 batch_pairs: int = 8, flip_aug: bool = True, seed: int = 0, keep_best: bool = False,
                 amp: bool = False, log=print):
    """Generic loop: loss_fn(pair, flip) -> scalar. Records the per-epoch val curve in model.history."""
    random.seed(seed)
    samples = _samples(train_pairs, flip_aug)
    steps = epochs * ((len(samples) + batch_pairs - 1) // batch_pairs)
    sched = cosine_schedule(opt, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_state, history = -1.0, None, []
    for ep in range(epochs):
        model.train()
        random.shuffle(samples)
        tot = 0.0
        for b in range(0, len(samples), batch_pairs):
            opt.zero_grad(set_to_none=True)
            batch = samples[b:b + batch_pairs]
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = sum(loss_fn(p, fl) for p, fl in batch) / len(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("training loss is NaN/inf -- check the backbone dtype / learning rate")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(batch)
        msg = f"epoch {ep + 1}/{epochs}  loss {tot / len(samples):.4f}"
        if val_fn is not None:
            model.eval()
            score = val_fn(model)
            msg += f"  val PCK {score:.4f}"
            history.append(score)
            if keep_best and score > best:
                best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model.history = history
    return model


def train_adapter(bank: FeatureBank, train_pairs: list[Pair], val_fn=None, epochs: int = 5,
                  lr: float = 3e-4, wd: float = 1e-2, tau: float = 0.05, sigma_rel: float = 0.04,
                  batch_pairs: int = 8, hidden: int = 512, c_out: int = 384, flip_aug: bool = True,
                  seed: int = 0, device=None, log=print, keep_best: bool = False):
    """Train an Adapter on cached (frozen) features. `val_fn(adapter) -> float` is optional; the
    per-epoch curve is stored in `adapter.history`. keep_best=True restores the best epoch
    (optimistic if the same val split is then reported)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    c_in = bank.get(train_pairs[0].source_image)[0].shape[0]
    adapter = Adapter(c_in, hidden, c_out).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=wd)
    loss_fn = lambda p, fl: pair_loss(adapter, bank, p, fl, tau, sigma_rel, device)
    return run_training(adapter, opt, loss_fn, train_pairs, val_fn, epochs, batch_pairs, flip_aug, seed,
                        keep_best, amp=False, log=log)


# ==================== notebook cell 17
def _blocks(model) -> nn.ModuleList:
    for path in ("encoder.layer", "model.layer", "layer"):  # DINOv2 / DINOv3 / fallback
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    raise AttributeError("cannot find transformer blocks")


def _final_norm(model) -> nn.Module:
    for name in ("layernorm", "norm"):
        if hasattr(model, name):
            return getattr(model, name)
    raise AttributeError("cannot find final norm")


class _Stop(Exception):
    pass


def _to(obj, device, dtype=None):
    if torch.is_tensor(obj):
        obj = obj.to(device)
        return obj.to(dtype) if dtype is not None and obj.is_floating_point() else obj
    if isinstance(obj, (tuple, list)):
        return type(obj)(_to(o, device, dtype) for o in obj)
    if isinstance(obj, dict):
        return {k: _to(v, device, dtype) for k, v in obj.items()}
    return obj


class TokenBank:
    """Caches the INPUT of block N-K (plus the extra args the block receives, e.g. RoPE for DINOv3)."""

    def __init__(self, extractor, root, k: int):
        self.ex, self.root, self.k = extractor, root, k
        self.start = len(_blocks(extractor.model)) - k
        self.store = {}

    @torch.no_grad()
    def get(self, rel: str, flip: bool = False, img=None):
        """`img` (lab addition): tokens of an in-memory image such as a zoom crop, returned uncached."""
        key = (rel, bool(flip))
        if img is not None or key not in self.store:
            x, sxsy = self.ex.preprocess(img if img is not None else load_image(self.root, rel), flip)
            cap = {}

            def hook(mod, args, kwargs):
                cap["h"] = args[0] if args else kwargs.pop("hidden_states")
                cap["args"] = _to(tuple(args[1:]), "cpu")
                cap["kwargs"] = _to({k: v for k, v in kwargs.items() if k != "hidden_states"}, "cpu")
                raise _Stop

            handle = _blocks(self.ex.model)[self.start].register_forward_pre_hook(hook, with_kwargs=True)
            try:
                self.ex.model(pixel_values=x)
            except _Stop:
                pass
            finally:
                handle.remove()
            hw = (x.shape[-2] // self.ex.patch, x.shape[-1] // self.ex.patch)
            h = cap["h"][0].float()
            if not torch.isfinite(h).all():
                raise FloatingPointError(f"non-finite tokens from {self.ex.name} on {rel} (dtype {self.ex.dtype})")
            # fp16 storage halves RAM, but its range (6.5e4) is too small for some backbones -> bf16 then
            store_dtype = torch.float16 if h.abs().max() < 3e4 else torch.bfloat16
            entry = (h.to(store_dtype).cpu(), cap["args"], cap["kwargs"], hw, sxsy)
            if img is not None:
                return entry
            self.store[key] = entry
        return self.store[key]

    def precompute(self, images, flips=(False, True)):
        for rel in tqdm(sorted(set(images)), desc=f"tokens[{self.ex.name}, last {self.k} blocks]"):
            for fl in flips:
                self.get(rel, fl)


class TailModel(nn.Module):
    """Last K blocks + final norm (initialised from the pretrained model) + optional adapter head."""

    def __init__(self, extractor, k: int, head: bool = True, hidden: int = 512, c_out: int = 384,
                 mask_ratio: float = 0.0):
        super().__init__()
        m = extractor.model
        self.blocks = nn.ModuleList(copy.deepcopy(b).float() for b in _blocks(m)[-k:])
        self.norm = copy.deepcopy(_final_norm(m)).float()
        self.n_special = extractor.n_special
        c = m.config.hidden_size
        self.head = Adapter(c, hidden, c_out) if head else None
        self.mask_ratio, self._pos = mask_ratio, {}
        emb, p = getattr(m, "embeddings", None), extractor.patch
        self._pos_fn = None  # DINOv2: absolute position embeddings; DINOv3 gets positions from RoPE (kwargs)
        if hasattr(emb, "interpolate_pos_encoding"):
            w = next(emb.parameters())
            self._pos_fn = lambda hw: emb.interpolate_pos_encoding(
                torch.zeros(1, 1 + hw[0] * hw[1], c, device=w.device, dtype=w.dtype), hw[0] * p, hw[1] * p)[0, 1:]

    def _mask(self, h, hw):
        """Lab addition, BERT/MAE principle: replace a random `mask_ratio` of the patch tokens entering the tuned
        blocks with the image's mean token (+ its position embedding), so their features must come from context."""
        x = h[0, self.n_special:]
        idx = torch.randperm(len(x), device=x.device)[: int(self.mask_ratio * len(x))]
        fill = x.mean(0).detach().expand(len(idx), -1)
        if self._pos_fn is not None:
            if hw not in self._pos:
                with torch.no_grad():
                    self._pos[hw] = self._pos_fn(hw).float()
            fill = fill + self._pos[hw].to(x.device)[idx]
        return torch.cat([h[:, :self.n_special], x.index_put((idx,), fill)[None]], 1)

    def forward(self, tokens, args, kwargs, hw):
        dev = next(self.parameters()).device
        h = tokens.to(dev, torch.float32)[None]
        if self.training and self.mask_ratio > 0:
            h = self._mask(h, tuple(hw))
        args, kwargs = _to(args, dev, torch.float32), _to(kwargs, dev, torch.float32)
        for b in self.blocks:
            out = b(h, *args, **kwargs)
            h = out[0] if isinstance(out, (tuple, list)) else out
        h = self.norm(h)[0, self.n_special:]
        f = F.normalize(h.float().T.reshape(-1, *hw), dim=0)
        if self.head is not None:
            f = self.head(f[None])[0]
        return f


class TunedBank:
    """FeatureBank-compatible view of a TailModel (features recomputed lazily, cached under no_grad)."""

    def __init__(self, tokens: TokenBank, tail: TailModel, cache: bool = True):
        self.tokens, self.tail, self.cache = tokens, tail, cache
        self.ex = SimpleNamespace(patch=tokens.ex.patch, name=f"{tokens.ex.name}+ft{tokens.k}")
        self.store = {}

    @torch.no_grad()
    def get(self, rel: str, flip: bool = False):
        key = (rel, bool(flip))
        if key in self.store:
            f, s = self.store[key]
            return f.float(), s
        t, a, kw, hw, sxsy = self.tokens.get(rel, flip)
        self.tail.eval()
        f = self.tail(t, a, kw, hw).float().cpu()
        if self.cache:
            self.store[key] = (f.half(), sxsy)
        return f, sxsy


def frozen_bank(tokens: TokenBank, device=None) -> TunedBank:
    """Pretrained (untrained-tail) features recomputed from the cached tokens, without storing them:
    avoids keeping a second full-resolution feature cache in RAM."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tb = TunedBank(tokens, TailModel(tokens.ex, tokens.k, head=False).to(device).eval(), cache=False)
    tb.ex.name = f"{tokens.ex.name} (frozen)"
    return tb


def train_tail(tokens: TokenBank, train_pairs: list[Pair], val_fn=None, epochs: int = 5, head_lr: float = 3e-4,
               block_lr: float = 2e-5, wd: float = 0.05, tau: float = 0.05, sigma_rel: float = 0.04,
               batch_pairs: int = 8, flip_aug: bool = True, swap_aug: bool = False, seed: int = 0,
               device=None, log=print, keep_best: bool = False, mask_ratio: float = 0.0):
    """Fine-tune the last K blocks (small lr) jointly with the adapter head (larger lr).
    `val_fn(TunedBank) -> float` is optional."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    tail = TailModel(tokens.ex, tokens.k, mask_ratio=mask_ratio).to(device)
    head_params = list(tail.head.parameters()) if tail.head is not None else []
    hp = {id(p) for p in head_params}
    body_params = [p for p in tail.parameters() if id(p) not in hp]
    opt = torch.optim.AdamW([{"params": body_params, "lr": block_lr}, {"params": head_params, "lr": head_lr}],
                            weight_decay=wd)
    patch = tokens.ex.patch

    def loss_fn(pair, flip):
        ts, as_, ks, hws, ss = tokens.get(pair.source_image, flip)
        tt, at, kt, hwt, st = tokens.get(pair.target_image, flip)
        fs, ft = tail(ts, as_, ks, hws), tail(tt, at, kt, hwt)
        return pair_loss_from_feats(fs, ss, ft, st, patch, pair, flip, tau, sigma_rel, device)

    wrapped_val = (lambda m: val_fn(TunedBank(tokens, m))) if val_fn is not None else None
    # mixed precision only for backbones that are fp16-safe (not DINOv3, see backbones.py)
    amp = str(device).startswith("cuda") and tokens.ex.dtype == torch.float16
    if swap_aug:
        train_pairs = list(train_pairs) + reversed_pairs(train_pairs)
    run_training(tail, opt, loss_fn, train_pairs, wrapped_val, epochs, batch_pairs, flip_aug, seed,
                 keep_best, amp=amp, log=log)
    return tail


# ==================== notebook cell 21
def _cfg(args):
    return MatchConfig(work_res=args.work_res, radius=args.radius, tau=args.tau, flip=args.flip,
                       adapter_weight=args.adapter_weight, geo_lambda=getattr(args, "geo_lambda", 0.0),
                       geo_thresh=getattr(args, "geo_thresh", 0.08))


def setup(args, splits):
    """Load data and build one bank per backbone.

    Adapter / zero-shot mode: FeatureBank (cached final features).
    Fine-tuning mode: TokenBank (cached tokens entering the last K blocks) + a frozen view computed
    from those tokens on the fly, so only ONE cache per backbone lives in RAM.
    """
    layers = tuple(int(x) for x in args.layers.split(","))
    exts = build_extractors(args.backbones, args.long_side, layers)
    ft = getattr(args, "finetune_blocks", 0) > 0
    data = {}
    for s in splits:
        pairs = load_split(args.data, s)
        if args.limit_pairs:
            pairs = pairs[: args.limit_pairs]
        data[s] = pairs
    imgs = [im for s in splits for p in data[s] for im in (p.source_image, p.target_image)]
    banks = []
    for ex, w in exts:
        if ft:
            tokens = TokenBank(ex, args.data, args.finetune_blocks)
            tokens.precompute(imgs)
            bank = frozen_bank(tokens)
            bank.token_bank = tokens
        else:
            bank = FeatureBank(ex, args.data)
            if args.cache_dir:
                Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
                tag = f"{ex.name.replace('/', '_')}_{args.long_side}_{args.layers.replace(',', '_')}"
                bank.cache_path = Path(args.cache_dir) / f"{tag}.pt"
                bank.load(bank.cache_path)
            bank.precompute(imgs)
            if getattr(bank, "cache_path", None):
                bank.save(bank.cache_path)
        f0, _ = bank.get(imgs[0], False)  # fail fast instead of training on NaNs for hours
        if not torch.isfinite(f0).all():
            raise FloatingPointError(f"backbone {ex.name} produces non-finite features")
        banks.append((bank, w))
    return banks, _cfg(args), data


def queries_of(pairs):
    return [q for p in pairs for q in p.queries]


def cmd_eval(args):
    banks, cfg, data = setup(args, ["train"])
    pairs = data["train"]
    q = queries_of(pairs)
    t = time.time()
    preds = Matcher(banks, cfg).predict(pairs)
    print(pck_report(preds, q))
    print(f"({time.time() - t:.1f}s)")
    # ablation of matching knobs without re-extracting features
    for flip in ("none", "auto"):
        for radius in (0, cfg.radius):
            c = MatchConfig(**{**cfg.__dict__, "flip": flip, "radius": radius})
            print(f"flip={flip:5s} radius={radius}: PCK={pck(Matcher(banks, c).predict(pairs, False), q):.4f}")
    if len(banks) > 1:
        for bank, _ in banks:
            print(f"single backbone {bank.ex.name}: PCK={pck(Matcher([(bank, 1.0)], cfg).predict(pairs, False), q):.4f}")
    return pck(preds, q)


def fit(args, banks, cfg, tr, va=None):
    """Train the learned model on `tr` (optionally logging val PCK on `va` every epoch).

    Fine-tuning mode trains one tail (last K blocks + head) PER BACKBONE and ensembles the tuned
    features with the backbone weights. Returns (make_matcher(cfg) -> Matcher, models, histories).
    """
    vq = queries_of(va) if va else None
    common = dict(epochs=args.epochs, tau=args.train_tau, sigma_rel=args.sigma_rel, seed=args.seed)
    if args.finetune_blocks > 0:
        tuned = []
        models, histories = [], []
        for raw, w in banks:
            val_fn = (lambda tb: pck(Matcher([(tb, 1.0)], cfg).predict(va, False), vq)) if va else None
            print(f"--- fine-tuning {raw.token_bank.ex.name} (last {args.finetune_blocks} blocks)")
            tail = train_tail(raw.token_bank, tr, val_fn, head_lr=args.lr, block_lr=args.block_lr,
                              swap_aug=args.swap_aug, mask_ratio=getattr(args, "mask_ratio", 0.0), **common)
            tuned.append((TunedBank(raw.token_bank, tail), w))
            models.append(tail)
            histories.append(tail.history)
        mix = args.raw_weight
        extra = [(raw, mix * w) for raw, w in banks] if mix > 0 else []
        return (lambda c: Matcher(tuned + extra, c)), models, histories

    val_fn = (lambda ad: pck(Matcher(banks, cfg, adapter=ad).predict(va, False), vq)) if va else None
    ad = train_adapter(banks[0][0], tr, val_fn=val_fn, lr=args.lr, **common)
    return (lambda c: Matcher(banks, c, adapter=ad)), [ad], [ad.history]


def sweep_configs(cfg):
    grid = itertools.product((4, 6, 9), (0.01, 0.02, 0.04), (0.0, 0.5, 1.0, 2.0, 4.0))
    return [MatchConfig(**{**cfg.__dict__, "radius": r, "tau": t, "geo_lambda": g}) for r, t, g in grid]


def cmd_cv(args):
    """Honest K-fold CV: every fold trains for exactly `epochs` epochs (no best-epoch picking on the
    val fold). Prints the mean per-epoch val curve. With --sweep, inference-time knobs (soft-argmax
    radius/tau, geometric prior) are tuned on the pooled out-of-fold predictions; the best setting
    is returned so the final model can use it."""
    banks, cfg, data = setup(args, ["train"])
    pairs = data["train"]
    folds = make_folds(pairs, args.folds, args.seed)
    learned = args.adapter or args.finetune_blocks > 0
    configs = sweep_configs(cfg) if args.sweep else []
    all_zero, all_model = {}, {}
    all_sweep = [{} for _ in configs]
    curves = []
    for k in range(args.folds):
        tr = [p for p in pairs if folds[p.pair_id] != k]
        va = [p for p in pairs if folds[p.pair_id] == k]
        vq = queries_of(va)
        zero = Matcher(banks, cfg).predict(va, False)
        all_zero.update(zero)
        if learned:
            make, models, hist = fit(args, banks, cfg, tr, va)
            curves.append(np.mean(np.array(hist), 0))
            matcher = make(cfg)
        else:
            matcher = Matcher(banks, cfg)
        pred = matcher.predict(va, False)
        all_model.update(pred)
        if configs:
            for o, r in zip(all_sweep, matcher.predict_multi(va, configs)):
                o.update(r)
        print(f"fold {k}: n={len(vq)} zero-shot {pck(zero, vq):.4f} -> model {pck(pred, vq):.4f}")
        if learned:
            del make, models, matcher
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    q = queries_of(pairs)
    print("=== CV zero-shot ===\n" + pck_report(all_zero, q))
    name = f"fine-tune K={args.finetune_blocks}" if args.finetune_blocks > 0 else ("adapter" if args.adapter else "zero-shot")
    print(f"=== CV {name}, {args.epochs} epochs, radius={cfg.radius} tau={cfg.tau} geo={cfg.geo_lambda} ===\n"
          + pck_report(all_model, q))
    if curves:
        mc = np.mean(np.array(curves), 0)
        print("mean val PCK per epoch (avg over backbones): " + " ".join(f"{e + 1}:{v:.4f}" for e, v in enumerate(mc)))
    best = None
    if configs:
        scores = [pck(o, q) for o in all_sweep]
        # ties -> prefer no geometric prior and the default soft-argmax settings
        order = sorted(range(len(configs)), key=lambda i: (-scores[i], configs[i].geo_lambda,
                                                            abs(configs[i].radius - 6), abs(configs[i].tau - 0.02)))
        print("inference sweep on out-of-fold predictions (top 8 of %d):" % len(configs))
        for i in order[:8]:
            c = configs[i]
            print(f"  radius={c.radius:2d} tau={c.tau:.2f} geo_lambda={c.geo_lambda:.1f}: {scores[i]:.4f}")
        for g in (0.0, 0.5, 1.0, 2.0, 4.0):
            sel = [s for s, c in zip(scores, configs) if c.geo_lambda == g]
            print(f"  best with geo_lambda={g}: {max(sel):.4f}")
        b = configs[int(order[0])]
        best = {"radius": b.radius, "tau": b.tau, "geo_lambda": b.geo_lambda, "cv_pck": float(scores[order[0]])}
        print("best inference setting:", best)
    return best


def write_submission(preds, test_csv, path):
    test = pd.read_csv(test_csv)
    missing = set(test.id) - set(preds)
    assert not missing, f"{len(missing)} test ids without prediction"
    xy = np.array([preds[i] for i in test.id], dtype=np.float64)
    assert np.isfinite(xy).all(), "non-finite predictions -- refusing to write the submission"
    rows = [json.dumps([round(float(preds[i][0]), 3), round(float(preds[i][1]), 3)], separators=(",", ":"))
            for i in test.id]
    pd.DataFrame({"id": test.id, "prediction": rows}).to_csv(path, index=False)


def cmd_predict(args):
    learned = args.adapter or args.finetune_blocks > 0
    banks, cfg, data = setup(args, ["train", "test"] if learned else ["test"])
    matcher = Matcher(banks, cfg)
    if learned:
        make, models, _ = fit(args, banks, cfg, data["train"])
        matcher = make(cfg)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        torch.save([m.state_dict() for m in models], Path(args.out) / "model.pt")
    preds = matcher.predict(data["test"])
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out = Path(args.out) / "submission.csv"
    write_submission(preds, Path(args.data) / "test.csv", out)
    print(f"wrote {out} ({len(preds)} rows)")


if __name__ == "__main__":  # notebook cell 19: coordinate-pipeline sanity check
    class _FakeExtractor:
        patch, name = 14, "fake"
        def __init__(self, size, long_side=448): self.size, self.long_side = size, long_side
        def __call__(self, img, flip=False):
            W, H = self.size; s = self.long_side / max(W, H)
            w, h = int(round(W * s / 14)) * 14, int(round(H * s / 14)) * 14
            sx, sy = w / W, h / H
            xs = (torch.arange(w // 14) + 0.5) * 14 / sx; ys = (torch.arange(h // 14) + 0.5) * 14 / sy
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            fr = torch.linspace(0.5, 6, 16)[:, None, None] * np.pi / max(W, H)
            f = torch.cat([torch.sin(gx * fr), torch.cos(gx * fr), torch.sin(gy * fr), torch.cos(gy * fr)])
            return F.normalize(f, dim=0), (sx, sy)

    class _FakeBank:
        def __init__(self, ex): self.ex = ex
        def get(self, rel, flip=False): return self.ex(None, flip)

    _rng = np.random.default_rng(0)
    for W, H in [(500, 375), (333, 499), (877, 640), (160, 120)]:
        pts = np.stack([_rng.uniform(.05 * W, .95 * W, 20), _rng.uniform(.05 * H, .95 * H, 20)], 1).astype(np.float32)
        pr = Pair("p", "a", "b", (W, H), (W, H), pts, [Query(f"q{i}", "p", i, tuple(pts[i])) for i in range(20)])
        pred = Matcher([(_FakeBank(_FakeExtractor((W, H))), 1.0)], MatchConfig(flip="none", tau=0.01)).predict_pair(pr)
        err = np.array([np.hypot(pred[f"q{i}"][0] - pts[i, 0], pred[f"q{i}"][1] - pts[i, 1]) for i in range(20)])
        print(f"{W}x{H}: max error {err.max():.2f}px = {err.max() / max(W, H):.4f} of long side")
        assert err.max() / max(W, H) < 0.02
    print("coordinate pipeline OK")
