"""Build a single self-contained Kaggle notebook from the modules in src/semcorr.

The package stays the source of truth (team development on GitHub); this script
inlines it into notebooks/aml_semcorr_standalone.ipynb, which needs no repo clone.

    python tools/build_notebook.py
"""
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "semcorr"
OUT = ROOT / "notebooks" / "aml_semcorr_v2_1.ipynb"  # v1 and v2 notebooks are kept frozen


def module(name: str) -> str:
    code = (SRC / f"{name}.py").read_text()
    code = re.sub(r'^"""[\s\S]*?"""\n', "", code, count=1)  # module docstring -> markdown cell
    code = re.sub(r"^from __future__ import annotations\n", "", code, flags=re.M)
    code = re.sub(r"^from \.\w+ import \([\s\S]*?\)\n", "", code, flags=re.M)
    code = re.sub(r"^from \.\w+ import .*\n", "", code, flags=re.M)
    return code.strip() + "\n"


def cli_functions() -> str:
    """Reuse the command implementations from cli.py (everything but argparse/main)."""
    code = (SRC / "cli.py").read_text()
    start = code.index("def _cfg(")
    end = code.index("def main(")
    return code[start:end].strip() + "\n"


md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells += [md("""# AML Challenge 2026/27 — Semantic Correspondence (v2.1)

**Task.** Given a source image, a target image and a point on the source object, predict the location of the
same semantic part in the target image. **Metric:** PCK@0.10, where a prediction is correct if it lies within 10%
of the target object's bounding-box long side from *any* accepted target point.

**Method overview**

1. **Dense features**: DINOv2 (+ DINOv3 when an HF token is available) patch tokens at high resolution
   (long side 840 px), L2-normalised. The backbones are ensembled by weighting their similarity maps.
2. **Matching**: the source feature is sampled bilinearly at the query point, its cosine similarity is computed
   against all target patches and upsampled to a fine grid, and an argmax with local soft-argmax gives sub-patch accuracy.
3. **Flip alignment**: for every pair we test the source image as-is and mirrored, and keep the orientation in which
   *all* source keypoints of the pair match the target best. This reduces left/right confusions under viewpoint changes.
4. **Learned model** (trained only on the provided training set): fine-tuning of the last 2 ViT blocks of *each*
   backbone plus a small conv head, trained with a dense contrastive loss. The soft label is a Gaussian around the
   nearest accepted target point. Augmentations: mirroring both images, and swapping source and target.
5. **Geometric consistency** (optional, tuned by CV): a robust affine fit (RANSAC) maps all source keypoints of the pair
   to their best matches. The resulting position becomes a soft prior that penalises far-away look-alike peaks.
6. **Validation**: 5-fold CV with folds built from connected components of the image-pair graph, so no image is
   shared between train and validation folds, mirroring the official train/test split.

**What changed from v1** (CV 0.5435 with DINOv2-L, 644 px, fine-tuning of 2 blocks for 5 epochs):

- Higher resolution: 840 px instead of 644. A zero-shot comparison of both resolutions runs first.
- A DINOv3-L ensemble, enabled automatically when the `HF_TOKEN` secret exists; each backbone is fine-tuned separately.
- 3 epochs instead of 5, since the v1 curve peaked at epoch 2–3.
- Source/target swap augmentation for single-target queries.
- Inference knobs (soft-argmax radius/temperature, geometric-prior strength) tuned on out-of-fold CV predictions
  and then used for the test set.

**v2.1 (bug fix after the first v2 run).** DINOv3 overflowed in fp16: its features came out NaN and broke the
ensemble, the CV and the submission. DINOv3 now runs in fp32, its cached tokens are stored in bf16 when they exceed the
fp16 range, and mixed-precision training is disabled for it. Several fail-fast checks were added: non-finite features,
non-finite training loss, and non-finite predictions all stop the run. The v2 run had already shown that 840 px beats
644 px zero-shot with DINOv2 (0.506 vs 0.447), so that check is now off by default (`run_res_sweep`).

Only the competition data and publicly available pretrained models are used. This notebook is self-contained:
it does not clone any repository.

*Kaggle settings: Accelerator **GPU**, Internet **On** (needed to download pretrained weights); add the
competition data with **Add Input**.*"""),
          md("## 0. Configuration"),
          code("""import os, glob, json, time, random, gc
from types import SimpleNamespace

def find_data_dir():
    hits = glob.glob("/kaggle/input/**/train_pairs.json", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    return "."  # local run: execute the notebook from the extracted participant_data folder

ON_KAGGLE = os.path.exists("/kaggle/working")
WORK = "/kaggle/working" if ON_KAGGLE else "outputs"

# Hugging Face token for the gated DINOv3 weights (Kaggle: Add-ons > Secrets > HF_TOKEN).
# Accept the license on https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m first.
HAS_TOKEN = False
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    HAS_TOKEN = True
    print("HF_TOKEN loaded -> DINOv2 + DINOv3 ensemble")
except Exception:
    print("No HF_TOKEN secret -> DINOv2 only")

CFG = SimpleNamespace(
    data=find_data_dir(),
    backbones="dinov2_l_reg:1,dinov3_l:1" if HAS_TOKEN else "dinov2_l_reg",
    long_side=840,                # input resolution (long side, px); v1 used 644
    layers="-1",                  # hidden-state indices, -1 = final normalised output
    work_res=512,                 # similarity grid resolution
    radius=6, tau=0.02,           # soft-argmax window radius / temperature (re-tuned by the CV sweep)
    flip="auto",                  # "auto" | "none"
    geo_lambda=0.0, geo_thresh=0.08,  # geometric-consistency prior (strength re-tuned by the CV sweep)
    cache_dir=None,
    limit_pairs=0,                # >0 = quick debug on the first N pairs
    out=WORK,
    # learned model: fine-tune the last K blocks of every backbone + adapter head
    adapter=False, adapter_weight=1.0,
    finetune_blocks=2, block_lr=2e-5, lr=3e-4, epochs=3, train_tau=0.05, sigma_rel=0.04,
    swap_aug=True, raw_weight=0.0,
    sweep=True,                   # tune radius / tau / geo_lambda on out-of-fold CV predictions
    folds=5, seed=0,
    # what to run
    run_res_sweep=False, run_eval=True, run_cv=True, run_predict=True,
)
print("data dir:", CFG.data, "| backbones:", CFG.backbones)"""),
          code("""# DINOv3 requires transformers >= 4.56
import transformers
from packaging.version import Version
if Version(transformers.__version__) < Version("4.56"):
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.56"], check=True)
    print("transformers upgraded: RESTART the kernel and run again")
else:
    print("transformers", transformers.__version__)"""),
          md("## 1. Imports")]

cells.append(code("""import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field
from tqdm.auto import tqdm
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())"""))

sections = [
    ("data", "## 2. Data loading and leakage-free folds\nEach `Pair` holds its images, sizes, all source points "
             "(`source_xy`, including context points without a query) and its queries. Folds group pairs by "
             "connected components of the image graph."),
    ("metric", "## 3. Metric: PCK@0.10\nDistance to the *nearest* accepted target, divided by the target bbox long side "
               "(`scale`). It also reports a breakdown by object size."),
    ("backbones", "## 4. Backbones: DINOv2 / DINOv3 dense features\nImages are resized so the long side is about "
                  "`long_side` px (a multiple of the patch size, aspect ratio preserved). Patch tokens, without the CLS "
                  "and register tokens, form a `(C,h,w)` map. `(sx, sy)` map original pixels to resized pixels."),
    ("matching", "## 5. Matching: similarity, soft-argmax and flip alignment"),
    ("adapter", "## 6. Learned adapter (trained on the provided training data only)\nThe backbone stays frozen and "
                "its features are cached. A light conv head is trained so that the adapted source feature at the "
                "query point scores highest near any accepted target point. The augmentation mirrors *both* images, "
                "which preserves semantics."),
    ("finetune", "## 7. Partial fine-tuning of the last K blocks (optional, `CFG.finetune_blocks > 0`)\n"
                 "The frozen part of the ViT runs once per image and its tokens are cached. Only the last K blocks "
                 "(small learning rate), the final norm and the adapter head are trained. Full-resolution "
                 "fine-tuning of a ViT-L therefore stays cheap on one Kaggle GPU."),
]
for name, title in sections:
    cells.append(md(title))
    cells.append(code(module(name)))

cells += [md("## 8. Sanity check of the coordinate pipeline\nA fake extractor whose features encode the original "
             "pixel position: matching an image against itself must give back the query point, with error well "
             "below the PCK tolerance."),
          code("""class _FakeExtractor:
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
print("coordinate pipeline OK")"""),
          md("## 9. Pipeline functions (feature extraction, evaluation, CV, submission)"),
          code(cli_functions()),
          md("## 10. Resolution check (zero-shot, frozen first backbone)\nThis compares input resolutions "
             "without any training, so it is an unbiased choice of `long_side`. Features are freed afterwards."),
          code("""if CFG.run_res_sweep:
    first = CFG.backbones.split(",")[0].split(":")[0]
    res_scores = {}
    for ls in (644, 840):
        c = SimpleNamespace(**{**vars(CFG), "backbones": first, "long_side": ls, "finetune_blocks": 0})
        print(f"===== {first} @ long side {ls} =====")
        res_scores[ls] = cmd_eval(c)
        gc.collect(); torch.cuda.empty_cache()
    print("zero-shot PCK by resolution:", res_scores)
    # CFG.long_side = max(res_scores, key=res_scores.get)   # uncomment to pick automatically"""),
          md("## 11. Zero-shot evaluation of the full configuration\nThis uses the frozen ensemble at `CFG.long_side`, "
             "and also reports each backbone alone."),
          code("""if CFG.run_eval:
    cmd_eval(CFG)
    gc.collect(); torch.cuda.empty_cache()"""),
          md("## 12. Cross-validation: zero-shot vs fine-tuned model, plus tuning of the inference knobs\n"
             "Every fold trains for a fixed number of epochs, with no best-epoch picking. The soft-argmax and "
             "geometric-prior settings are chosen on the pooled **out-of-fold** predictions and then reused "
             "for the test set."),
          code("""BEST = None
if CFG.run_cv and (CFG.adapter or CFG.finetune_blocks > 0):
    BEST = cmd_cv(CFG)
    gc.collect(); torch.cuda.empty_cache()
if BEST:
    CFG.radius, CFG.tau, CFG.geo_lambda = BEST["radius"], BEST["tau"], BEST["geo_lambda"]
print("inference settings for the test set:", CFG.radius, CFG.tau, CFG.geo_lambda)"""),
          md("## 13. Final model and submission\nThe model is retrained on **all** training pairs, then the "
             "test set is predicted. Coordinates are in original target-image pixels."),
          code("""if CFG.run_predict:
    cmd_predict(CFG)
    gc.collect(); torch.cuda.empty_cache()
    sub = pd.read_csv(Path(CFG.out) / "submission.csv")
    test_ids = pd.read_csv(Path(CFG.data) / "test.csv").id  # same ids as sample_submission.csv
    assert list(sub.columns) == ["id", "prediction"] and set(sub.id) == set(test_ids) and len(sub) == len(test_ids)
    print(sub.head())"""),
          md("## 14. Qualitative examples (training pairs, zero-shot)\nGreen: accepted targets, red: prediction, "
             "circle: PCK@0.10 tolerance."),
          code("""import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def show_examples(n=4, seed=1):
    small = SimpleNamespace(**{**vars(CFG), "finetune_blocks": 0, "limit_pairs": 40})  # cheap: frozen, 40 pairs
    banks, cfg, data = setup(small, ["train"])
    pairs = random.Random(seed).sample(data["train"], n)
    m = Matcher(banks, cfg)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    for r, p in enumerate(pairs):
        q = p.queries[0]
        pred = m.predict_pair(p)[q.id]
        axes[r, 0].imshow(load_image(CFG.data, p.source_image)); axes[r, 0].plot(*q.src_xy, "yo", ms=8)
        axes[r, 1].imshow(load_image(CFG.data, p.target_image))
        axes[r, 1].plot(q.targets[:, 0], q.targets[:, 1], "g+", ms=12, mew=2)
        for t in q.targets:
            axes[r, 1].add_patch(Circle(t, 0.1 * q.scale, fill=False, color="g", lw=1))
        axes[r, 1].plot(*pred, "rx", ms=10, mew=2)
        axes[r, 0].set_title(f"{p.pair_id} source"); axes[r, 1].set_title("target")
        for a in axes[r]: a.axis("off")
    plt.tight_layout(); plt.show()

show_examples()"""),
          ]

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, OUT)
print("wrote", OUT)
