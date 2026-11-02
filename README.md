# AML Challenge 2026/27 — Semantic Correspondence

Given a source image, a target image and a source point, predict the corresponding point in the target image. Metric: **PCK@0.10**, where a prediction counts as correct when its distance to the nearest accepted target is at most 10% of the longest side of the target object's bounding box.

Leaderboard reference points: DINOv3 baseline 0.129, pass threshold 0.297, current top score 0.624.

## Pipeline

1. **Frozen dense features** from DINOv2 or DINOv3 (HuggingFace `transformers`). Each image is resized so its long side is about `--long-side` pixels, rounded to a multiple of the patch size, with the aspect ratio kept. Several backbones and layers can be combined with weights.
2. **Matching.** The source feature is sampled bilinearly at the query point. Cosine similarity is computed against every target patch and upsampled to a `--work-res` grid. The final point comes from an argmax followed by a local soft-argmax (`--radius`, `--tau`).
3. **Flip alignment** (`--flip auto`). For each pair, the code decides whether to use the source image as-is or mirrored. It picks the orientation in which *all* the source points of the pair (`source_xy`, context points included) match the target best. This helps with left/right ambiguity and opposite viewpoints.
4. **Learned adapter** (`--adapter`, optional), or **partial fine-tuning** (`--finetune-blocks K`): the tokens entering the last K blocks are cached once, and the last K blocks plus the adapter head are trained. A small convolutional head is trained on top of the cached frozen features. It uses a dense contrastive loss whose soft label is a Gaussian around the nearest accepted target, so multiple accepted answers are handled naturally. Training is augmented by mirroring both images of a pair together.

**Validation.** Folds are built from the connected components of the pair graph, so no image appears in two folds. This mirrors the official train/test separation (`semcorr.data.make_folds`).

## Usage

```bash
pip install -r requirements.txt
export PYTHONPATH=src
DATA=/path/to/participant_data   # contains train.csv, test.csv, *_pairs.json, images/

# zero-shot PCK on the whole train set, plus a flip/soft-argmax ablation
python -m semcorr.cli eval --data $DATA --backbones dinov2_l_reg --long-side 644

# K-fold CV, zero-shot vs adapter
python -m semcorr.cli cv --data $DATA --backbones dinov2_l_reg --adapter --epochs 5

# K-fold CV with partial fine-tuning of the last 2 ViT blocks (+ adapter head)
python -m semcorr.cli cv --data $DATA --backbones dinov2_l_reg --finetune-blocks 2 --epochs 5

# submission (adapter trained on all of train; add --finetune-blocks 2 for the fine-tuned model)
python -m semcorr.cli predict --data $DATA --backbones dinov2_l_reg --adapter --epochs 5 --out outputs
```

Useful flags:

- `--cache-dir` saves extracted features so later runs skip extraction.
- `--limit-pairs N` gives a quick debug run.
- `--layers=-1,18` concatenates an intermediate layer with the final one.
- `--backbones dinov2_l_reg:1,dinov3_l:1` builds an ensemble. The adapter is trained on the first backbone.

Backbone aliases live in `src/semcorr/backbones.py`. Any HF id or local path also works. DINOv3 is **gated** on HF: accept the license on the model page and set `HF_TOKEN` (on Kaggle: Add-ons → Secrets).

Sanity test for the coordinate mapping, which runs offline: `PYTHONPATH=src python tests/test_geometry.py`

## Kaggle

The notebooks are single and self-contained, with all the code inlined, so they need no repo clone. Upload one to Kaggle, add the competition data, enable GPU and Internet, and choose *Run All*.

- **`notebooks/aml_semcorr_v2_1.ipynb`** (current, v2.1: fixes DINOv3 fp16 NaNs). `aml_semcorr_v2.ipynb` is the buggy first v2. It runs at 840 px, adds a DINOv3 ensemble when the `HF_TOKEN` secret exists, fine-tunes the last 2 blocks of every backbone, uses source/target swap augmentation, and tunes soft-argmax and the geometric prior on out-of-fold CV.
- `notebooks/aml_semcorr_standalone.ipynb` (v1, frozen; CV 0.5435). It is kept as the reference version.
- `notebooks/kaggle_run.ipynb` clones the repo instead. It is handy while iterating.

**The package in `src/semcorr` is the source of truth.** After changing it, regenerate the current notebook (v2.1):

```bash
python tools/build_notebook.py
```

## Rules checklist

- Use only the competition data plus pretrained models. No extra SOCO data and no external datasets.
- No hand-labelling of test images and no lookups by id or filename.
- Open-source code under OSI licenses.

## Roadmap / ideas

- [x] Resolution check 644 vs 840 (v2). Multi-scale feature averaging is still open.
- [x] DINOv2 + DINOv3 ensemble (v2, needs HF_TOKEN). Still open: Stable Diffusion (DIFT) feature fusion (SD+DINO is a strong zero-shot combo).
- [x] Source/target swap augmentation (v2). Still open: stronger regularisation (it overfits after ~3-5 epochs), train on the concatenation of backbones.
- [x] Geometric-consistency prior from a RANSAC affine fit of all source keypoints (v2, `--geo-lambda`, tuned by `--sweep`). TPS is still open.
- [x] Partial fine-tuning of the last K ViT blocks on cached intermediate tokens (`--finetune-blocks K`). LoRA on more blocks is a possible next step.
