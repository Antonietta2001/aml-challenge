"""Build aml_lab_kaggle.ipynb: semcorr.py and lab.py inlined with %%writefile, then the tests and a submission.
On Kaggle: upload the notebook, add the competition data, Accelerator GPU, Internet on, Run All.

    python build_kaggle.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def cell(kind, src):
    extra = {"outputs": [], "execution_count": None} if kind == "code" else {}
    return {"cell_type": kind, "metadata": {}, "source": src.strip("\n"), **extra}


md = lambda s: cell("markdown", s)
code = lambda s: cell("code", s)

cells = [
    md("""
# AML Challenge 2026/27: idea lab (Kaggle GPU)

Builds on the v2.1 pipeline (`semcorr.py`, verbatim plus two small additions) and tests new ideas against it.
**Run All** with Accelerator GPU, Internet on, and the competition data added.
Defaults: `LAB = '{"zoom": 1.0}'` (Test 1's winner, PCK 0.562, so Test 1 is skipped) and `QUICK = False`
(Tests 2-3 and the fine-tuned submission, ~3 h). `LAB = None` re-runs Test 1 (~1 h); `QUICK = True` stops after the
zero-shot submission. Kaggle stops a session after 12 h.

| Test | What it answers | Time (P100) |
|---|---|---|
| 1. Zero-shot ablation | Which inference ideas help: *zoom* (small objects), *cycle* (look-alike background / wrong part), *context=queries*, *geo*. Uses all 2,495 train queries, since nothing is trained. | ~1 h |
| 2. Fine-tuning CV | Plain fine-tuning vs **BERT/MAE-style token masking** (the hidden tokens' features must be inferred from context). Each out-of-fold query is scored three ways: zero-shot, fine-tuned, and fine-tuned + the best ideas from Test 1. | ~1.5-2 h per mask setting (`FOLDS=3` halves it) |
| 3. DINOv2+DINOv3 | Ensemble with the best ideas. Runs only when the `HF_TOKEN` secret exists. | ~30 min |
| Submission | Zero-shot fallback right after Test 1, then the fine-tuned winner. | ~5 + 30 min |

**How to read the tables.** `delta [95% CI]` is the paired bootstrap (resampling whole pairs) against the first row.
An idea is real when the whole interval is above 0. The error taxonomy shows which hazard an idea fixed:
*near miss* is localization (small objects), *another part* is repeated or look-alike parts, and
*background / other instance* is look-alike background. The top-k oracle is the ceiling for any re-ranking idea.
"""),
    code("""
import glob, json, os, shutil

DATA = os.path.dirname(glob.glob("/kaggle/input/**/train_pairs.json", recursive=True)[0])
FOLDS, EPOCHS, MASKS = 5, 3, (0.0, 0.3)   # FOLDS=3 halves Test 2; MASKS=(0.0,) skips the masked run
QUICK = False                               # True: Test 1 + zero-shot submission only (~1 h)
LAB = '{"zoom": 1.0}'                       # Test 1 winner (PCK 0.562); None re-runs Test 1
HAS_TOKEN = False                           # DINOv3 is gated: accept its licence on HF, add the HF_TOKEN secret
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    HAS_TOKEN = True
except Exception:
    pass
print("data:", DATA, "| DINOv3 test:", HAS_TOKEN)
if HAS_TOKEN and not QUICK:  # DINOv3 needs transformers >= 4.56
    import transformers
    from packaging.version import Version
    if Version(transformers.__version__) < Version("4.56"):
        !pip install -q -U "transformers>=4.56"
        print("transformers upgraded: restart the kernel and Run All again")
"""),
    code("%%writefile semcorr.py\n" + (HERE / "semcorr.py").read_text()),
    code("%%writefile lab.py\n" + (HERE / "lab.py").read_text()),
    code("!python semcorr.py && python lab.py selftest"),
    md("## Test 1: zero-shot ablation (DINOv2-L/14-reg, 840 px)"),
    code("""
if LAB is None:
    !python lab.py ablate --data "{DATA}"
    rows = json.load(open("lab_out/ablation.json"))
    best = max(rows, key=lambda r: r["pck"])
    LAB = json.dumps(best["cfg"])
    print(f"best zero-shot row: {best['name']}  PCK {best['pck']:.4f}  delta {best['delta']:+.4f} CI {best['ci']}")
print("ideas used from here on: --cfg", LAB)
"""),
    code("""
# zero-shot submission with the best ideas (fallback / quick leaderboard check)
# Steps whose output file already exists are skipped, so Run All resumes; delete the file to redo a step.
if not os.path.exists("lab_out/zeroshot/submission.csv"):
    !python lab.py predict --data "{DATA}" --cfg '{LAB}' --out lab_out/zeroshot
shutil.copy("lab_out/zeroshot/submission.csv", "submission.csv")
"""),
    md("## Test 2: fine-tuning CV, without vs with BERT/MAE-style token masking"),
    code("""
if not QUICK:
    for r in MASKS:
        if os.path.exists(f"lab_out/oof_dinov2_l_reg_840_mask{r}.json"):
            print(f"mask {r}: already done, skipping")
            continue
        !python lab.py ft-cv --data "{DATA}" --folds {FOLDS} --epochs {EPOCHS} --mask-ratio {r} --cfg '{LAB}'
"""),
    code("""
if not QUICK:
    import lab as L
    f = {r: f"lab_out/oof_dinov2_l_reg_840_mask{r}.json" for r in MASKS}
    A = f[MASKS[0]]
    print("fine-tuned vs fine-tuned + best zero-shot ideas (paired, out-of-fold):")
    !python lab.py compare --data "{DATA}" "{A}:fine-tuned" "{A}:fine-tuned + lab"
    for r in MASKS[1:]:
        B = f[r]
        print(f"mask {r} vs mask {MASKS[0]}: plain fine-tuning, then with the ideas on top")
        !python lab.py compare --data "{DATA}" "{A}:fine-tuned" "{B}:fine-tuned"
        !python lab.py compare --data "{DATA}" "{A}:fine-tuned + lab" "{B}:fine-tuned + lab"
    q = L.sc.queries_of(L.sc.load_split(DATA, "train"))
    cands = {(r, k): L.sc.pck(json.load(open(f[r]))[k], q) for r in MASKS for k in ("fine-tuned", "fine-tuned + lab")}
    for k, v in sorted(cands.items(), key=lambda kv: -kv[1]):
        print(f"  mask {k[0]}  {k[1]:17s} CV PCK {v:.4f}")
    MASK, KIND = max(cands, key=cands.get)
    SUB_CFG = LAB if KIND == "fine-tuned + lab" else "{}"
"""),
    md("## Test 3: DINOv2 + DINOv3 ensemble (only with the HF_TOKEN secret)"),
    code("""
if HAS_TOKEN and not QUICK:
    ens_rows = json.dumps([["dinov2+dinov3 baseline", {}], ["dinov2+dinov3 + best ideas", json.loads(LAB)]])
    !python lab.py ablate --data "{DATA}" --backbone dinov2_l_reg:1,dinov3_l:1 --rows '{ens_rows}' --out lab_out/ens
"""),
    md("## Submission: fine-tuned on all of train with the CV winner"),
    code("""
if not QUICK:
    if not os.path.exists("lab_out/final/submission.csv"):
        !python lab.py predict --data "{DATA}" --finetune --epochs {EPOCHS} --mask-ratio {MASK} --cfg '{SUB_CFG}' --out lab_out/final
    shutil.copy("lab_out/final/submission.csv", "submission.csv")
print(open("submission.csv").read()[:200])
"""),
]

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}}
out = HERE / "aml_lab_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
