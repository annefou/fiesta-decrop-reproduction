# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CNN predictions on Decrop's `val.txt` (for downstream use)
#
# This notebook runs Decrop et al. 2025's pre-trained EfficientNetV2-B0 on
# their **validation** split (`val.txt`, 33,829 images) and saves the full
# 95-class probability matrix. The primary consumer is
# [fiesta-scattering-bio](https://github.com/annefou/fiesta-scattering-bio),
# which uses these predictions as meta-training data for a stacked
# CNN + scattering classifier.
#
# Compared with `01_reproduce_decrop.py`, this notebook:
#
# - runs the **same pretrained model** on the **same artefacts**,
# - but targets `val.txt` (held out from training but **not** used in the
#   paper's reported 86.34 % test accuracy),
# - produces a **persistent artefact** (`results/cnn_predictions_val.npz`)
#   that downstream repositories can reuse without re-running inference.

# %%
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
from sklearn.metrics import (
    accuracy_score, top_k_accuracy_score, f1_score, classification_report,
)

# %%
PROJECT = Path.cwd()
DATA_DIR = PROJECT / 'data'
IMAGES_DIR = DATA_DIR / 'images_DS'
MODELS_DIR = PROJECT / 'models'
TIMESTAMP = 'Phytoplankton_EfficientNetV2B0'
MODEL_DIR = MODELS_DIR / TIMESTAMP
MODEL_NAME = 'final_model.h5'

RESULTS = PROJECT / 'results'
RESULTS.mkdir(exist_ok=True)

CI_MODE = os.environ.get('CI', '').lower() in ('true', '1')
CI_SAMPLE_SIZE = 200 if CI_MODE else None

print(f'CI_MODE = {CI_MODE}')
print(f'Project root: {PROJECT}')

# %% [markdown]
# ## 1. Download the dataset (reused from 01_reproduce_decrop.py)

# %%
if not IMAGES_DIR.exists():
    print('Downloading LifeWatch FlowCam plankton dataset from Zenodo 10554845...')
    archive = DATA_DIR / 'phytoplankton.7z'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        url = (
            'https://zenodo.org/records/10554845/files/'
            'phytoplankton_images_and_datasplit.7z?download=1'
        )
        urllib.request.urlretrieve(url, str(archive))
        print(f'  Downloaded {archive.stat().st_size / 1e6:.0f} MB')
    print('  Extracting...')
    subprocess.run(
        ['7z', 'x', str(archive), f'-o{archive.parent}', '-y'], check=True,
    )
    print(f'  {sum(1 for _ in IMAGES_DIR.rglob("*.jpg")):,} images')
else:
    print(f'Dataset present at {IMAGES_DIR}')

# %% [markdown]
# ## 2. Download the pretrained model

# %%
if not (MODEL_DIR / 'ckpts' / MODEL_NAME).exists():
    print('Downloading pretrained EfficientNetV2-B0 from Zenodo 15269453...')
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    tar_path = MODELS_DIR / f'{TIMESTAMP}.tar.gz'
    if not tar_path.exists():
        url = (f'https://zenodo.org/records/15269453/files/'
               f'{TIMESTAMP}.tar.gz?download=1')
        urllib.request.urlretrieve(url, str(tar_path))
        print(f'  Downloaded {tar_path.stat().st_size / 1e6:.1f} MB')
    with tarfile.open(tar_path, 'r:gz') as tf:
        tf.extractall(MODELS_DIR)
    if not (MODEL_DIR / 'ckpts' / MODEL_NAME).exists():
        candidates = [p for p in MODELS_DIR.iterdir()
                      if p.is_dir() and (p / 'ckpts' / MODEL_NAME).exists()]
        if candidates and candidates[0] != MODEL_DIR:
            candidates[0].rename(MODEL_DIR)
    print(f'  Model at {MODEL_DIR}')
else:
    print(f'Model present at {MODEL_DIR}')

# %% [markdown]
# ## 3. Load the val.txt split distributed with the model

# %%
DS_DIR = MODEL_DIR / 'dataset_files'
val_txt = DS_DIR / 'val.txt'

val_paths, val_labels = [], []
with open(val_txt) as f:
    for line in f:
        rel, label = line.strip().rsplit(' ', 1)
        full = IMAGES_DIR / rel
        if not full.exists():
            continue
        val_paths.append(str(full))
        val_labels.append(int(label))
val_labels = np.array(val_labels)

if CI_SAMPLE_SIZE is not None and len(val_paths) > CI_SAMPLE_SIZE:
    rng = np.random.default_rng(0)
    keep = rng.choice(len(val_paths), size=CI_SAMPLE_SIZE, replace=False)
    val_paths = [val_paths[i] for i in keep]
    val_labels = val_labels[keep]
    print(f'CI smoke run: sampled {len(val_paths)} / 33,829 val images')
else:
    print(f'Full val run: {len(val_paths)} images')

# %% [markdown]
# ## 4. Load model + predict

# %%
from planktonclas import config, paths, utils
from planktonclas.test_utils import predict
from tensorflow.keras.models import load_model

config.set_config_path(str(PROJECT / 'config.yaml'))
paths.CONF = config.get_conf_dict()
paths.timestamp = TIMESTAMP

with open(os.path.join(paths.get_conf_dir(), 'conf.json')) as f:
    conf = json.load(f)

ckpt = os.path.join(paths.get_checkpoints_dir(), MODEL_NAME)
print(f'Loading {ckpt}')
model = load_model(ckpt, custom_objects=utils.get_custom_objects())

with open(DS_DIR / 'classes.txt') as f:
    all_class_names = [ln.strip() for ln in f if ln.strip()]

# %%
print(f'\nPredicting {len(val_paths)} val images (10-crop TTA)...')
t0 = time.time()
pred_lab, pred_prob = predict(
    model, val_paths, conf,
    top_K=len(all_class_names), filemode='local',
)
elapsed = time.time() - t0
print(f'Inference: {elapsed/60:.1f} min ({elapsed/len(val_paths)*1000:.1f} ms/image)')

N = len(val_paths)
full_probs = np.zeros((N, len(all_class_names)), dtype=np.float32)
for i in range(N):
    full_probs[i, pred_lab[i]] = pred_prob[i]

# %% [markdown]
# ## 5. Sanity metrics (val is held out from training so these are meaningful)

# %%
y_pred = full_probs.argmax(axis=1)
acc_top1 = accuracy_score(val_labels, y_pred)
acc_top5 = top_k_accuracy_score(val_labels, full_probs, k=5,
                                 labels=np.arange(len(all_class_names)))
macro_f1 = f1_score(val_labels, y_pred, average='macro', zero_division=0)

print(f'\nVal top-1: {acc_top1:.4f}  top-5: {acc_top5:.4f}  macro-F1: {macro_f1:.4f}')
if CI_MODE:
    print('⚠ CI smoke run — only 200 images, numbers are not representative.')

# %% [markdown]
# ## 6. Save

# %%
suffix = '_smoke' if CI_MODE else ''
out_path = RESULTS / f'cnn_predictions_val{suffix}.npz'
np.savez_compressed(
    out_path,
    y_true=val_labels,
    y_pred=y_pred,
    full_probs=full_probs.astype(np.float16),
    val_paths=np.array(val_paths),
)

summary = {
    'purpose': 'CNN predictions on Decrop et al. 2025 val.txt — downstream meta-training for fiesta-scattering-bio stacking',
    'paper_doi': '10.3389/fmars.2025.1699781',
    'model_doi': '10.5281/zenodo.15269453',
    'split': 'val.txt (held-out from training, 33,829 images)',
    'n_images': int(N),
    'top1': float(acc_top1),
    'top5': float(acc_top5),
    'macro_f1': float(macro_f1),
    'inference_seconds': float(elapsed),
    'ci_smoke': CI_MODE,
}
(RESULTS / f'cnn_predictions_val_summary{suffix}.json').write_text(json.dumps(summary, indent=2))

print(f'\nSaved predictions → {out_path.relative_to(PROJECT)}')
print(f'Saved summary     → results/cnn_predictions_val_summary{suffix}.json')
