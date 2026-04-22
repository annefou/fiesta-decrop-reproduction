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
# # Reproduction of Decrop et al. 2025
#
# This notebook runs an **independent computational reproduction** of the
# phytoplankton image classifier published in:
#
# > Decrop, W., Lagaisse, R., Mortelmans, J., Muñiz, C., Heredia, I.,
# > Calatrava, A., Deneudt, K. (2025).
# > *Automated image classification workflow for phytoplankton monitoring.*
# > Frontiers in Marine Science 12:1699781.
# > [DOI: 10.3389/fmars.2025.1699781](https://doi.org/10.3389/fmars.2025.1699781)
#
# The authors released three openly-licensed artefacts which together make
# this a FAIR reproduction with no guessing:
#
# 1. **Dataset** — 337,567 annotated FlowCam images in 95 classes
#    ([Zenodo 10.5281/zenodo.10554845](https://doi.org/10.5281/zenodo.10554845),
#    CC-BY 4.0)
# 2. **Pretrained model** — EfficientNetV2-B0 weights plus the exact
#    train/val/test split files used in the paper
#    ([Zenodo 10.5281/zenodo.15269453](https://doi.org/10.5281/zenodo.15269453),
#    CC-BY 4.0)
# 3. **Source code** — `planktonclas`
#    ([GitHub](https://github.com/lifewatch/planktonclas), Apache 2.0)
#
# We download (1) and (2), install (3) from PyPI, and run inference on the
# held-out `test.txt` to check whether we get the paper's numbers.
#
# ## Paper numbers to reproduce
#
# | Metric | Paper |
# |--------|------:|
# | Top-1 accuracy | 86.34 % |
# | Top-5 accuracy | 98.76 % |
# | Micro F1       | 86.34 % |
# | Macro F1       | 78.76 % |
# | Weighted F1    | 86.25 % |

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

# Smoke-test mode for CI — validates the pipeline on a small subset of
# test.txt. In this mode the paper numbers will NOT match because we use
# a random subset. For the real reproduction, run with CI unset.
CI_MODE = os.environ.get('CI', '').lower() in ('true', '1')
CI_SAMPLE_SIZE = 200 if CI_MODE else None

print(f'CI_MODE = {CI_MODE}')
print(f'Project root: {PROJECT}')

# %% [markdown]
# ## 1. Download the dataset (337k images, ~650 MB)

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

    print('  Extracting (this takes several minutes)...')
    subprocess.run(
        ['7z', 'x', str(archive), f'-o{archive.parent}', '-y'],
        check=True,
    )
    n_images = sum(1 for _ in IMAGES_DIR.rglob('*.jpg'))
    print(f'  Extracted {n_images:,} images')
else:
    print(f'Dataset already present at {IMAGES_DIR}')

# %% [markdown]
# ## 2. Download the pretrained model (47 MB)

# %%
if not (MODEL_DIR / 'ckpts' / MODEL_NAME).exists():
    print('Downloading pretrained EfficientNetV2-B0 from Zenodo 15269453...')
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    tar_path = MODELS_DIR / f'{TIMESTAMP}.tar.gz'

    if not tar_path.exists():
        url = (
            'https://zenodo.org/records/15269453/files/'
            f'{TIMESTAMP}.tar.gz?download=1'
        )
        urllib.request.urlretrieve(url, str(tar_path))
        print(f'  Downloaded {tar_path.stat().st_size / 1e6:.1f} MB')

    print('  Extracting...')
    with tarfile.open(tar_path, 'r:gz') as tf:
        tf.extractall(MODELS_DIR)

    # Zenodo tar extracts into a folder with the same name; check both layouts
    if not (MODEL_DIR / 'ckpts' / MODEL_NAME).exists():
        # Sometimes the tar's top-level folder is named differently; find it.
        candidates = [p for p in MODELS_DIR.iterdir()
                      if p.is_dir() and (p / 'ckpts' / MODEL_NAME).exists()]
        if candidates and candidates[0] != MODEL_DIR:
            candidates[0].rename(MODEL_DIR)

    print(f'  Model extracted at {MODEL_DIR}')
else:
    print(f'Pretrained model already present at {MODEL_DIR}')

# %% [markdown]
# ## 3. Load the test split distributed with the model

# %%
DS_DIR = MODEL_DIR / 'dataset_files'
test_txt = DS_DIR / 'test.txt'

test_paths = []
test_labels = []
with open(test_txt) as f:
    for line in f:
        rel, label = line.strip().rsplit(' ', 1)
        full = IMAGES_DIR / rel
        if not full.exists():
            continue
        test_paths.append(str(full))
        test_labels.append(int(label))
test_labels = np.array(test_labels)

if CI_SAMPLE_SIZE is not None and len(test_paths) > CI_SAMPLE_SIZE:
    rng = np.random.default_rng(0)
    keep = rng.choice(len(test_paths), size=CI_SAMPLE_SIZE, replace=False)
    test_paths = [test_paths[i] for i in keep]
    test_labels = test_labels[keep]
    print(f'CI smoke run: sampled {len(test_paths)} / 33,718 test images')
else:
    print(f'Full reproduction: {len(test_paths)} test images')

print(f'Classes in (sub)test: {len(np.unique(test_labels))}')

# %% [markdown]
# ## 4. Load the pretrained model via `planktonclas`

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
print(f'Input shape: {model.input_shape}  Output shape: {model.output_shape}')

classes_file = DS_DIR / 'classes.txt'
with open(classes_file) as f:
    all_class_names = [ln.strip() for ln in f if ln.strip()]
print(f'Model trained on {len(all_class_names)} classes')

# %% [markdown]
# ## 5. Run inference (10-crop test-time augmentation)

# %%
print(f'Predicting {len(test_paths)} images...')
t0 = time.time()
pred_lab, pred_prob = predict(
    model, test_paths, conf,
    top_K=len(all_class_names),
    filemode='local',
)
elapsed = time.time() - t0
print(f'Inference: {elapsed / 60:.1f} min '
      f'({elapsed / len(test_paths) * 1000:.1f} ms/image)')

# Reconstruct the full (N, 95) probability matrix in class-index order.
N = len(test_paths)
full_probs = np.zeros((N, len(all_class_names)), dtype=np.float32)
for i in range(N):
    full_probs[i, pred_lab[i]] = pred_prob[i]

# %% [markdown]
# ## 6. Compute metrics and compare to the paper

# %%
y_true = test_labels
y_pred = full_probs.argmax(axis=1)

acc_top1    = accuracy_score(y_true, y_pred)
acc_top5    = top_k_accuracy_score(y_true, full_probs, k=5,
                                   labels=np.arange(len(all_class_names)))
micro_f1    = f1_score(y_true, y_pred, average='micro', zero_division=0)
macro_f1    = f1_score(y_true, y_pred, average='macro', zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

paper = {
    'top1':        0.8634,
    'top5':        0.9876,
    'micro_f1':    0.8634,
    'macro_f1':    0.7876,
    'weighted_f1': 0.8625,
}

print('\n=== Reproduction vs Decrop et al. 2025 ===')
print(f'{"Metric":<20} {"Ours":>10} {"Paper":>10} {"Delta (pp)":>12}')
for name, ours, papv in [
    ('Top-1 accuracy', acc_top1, paper['top1']),
    ('Top-5 accuracy', acc_top5, paper['top5']),
    ('Micro F1',        micro_f1,    paper['micro_f1']),
    ('Macro F1',        macro_f1,    paper['macro_f1']),
    ('Weighted F1',     weighted_f1, paper['weighted_f1']),
]:
    print(f'{name:<20} {ours:>9.2%} {papv:>9.2%} {(ours - papv) * 100:>+11.3f}')

if CI_MODE:
    print('\n⚠ CI smoke run on a random subset — paper numbers will not match.')
    print('  For the real reproduction, run without CI=true.')

# %% [markdown]
# ## 7. Save results

# %%
out = {
    'reproduction_of': {
        'paper': 'Decrop et al. 2025, Frontiers in Marine Science 12:1699781',
        'paper_doi': '10.3389/fmars.2025.1699781',
        'model_doi': '10.5281/zenodo.15269453',
        'dataset_doi': '10.5281/zenodo.10554845',
    },
    'test_set': {
        'source': 'test.txt distributed with pretrained model',
        'n_images': int(N),
        'n_classes': int(len(np.unique(y_true))),
        'ci_smoke_subset': CI_MODE,
    },
    'inference': {
        'tta': '10-crop',
        'time_seconds': float(elapsed),
        'device': 'CPU',
    },
    'ours': {
        'top1_accuracy':  float(acc_top1),
        'top5_accuracy':  float(acc_top5),
        'micro_f1':       float(micro_f1),
        'macro_f1':       float(macro_f1),
        'weighted_f1':    float(weighted_f1),
    },
    'paper': paper,
    'deltas_pp': {
        'top1':        float((acc_top1    - paper['top1'])        * 100),
        'top5':        float((acc_top5    - paper['top5'])        * 100),
        'micro_f1':    float((micro_f1    - paper['micro_f1'])    * 100),
        'macro_f1':    float((macro_f1    - paper['macro_f1'])    * 100),
        'weighted_f1': float((weighted_f1 - paper['weighted_f1']) * 100),
    },
}

# CI smoke runs write to a *_smoke.json name so the canonical full-run
# artefact is never overwritten by a sub-sample.
suffix = '_smoke' if CI_MODE else ''
json_path = RESULTS / f'reproduce_decrop_results{suffix}.json'
npz_path = RESULTS / f'reproduce_decrop_predictions{suffix}.npz'

with open(json_path, 'w') as f:
    json.dump(out, f, indent=2)

np.savez_compressed(
    npz_path,
    y_true=y_true,
    y_pred=y_pred,
    full_probs=full_probs.astype(np.float16),
    test_paths=np.array(test_paths),
)

print(f'\nSaved:  {json_path.relative_to(PROJECT)}')
print(f'Saved:  {npz_path.relative_to(PROJECT)}')
