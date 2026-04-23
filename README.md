# Reproduction of Decrop et al. 2025 — phytoplankton CNN classifier

[![Source DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19701133.svg)](https://doi.org/10.5281/zenodo.19701133)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Independent computational reproduction** of the phytoplankton image classifier
> reported in:
>
> Decrop, W., Lagaisse, R., Mortelmans, J., Muñiz, C., Heredia, I., Calatrava, A.,
> Deneudt, K. (2025). *Automated image classification workflow for phytoplankton
> monitoring.* **Frontiers in Marine Science 12: 1699781.**
> DOI: [10.3389/fmars.2025.1699781](https://doi.org/10.3389/fmars.2025.1699781)

This repository runs the authors' own pre-trained **EfficientNetV2-B0** model
([Zenodo 10.5281/zenodo.15269453](https://doi.org/10.5281/zenodo.15269453)) on
their exact held-out test set (`test.txt`, 33,718 images across 95 classes) and
compares every reported metric to the paper. This is a **FAIR reproduction**:
same data, same weights, same split, run independently on different hardware.

## Results

We reproduce all five paper-reported metrics to within **0.003 percentage points**.

| Metric | Paper (Decrop et al. 2025) | This reproduction | Delta (pp) |
|---|---:|---:|---:|
| Top-1 accuracy | 86.34 % | **86.3426 %** | +0.003 |
| Top-5 accuracy | 98.76 % | **98.7633 %** | +0.003 |
| Micro F1      | 86.34 % | **86.3426 %** | +0.003 |
| Macro F1      | 78.76 % | **78.7583 %** | -0.002 |
| Weighted F1   | 86.25 % | **86.2470 %** | -0.003 |

Inference took **27 minutes** on a single M1 Pro CPU core pool (10-crop test-time
augmentation, as in the paper).

Full results JSON + per-image prediction matrix are written to `results/`.

## Two notebooks

| Notebook | Input | Output | Purpose |
|---|---|---|---|
| `01_reproduce_decrop.py` | `test.txt` (33,718 images) | `results/reproduce_decrop_results.json` + `reproduce_decrop_predictions.npz` | Reproduce the five published metrics |
| `02_cnn_val_predictions.py` | `val.txt` (33,829 images) | `results/cnn_predictions_val.npz` | Provide CNN predictions on the held-out **val** split for downstream consumers (e.g. [fiesta-scattering-bio](https://github.com/annefou/fiesta-scattering-bio) uses these as meta-training data for a stacked CNN + scattering classifier) |

Both notebooks use **identical pipelines** — same pretrained weights, same
preprocessing, same 10-crop TTA. The only difference is which split they
consume.

## What this reproduction establishes

1. **The pre-trained model behaves as the paper reports.** The Zenodo weights
   give the published numbers bit-for-bit when run on the distributed
   `test.txt` split.
2. **The model runs on a CPU-only laptop in < 30 minutes** — no GPU required
   for inference. This matters for edge deployment and teaching.
3. **All artefacts needed for reproduction are openly archived**: dataset
   (Zenodo 10554845, CC-BY 4.0), model weights (Zenodo 15269453, CC-BY 4.0),
   source code ([`planktonclas`](https://github.com/lifewatch/planktonclas),
   Apache 2.0).

## What this reproduction does NOT do

- It does **not** re-train the model from scratch. That would require the
  authors' training code, GPU infrastructure, and days of compute.
- It does **not** validate the *training* pipeline — only that the distributed
  weights, when run on the distributed test split, produce the distributed
  numbers.
- A training-reproduction is planned as a separate repository.

## How to run

```bash
# 1. Create environment
micromamba create -f environment.yml
micromamba activate fiesta-decrop-reproduction

# 2. Convert and execute the notebook
#    (downloads ~700 MB of images + 47 MB pretrained model on first run)
jupytext --to notebook 01_reproduce_decrop.py
jupyter execute --inplace 01_reproduce_decrop.ipynb
```

Or, with Docker:

```bash
docker run --rm -v "$PWD/results:/app/results" \
    ghcr.io/annefou/fiesta-decrop-reproduction:latest
```

## Data and model provenance

- **Dataset**: [LifeWatch FlowCam phytoplankton annotated training set](https://doi.org/10.5281/zenodo.10554845)
  (VLIZ / LifeWatch Belgium, CC-BY 4.0). 337,567 annotated images, 95 classes.
  Split into `train.txt` (270,020), `val.txt` (33,829), `test.txt` (33,718) by
  the paper's authors.
- **Pretrained model**: [Phytoplankton_EfficientNetV2B0 (Zenodo 15269453)](https://doi.org/10.5281/zenodo.15269453),
  CC-BY 4.0. Distributed with the exact training/validation/test split files
  used in the paper.
- **Model code**: [`planktonclas`](https://github.com/lifewatch/planktonclas)
  (Wout Decrop, VLIZ). Apache 2.0.

## FIESTA-OSCARS companion repositories

This reproduction is part of the **FIESTA-OSCARS** project — Fair Image analysis
across sciencES. The CNN baseline established here is used for comparison in
the scattering-transform experiments:

| Repository | Scientific content |
|---|---|
| [fiesta-scattering-astro](https://github.com/annefou/fiesta-scattering-astro) | Scattering transform for astrophysical map synthesis (LSS / Planck-like) |
| [fiesta-scattering-sst](https://github.com/annefou/fiesta-scattering-sst) | Scattering transform for SST gap-filling (Copernicus Marine) |
| [fiesta-scattering-bio](https://github.com/annefou/fiesta-scattering-bio) | Scattering transform for plankton texture classification |
| [fiesta-scattering-sst-healpix-geo](https://github.com/annefou/fiesta-scattering-sst-healpix-geo) | SST gap-filling on WGS84 ellipsoid (healpix-geo) |
| **this repo** | Independent reproduction of the CNN baseline used by the bio comparison |

## Container image

A Docker container is built on every release and pushed to GitHub Container
Registry.

```bash
docker pull ghcr.io/annefou/fiesta-decrop-reproduction:latest
docker run --rm -v "$PWD/results:/app/results" \
    ghcr.io/annefou/fiesta-decrop-reproduction:latest
```

(Zenodo archive of the Docker image tarball will be available starting with
the next release; `ZENODO_TOKEN` was added after v0.2.0.)

## How to cite

If you use this repository, please cite it via its Zenodo DOI together
with the paper it reproduces (Decrop et al. 2025).

```
Fouilloux, A. (2026). Reproduction of Decrop et al. 2025 — phytoplankton
CNN classifier (v0.2.0). Zenodo. https://doi.org/10.5281/zenodo.19701133
```

BibTeX:

```bibtex
@software{fouilloux_fiesta_decrop_reproduction,
  author    = {Fouilloux, Anne},
  title     = {Reproduction of Decrop et al. 2025 — phytoplankton CNN classifier},
  year      = {2026},
  version   = {0.2.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19701133},
  url       = {https://doi.org/10.5281/zenodo.19701133}
}
```

The DOI above is the **concept DOI** — it always resolves to the latest
release. Specific version DOIs are available on the
[Zenodo record page](https://doi.org/10.5281/zenodo.19701133).

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

## Credits

- **Paper authors**: Decrop et al. 2025 (VLIZ / LifeWatch Belgium / iMagine
  project). We reproduce their published work using their openly archived
  artefacts.
- **This reproduction**: Anne Fouilloux, LifeWatch ERIC (FIESTA-OSCARS).

## License

Code: MIT. See `LICENSE`.
Data and pretrained model are CC-BY 4.0 — cite Decrop et al. 2025 and the
LifeWatch dataset (10554845) when re-using.
