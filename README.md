# Reproduction of Decrop et al. 2025 — phytoplankton CNN classifier

[![CI](https://github.com/annefou/fiesta-decrop-reproduction/actions/workflows/ci.yml/badge.svg)](https://github.com/annefou/fiesta-decrop-reproduction/actions/workflows/ci.yml)
[![Jupyter Book](https://github.com/annefou/fiesta-decrop-reproduction/actions/workflows/jupyter-book.yml/badge.svg)](https://annefou.github.io/fiesta-decrop-reproduction/)
[![Docker](https://github.com/annefou/fiesta-decrop-reproduction/actions/workflows/docker.yml/badge.svg?event=release)](https://github.com/annefou/fiesta-decrop-reproduction/pkgs/container/fiesta-decrop-reproduction)
[![Source DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19701133.svg)](https://doi.org/10.5281/zenodo.19701133)
[![Docker image DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19709184.svg)](https://doi.org/10.5281/zenodo.19709184)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FAIR4RS](https://img.shields.io/badge/FAIR4RS-conformant-brightgreen)](https://doi.org/10.15497/RDA00068)
[![FORRT](https://img.shields.io/badge/FORRT-replication-blue)](https://forrt.org/)
[![Science Live](https://img.shields.io/badge/Science%20Live-nanopub%20chain-purple)](https://w3id.org/sciencelive/np/RAn_DYEINS9hSmULEWNd8JmdPOykzptJPJCSqiatNLBrA)

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

## FORRT nanopublication chain

The full provenance of this reproduction is recorded as a six-step FORRT
nanopublication chain on the
[Science Live](https://platform.sciencelive4all.org) platform. Each step is
independently citable and machine-readable; together they form the FAIR
provenance receipt for this reproduction.

> **Headline assertion — machine-readable:**
> [**This reproduction `cito:confirms` Decrop et al. 2025**](https://w3id.org/sciencelive/np/RArFs1V-cAEw19zHbMFXmTr-R34PlGdMHCOt6EQeW64jU)
>
> This single nanopublication encodes the conclusion in a form that
> discovery tools (Scholia, Wikidata pipelines, SPARQL endpoints) can
> follow automatically — the paper's headline claim has been independently
> verified, and the verifier is this repository.

The five preceding nanopubs build the provenance ladder up to that citation:

| Step | Type | Asserts | Nanopub URI |
|---|---|---|---|
| 1 | Quote-with-comment | Verbatim quote of Decrop et al. 2025's headline performance claim, with a personal comment on its operational potential | [`RAm0R…`](https://w3id.org/sciencelive/np/RAm0R_xdfpZCX8jGhrMU8LQonht0X6qvfApCekmUCwvgQ) |
| 2 | AIDA sentence | Atomic, declarative restatement: a CNN identifies phytoplankton taxa across 95 classes at 86.3% top-1 / 98.8% top-5 | [`RAlNK…`](https://w3id.org/sciencelive/np/RAlNKRixcGMJ6-pz1AWMhtMv4JChJEock-rBRpEKUroas) |
| 3 | FORRT Claim (model performance) | The classification claim, typed as a FORRT model-performance claim | [`RAQbv…`](https://w3id.org/sciencelive/np/RAQbvusYubgaYlU7YEPIgPmTwqDoylwiq5FKyrVgF95qM) |
| 4 | FORRT Reproduction Study | Methodology: same code package (`planktonclas`), same weights, same `test.txt` split | [`RAGtx…`](https://w3id.org/sciencelive/np/RAGtxXgvYl-b7NOkyS3K34z1yDDkPqzGuiTr54nA2uH7U) |
| 5 | FORRT Replication Outcome (Validated, VeryHigh) | All five metrics match within 0.003 percentage points | [`RAn_D…`](https://w3id.org/sciencelive/np/RAn_DYEINS9hSmULEWNd8JmdPOykzptJPJCSqiatNLBrA) |
| 6 | **CiTO citation — `cito:confirms` Decrop 2025** | The headline assertion above | [**`RArFs…`**](https://w3id.org/sciencelive/np/RArFs1V-cAEw19zHbMFXmTr-R34PlGdMHCOt6EQeW64jU) |

The chain runs: paper → quote → atomic claim → FORRT claim → study (this repo)
→ outcome (the numbers in the Results table) → CiTO citation back to the
paper.

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

A Docker container is built on every release, pushed to GitHub Container
Registry, and archived to Zenodo.

```bash
docker pull ghcr.io/annefou/fiesta-decrop-reproduction:latest
docker run --rm -v "$PWD/results:/app/results" \
    ghcr.io/annefou/fiesta-decrop-reproduction:latest
```

Zenodo-archived image tarballs via the
[Docker image concept DOI 10.5281/zenodo.19709184](https://doi.org/10.5281/zenodo.19709184).

## How to cite

If you use this repository, please cite it via its Zenodo DOI together
with the paper it reproduces (Decrop et al. 2025).

```
Fouilloux, A. (2026). Reproduction of Decrop et al. 2025 — phytoplankton
CNN classifier (v0.2.1). Zenodo. https://doi.org/10.5281/zenodo.19701133
```

BibTeX:

```bibtex
@software{fouilloux_fiesta_decrop_reproduction,
  author    = {Fouilloux, Anne},
  title     = {Reproduction of Decrop et al. 2025 — phytoplankton CNN classifier},
  year      = {2026},
  version   = {0.2.1},
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
