# Reproduction of Decrop et al. 2025

> Independent computational reproduction of the phytoplankton image classifier
> published in Decrop et al. 2025
> ([10.3389/fmars.2025.1699781](https://doi.org/10.3389/fmars.2025.1699781)).

## Why reproduce this paper?

Decrop et al. 2025 release three things under open licenses:

1. An **annotated dataset** of 337,567 FlowCam phytoplankton images across 95
   classes
   ([Zenodo 10.5281/zenodo.10554845](https://doi.org/10.5281/zenodo.10554845),
   CC-BY 4.0).
2. **Pretrained EfficientNetV2-B0 model weights**
   ([Zenodo 10.5281/zenodo.15269453](https://doi.org/10.5281/zenodo.15269453),
   CC-BY 4.0) distributed together with their exact train / validation / test
   split files.
3. **Source code**
   ([`planktonclas`](https://github.com/lifewatch/planktonclas), Apache 2.0).

This combination — open data + open weights + open splits + open code — makes
a FAIR computational reproduction possible with no guessing and no luck. This
repository does that reproduction.

## What we verify

Running the authors' own weights on the authors' own held-out test set, with
10-crop test-time augmentation as described in the paper, we obtain:

| Metric | Paper | Ours | Delta (pp) |
|---|---:|---:|---:|
| Top-1 accuracy | 86.34 % | **86.3426 %** | +0.003 |
| Top-5 accuracy | 98.76 % | **98.7633 %** | +0.003 |
| Micro F1      | 86.34 % | **86.3426 %** | +0.003 |
| Macro F1      | 78.76 % | **78.7583 %** | -0.002 |
| Weighted F1   | 86.25 % | **86.2470 %** | -0.003 |

Every number matches the paper to three decimal places. This is what a FAIR
reproduction is supposed to look like.

## What this reproduction does NOT cover

- **Training**: we do not re-train the model from scratch. That would require
  GPU infrastructure and several days of compute. A training-reproduction
  repository is a possible follow-up.
- **Data provenance**: we take the Zenodo dataset at face value. Independent
  re-annotation of a sample would be a separate study.

## FIESTA-OSCARS context

This reproduction establishes the CNN baseline against which the
[fiesta-scattering-bio](https://github.com/annefou/fiesta-scattering-bio)
repository's scattering-transform approach is compared. The FIESTA-OSCARS
project demonstrates FAIR image analysis across astrophysics, environmental
sciences, and biodiversity.

## FORRT nanopublication chain

The full provenance of this reproduction is recorded as a six-step FORRT
nanopublication chain on the
[Science Live](https://platform.sciencelive4all.org) platform — paper → quote
→ atomic claim → FORRT claim → study → outcome → CiTO citation back to the
paper. Each step is independently citable and machine-readable.

> **Headline assertion — machine-readable:**
> [**This reproduction `cito:confirms` Decrop et al. 2025**](https://w3id.org/sciencelive/np/RArFs1V-cAEw19zHbMFXmTr-R34PlGdMHCOt6EQeW64jU)
>
> Discovery tools (Scholia, Wikidata pipelines, SPARQL endpoints) can follow
> this single citation to find that the paper's headline claim has been
> independently verified — the verifier is this repository.

The five preceding nanopubs build the provenance ladder up to that citation:

| Step | Type | Nanopub URI |
|---|---|---|
| 1 | Quote-with-comment | <https://w3id.org/sciencelive/np/RAm0R_xdfpZCX8jGhrMU8LQonht0X6qvfApCekmUCwvgQ> |
| 2 | AIDA sentence | <https://w3id.org/sciencelive/np/RAlNKRixcGMJ6-pz1AWMhtMv4JChJEock-rBRpEKUroas> |
| 3 | FORRT Claim (model performance) | <https://w3id.org/sciencelive/np/RAQbvusYubgaYlU7YEPIgPmTwqDoylwiq5FKyrVgF95qM> |
| 4 | FORRT Reproduction Study | <https://w3id.org/sciencelive/np/RAGtxXgvYl-b7NOkyS3K34z1yDDkPqzGuiTr54nA2uH7U> |
| 5 | FORRT Replication Outcome (Validated, VeryHigh) | <https://w3id.org/sciencelive/np/RAn_DYEINS9hSmULEWNd8JmdPOykzptJPJCSqiatNLBrA> |
| 6 | **CiTO `confirms` Decrop 2025** | **<https://w3id.org/sciencelive/np/RArFs1V-cAEw19zHbMFXmTr-R34PlGdMHCOt6EQeW64jU>** |

## How to reproduce

See the [notebook](01_reproduce_decrop.ipynb) for the full executable pipeline.
Approximately 700 MB of data + 47 MB of model weights are downloaded on first
run; inference takes about 27 minutes on a single CPU core pool.
