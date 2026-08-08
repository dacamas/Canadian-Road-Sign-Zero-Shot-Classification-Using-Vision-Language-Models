# Canadian Road Sign Zero-Shot Classification with Vision-Language Models

> Can a vision-language model recognise a road sign category it was **never supervised on**?

This project measures that under a **class-disjoint** protocol, comparing **CLIP**, **OpenCLIP** and
**SigLIP** against a supervised CNN and trivial baselines — with prompt selection, threshold tuning and
model selection all confined to categories that never appear in the final test.

Everything runs from a single Colab notebook: automated data acquisition, validation, taxonomy mapping,
class-disjoint splitting, evaluation, calibration, robustness testing, explainability and a Gradio demo.
**No manual uploads.** Every number below was produced by an actual run — `2026-08-08`, run mode `quick`,
seed `42`, device `cuda` (Tesla T4).

---

## Table of contents

- [The methodology that matters](#the-methodology-that-matters)
- [Results](#results)
- [Data and provenance](#data-and-provenance)
- [Running it](#running-it)
- [Repository layout](#repository-layout)
- [Design decisions worth knowing](#design-decisions-worth-knowing)
- [Limitations](#limitations)
- [Licence and attribution](#licence-and-attribution)

---

## The methodology that matters

Splitting images randomly and running CLIP on the test half **measures nothing** — CLIP never saw the
train half either. This project holds out **whole categories**:

| Partition | Categories | Used for |
| :--- | :---: | :--- |
| **Development** | 10 | supervised training, prompt selection, model selection, threshold tuning |
| **Zero-shot** | 5 | final evaluation only |

**Development classes (10)**
`bicycle_crossing` · `bumpy_road` · `children_crossing` · `curve_warning` · `no_passing` ·
`road_narrows` · `slippery_road` · `speed_limit` · `traffic_signal_ahead` · `yield_sign`

**Zero-shot classes (5)**
`stop` · `pedestrian_crossing` · `road_work` · `wildlife_warning` · `no_entry`

The hold-out list is declared in the configuration cell **before any data is inspected**. Splitting
inside the development classes is **group-aware** (one physical sign never spans two splits) **and
stratified** (every class appears in every split), and the notebook asserts both properties at runtime —
a violation aborts the run.

> **No zero-shot test image was used for** prompt selection · model selection · threshold tuning ·
> preprocessing choices · hyper-parameter tuning.

| Split | Images | Classes | Distinct sign instances |
| :--- | ---: | ---: | ---: |
| Train | 315 | 10 | 70 |
| Validation | 134 | 10 | 24 |
| Development test | 118 | 10 | 23 |
| **Zero-shot test** | **275** | **5** | **66** |

**Honest caveat.** VLM pre-training corpora are web-scale and undisclosed, and certainly contain
road-sign imagery. *Zero-shot* here means no supervision and no model-selection signal from the held-out
classes **in this pipeline** — not that the encoders have never seen a stop sign.

---

## Results

### Held-out categories (275 images, 5-way, chance = 0.200)

| Model | Training | Prompt strategy | Top-1 | Top-3 | Top-5 | Macro F1 |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| Random baseline | none | — | 0.200 | — | — | 0.194 |
| Majority class | none | — | 0.207 | — | — | 0.069 |
| **OpenCLIP** | zero-shot | `template:3` | **0.655** | **0.913** | 1.000 | **0.594** |
| **SigLIP** | zero-shot | `template:1` | 0.495 | 0.916 | 1.000 | 0.468 |
| **CLIP** | zero-shot | `prompt_ensemble` | 0.451 | 0.858 | 1.000 | 0.320 |
| ResNet18 | supervised | — | 0.000 | — | — | 0.000 |

**Pre-registered model: `siglip` / `template:1`** — 0.495 top-1, 95% CI **[0.433, 0.553]**, **2.47×
chance**.

> OpenCLIP scored higher (0.655), but it was **not** the validation-selected model. That is reported as
> a post-hoc observation, never promoted to the headline — selecting on the test set is exactly what
> this protocol exists to prevent.

### Zero-shot vs supervised

| Evaluated on | ResNet18 (supervised) | Best VLM (zero-shot) |
| :--- | ---: | ---: |
| Development test — *categories the CNN was trained on* | **0.593** | 0.297 |
| Zero-shot test — *categories held out entirely* | **0.000** | **0.495** |

The supervised model scores zero on held-out categories **by construction**: those labels do not exist
in its output layer, so it is forced to emit a development class. That asymmetry is the finding, not a
defect — the two approaches answer different questions.

### Statistical comparison

| Comparison | Δ Top-1 | 95% CI | McNemar *p* |
| :--- | ---: | :---: | ---: |
| CLIP vs OpenCLIP | −0.204 | [−0.266, −0.145] | < 0.001 |
| **CLIP vs SigLIP** | −0.044 | **[−0.109, +0.022]** | **0.230** |
| OpenCLIP vs SigLIP | +0.160 | [+0.095, +0.229] | < 0.001 |

CLIP and SigLIP are **not separable** at n = 275 — the interval contains zero. That is a limitation of
the test-set size, not evidence that the models are equivalent.

### Calibration — similarity scores are *not* probabilities

| Model | ECE | Mean softmax score | Accuracy | Overconfidence |
| :--- | ---: | ---: | ---: | ---: |
| CLIP | 0.342 | 0.793 | 0.451 | **+0.342** |
| SigLIP | 0.209 | 0.692 | 0.495 | +0.198 |
| OpenCLIP | **0.120** | 0.774 | 0.655 | +0.119 |

Every model reads as more confident than it is. Scores are cosine similarities passed through a softmax
and should never be presented as calibrated probabilities.

### Robustness (SigLIP, controlled perturbations)

| Condition | Top-1 | Δ |
| :--- | ---: | ---: |
| Clean | 0.513 | — |
| Brightness ↓ (0.5) | 0.520 | +0.007 |
| Contrast ↓ (1.0) | 0.373 | −0.140 |
| JPEG compression (1.0) | 0.240 | −0.273 |
| Occlusion (1.0) | 0.200 | −0.313 |
| Gaussian noise (1.0) | 0.200 | −0.313 |
| **Gaussian blur (0.5)** | **0.200** | **−0.313** |

Blur is the sharpest failure: even mild blur collapses performance to chance. Illumination changes are
tolerated well. Full grid in `outputs/robustness_results.csv`.

### Error analysis

Accuracy climbs **monotonically** with crop area — the closest available proxy for viewing distance:

| Crop area (px²) | Accuracy | n |
| :--- | ---: | ---: |
| 625 – 924 | 0.114 | 70 |
| 924 – 1,260 | 0.478 | 69 |
| 1,260 – 1,848 | 0.588 | 68 |
| 1,848 – 4,970 | **0.809** | 68 |

Dominant confusions: `no_entry → stop` (48), `wildlife_warning → stop` (29), `road_work → stop` (27) —
a red-circular-sign attractor. Per-class metrics in `outputs/per_class_metrics.csv`; a gallery of the
most *confident* failures in `outputs/figures/error_gallery.png`.

### Open-world rejection

Threshold selected on validation (0.0511), then frozen: **77.5%** acceptance rate, **61.5%** precision
among accepted predictions, 1.8% false rejection. A rejection option, **not** a claim of true open-set
recognition — that would need genuine out-of-distribution negatives.

---

## Data and provenance

| Source | Role | Status this run |
| :--- | :--- | :--- |
| Mapillary Traffic Sign Dataset | intended primary | **Not automatable** (account + licence form). Documented, not faked. |
| Mapillary Graph API v4 | genuine Canadian imagery | Unavailable — no `MAPILLARY_ACCESS_TOKEN` secret |
| GTSRB via Hugging Face | global fallback imagery | **Used** |
| Ville de Montréal sign catalogue | Canadian taxonomy grounding | Retrieved via CKAN API |

> **This run: Experiment B — global (German) imagery under a Canadian-compatible taxonomy.**
> This is **not** Canadian imagery and is never described as such anywhere in the outputs.

**842** usable images · **15** categories · **183** distinct physical sign instances · imbalance ratio
**1.11** · median crop **35 × 34 px** · Canadian images **0 of 842**.

GTSRB stores 30 consecutive frames of each physical sign, so ingestion **caps frames per track** —
otherwise a "900 image" dataset is really about 30 signs, and any group-aware split degenerates. Full
licensing and access dates in `outputs/data_provenance.json`.

---

## Running it

1. Open `canadian_road_sign_zero_shot.ipynb` in **Google Colab** and choose a **GPU** runtime
   (a T4 is plenty; CPU works, slower).
2. **Runtime → Run all.** Roughly 4 minutes in `quick` mode.
3. *Optional:* add a Colab secret `MAPILLARY_ACCESS_TOKEN` (free, from the Mapillary developer
   dashboard) to switch to **genuine Canadian imagery**.
4. Set `RUN_MODE = "full"` in the configuration cell for the larger evaluation.

Google Drive is mounted as a persistent cache when available. If the mount fails, the notebook falls
back to local storage and **outputs disappear when the runtime recycles** — download the bundle before
disconnecting.

---

## Repository layout

```text
canadian_road_sign_zero_shot.ipynb   the entire pipeline
outputs/
├── final_report.md                  auto-generated research report
├── dashboard.txt                    headline numbers
├── model_comparison.csv             the results table above
├── zero_shot_results.csv            per-model, per-label-space metrics
├── per_class_metrics.csv            precision / recall / F1 per category
├── predictions.csv                  every individual prediction
├── error_analysis.csv               confidence, correctness, crop size
├── calibration_summary.csv          ECE and overconfidence
├── robustness_results.csv           accuracy under perturbation
├── statistical_comparison.csv       bootstrap CIs, paired bootstrap, McNemar
├── prompt_selection_validation.csv  validation-only prompt search
├── taxonomy_mapping.csv             source label → canonical → Canadian category
├── taxonomy_exclusions.csv          every excluded class, with a reason
├── split_manifest.csv, splits/      exact split assignments
├── data_provenance.json             licence + access date per source
├── experiment_config.json           seeds, versions, checkpoints, frozen prompts
└── figures/                         confusion matrices, calibration, errors, explanations
```

---

## Design decisions worth knowing

**Class-disjoint, not image-disjoint.** The only split that makes the zero-shot claim testable. If the
category appears in development data, you can tune prompts until it works — it is no longer unseen,
only the pixels are.

**Excluded rather than force-mapped.** GTSRB classes with no Canadian counterpart (blue mandatory
arrows, priority road, derestriction signs) are dropped **with a documented reason** in
`taxonomy_exclusions.csv`, not bent into a Canadian label.

**Alignment is verified, not assumed.** Every model passes a behavioural image–text probe at load time.
A shape check alone lets a misaligned model report as a merely *weak* model — which is precisely how an
earlier version of this pipeline silently reported CLIP at chance.

**Occlusion sensitivity over SHAP.** A VLM's output is a similarity between two embeddings, not a scalar
over fixed features. Occlusion works identically across all three models, needs no gradients, and
answers the actual question: *which pixels support this text hypothesis?*

**Similarity ≠ probability.** Scores are reported as similarities, with calibration measured separately
rather than assumed.

---

## Limitations

1. **Not Canadian imagery.** This run used German GTSRB imagery. Results speak to sign *concepts* under
   a Canadian-compatible taxonomy — **not** to Canadian visual appearance.
2. **Pre-training contamination.** See the honest caveat above.
3. **Scale.** 275 zero-shot test images means unstable point estimates — hence confidence intervals
   everywhere.
4. **Taxonomy differences.** Canadian MUTCDC signage differs from Vienna-Convention signage in colour,
   shape and text.
5. **Licensing.** Licences were read from source documentation but **not legally verified**
   (`license_verified` is `false` throughout).
6. **Image quality.** Crops are small — median 35 × 34 px.
7. **No real-time safety validation.** No latency, adversarial, hardware-in-the-loop or regulatory
   testing was performed.

Full discussion in `outputs/final_report.md`.

---

## Licence and attribution

**Code:** MIT — see [`LICENSE`](LICENSE).

**Data is not redistributed.** The notebook fetches everything at runtime from the sources recorded in
`outputs/data_provenance.json`.

- **Mapillary** imagery is CC BY-SA 4.0 and requires attribution to Mapillary and the contributing user.
- **GTSRB** is subject to the original benchmark's terms — Stallkamp et al., Institut für
  Neuroinformatik, Ruhr-Universität Bochum.
- **Ville de Montréal** sign catalogue via the Données Québec / data.montreal.ca open-data portal.

---

## Disclaimer

**Research and portfolio demonstration only.** This is not a validated driver-assistance or
autonomous-driving component. No latency, adversarial, hardware-in-the-loop or regulatory testing was
performed, and the scores it reports are similarity scores, not calibrated probabilities.
