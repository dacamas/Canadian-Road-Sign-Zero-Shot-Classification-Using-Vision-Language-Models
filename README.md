Canadian Road Sign Zero-Shot Classification with Vision-Language Models
Generated automatically on 2026-08-08T15:07:05.706809+00:00 from a single execution of this notebook. Every number below was measured at runtime.

Problem
Can a vision-language model recognise Canadian road signs it was never supervised on? Road-sign recognition is normally solved with a supervised classifier over a closed, fixed label set; adding a new sign type means new labelled data and retraining. Zero-shot classification promises to add a category by writing a sentence. This project measures how far that promise holds for Canadian signage.

Dataset
Source tier actually used: 2 — Experiment B - GLOBAL (German) imagery under a Canadian-compatible taxonomy. This is NOT Canadian imagery and is never described as such.

842 usable images across 15 categories after validation (0 unreadable and 58 near-duplicate images removed; classes below 12 images dropped).
Median crop 35x34 px; class imbalance ratio 1.11.
Canadian images: 0 of 842.
Provenance, licence and access date for every source are in data_provenance.json. The Mapillary Traffic Sign Dataset was deliberately not used because it has no documented unauthenticated download path; that limitation is recorded rather than worked around.
Methodology
Zero-shot definition used here: target classes were excluded from all supervised development.

Development classes (10): bicycle_crossing, bumpy_road, children_crossing, curve_warning, no_passing, road_narrows, slippery_road, speed_limit, traffic_signal_ahead, yield_sign
Zero-shot classes (5): stop, pedestrian_crossing, road_work, wildlife_warning, no_entry
No zero-shot test image was used for prompt selection, model selection, threshold tuning, preprocessing choices or hyper-parameter tuning. The held-out list was fixed in the configuration cell before any data was inspected.
Splitting inside the development classes is group-aware (group = physical sign instance or capture track), so the same sign cannot appear in two splits. Assertions in the splitting cell enforce all of this and would abort the notebook if violated.
Models
clip (openai/clip-vit-base-patch32|default), openclip (ViT-B-32|laion2b_s34b_b79k), siglip (google/siglip-base-patch16-224|default), plus a supervised resnet18 transfer-learning baseline and random / majority-class baselines. All registered models passed their encoder self-test and are included.

Results
The pre-registered model — chosen on validation, before the held-out categories were touched — is siglip with the template:1 prompt strategy. Full ranking on the held-out categories (label space = the 5 unseen classes, chance = 20.0%):

openclip (template:3): top-1 65.5%, top-3 91.3%, macro F1 0.594
siglip (template:1): top-1 49.5%, top-3 91.6%, macro F1 0.468
clip (prompt_ensemble): top-1 45.1%, top-3 85.8%, macro F1 0.320
Prompt engineering, measured on validation only: averaged over models, the strongest of the three compared text strategies was prompt_ensemble (mean macro F1 0.244 vs 0.178 for the weakest). Supervised comparison on the development-test split, where the comparison is like-for-like: resnet18 reached 59.3% top-1 against 29.7% for the best zero-shot model. On the held-out categories the supervised model scores 0.0% — not because it is weak, but because those categories do not exist in its output layer. That asymmetry is the entire point: the two approaches answer different questions.

Error Analysis
The most frequent confusion was no_entry -> stop (48 errors). Accuracy also varies with crop size, which is the closest available proxy for viewing distance and image quality; the breakdown is in error_analysis.csv and the confident-failure gallery is in figures/error_gallery.png.

Robustness
Under controlled perturbations of the zero-shot test images, the largest degradation came from gaussian_blur at severity 0.5: top-1 fell from 51.3% to 20.0% (-0.313). Weather, lighting and occlusion attributes are not present in the source metadata and are therefore not reported as such.

Calibration
Mean gap between softmax score and empirical accuracy across models: +0.220 (positive = overconfident). Expected calibration error per model is in calibration_summary.csv. These scores are similarity scores passed through a softmax, not calibrated probabilities, and should not be treated as such.

Limitations
Pre-training contamination. CLIP/OpenCLIP/SigLIP were trained on undisclosed web-scale corpora that certainly contain road-sign imagery. "Zero-shot" here means no supervision or model selection from the held-out classes in this pipeline, not that the encoders are naive about stop signs.
Geographic coverage. The imagery is German (GTSRB), not Canadian. Results speak to the sign concepts under a Canadian-compatible taxonomy, not to Canadian visual appearance.
Taxonomy differences. Canadian MUTCDC signage differs from Vienna-Convention signage in colour, shape and text. Source classes without a genuine Canadian counterpart were excluded rather than force-mapped; see taxonomy_exclusions.csv.
Dataset bias and scale. 275 zero-shot test images with an imbalance ratio of 1.11. Bootstrap intervals are reported precisely because point estimates at this scale are unstable.
Model bias. VLM performance is known to be uneven across geographies and languages; Québec's bilingual and French-only signage is a plausible failure mode that this dataset only lightly probes.
Licensing. Licences were read from the sources listed in data_provenance.json but not legally verified. license_verified is false everywhere for that reason.
Image quality. Crops are small; the sign is often a few dozen pixels across.
No real-time safety validation. No latency, adversarial, hardware-in-the-loop or regulatory testing was performed. This is not a driver-assistance component.
Conclusion
On 275 images from 5 categories that were never used for training, prompt selection, model selection or threshold tuning, the pre-registered model (siglip) reached 49.5% top-1 (95% CI [43.3%, 55.3%]) against a chance level of 20.0%, i.e. 2.47x chance, with macro F1 0.468. This is well above chance and supports the claim that modern VLMs transfer to unseen Canadian sign categories from text descriptions alone. The supervised baseline remains stronger on the categories it was trained on (59.3%), while being structurally incapable of naming a new category at all — which is precisely the trade-off this project set out to measure. Note that openclip scored higher on the held-out test (65.5% top-1). That is reported as an observation only: promoting it to the headline would be selecting a model on the test set, which this protocol exists to avoid.
