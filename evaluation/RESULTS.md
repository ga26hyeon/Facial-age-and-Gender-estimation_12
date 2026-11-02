# FairFace evaluation

## Data and terms

- Source: [`HuggingFaceM4/FairFace`, config `0.25`, validation split](https://huggingface.co/datasets/HuggingFaceM4/FairFace), dataset-server revision `54d573cdb8b5af490ba8da9da2799628f6e5c496`.
- The Hub lists the dataset under CC BY 4.0. FairFace describes its age, gender, and race labels and requests citation of Karkkainen & Joo (WACV 2021); see the [original project](https://github.com/joojs/fairface).
- An older full validation Parquet from earlier exploratory work was deleted before this split was prepared and was not used for these results. For this evaluation, the viewer filter API supplied the 30 tuning images. To avoid records already used in earlier exploratory inference, 100-row viewer slices were scanned through row 2,499 for labels/URLs; only 70 previously unseen final images were downloaded. Exactly 100 224×224 images (30 tune + 70 final) are stored under `evaluation/data/images/`; no full dataset or Parquet was downloaded for this split.
- The fixed sample is balanced across dataset gender labels (50 Male / 50 Female) and all 9 dataset age labels. Each gender×age cell contributes 5 or 6 rows. The deterministic row IDs and labels are in [`sample_manifest.csv`](sample_manifest.csv).
- Within each cell, 30 rows are assigned to `tune` (two rows in age classes 0–59, one row in the three oldest classes); the remaining 70 are assigned to `final`. The final cohort has 35 rows per gender and 6–8 rows per age group after mapping the dataset's nine age labels to the app's seven output groups. The final row IDs are disjoint from the tuning IDs and all earlier exploratory inference IDs. A provisional final sample was discarded after discovering overlap with earlier exploratory rows; its scores are not used here. No output from the current, disjoint final set was inspected to choose a model or preprocessing.

## Candidate and fixed protocol

The original app used the OpenCV age/gender Caffe models and returned eight age buckets. The candidate replaces just those two attribute heads with the pinned [`snakajima/face-lens`](https://huggingface.co/snakajima/face-lens) ONNX model at revision `d8e2c4cafb5e14a258202d8f5176973d25a244df`. The existing OpenCV face detector, FastAPI routes, upload handling, UI, and response fields remain. Model metadata specifies seven age groups, the female/male head order, RGB normalization, and the crop settings. Its card identifies CC BY 4.0 and says the age/gender heads were trained from FairFace human labels; model metadata/weights are pinned and the publisher's `ATTRIBUTION.csv` is retained in the Docker image.

Only two preprocessing variants were tried, both on the 30 tuning rows:

| Version | Crop sent to candidate model | Gender accuracy | Age accuracy | Age ordinal MAE |
| --- | --- | ---: | ---: | ---: |
| Existing service | Existing detector crop and Caffe preprocessing | 40.0% | 26.7% | 1.90 |
| Attempt A | Detector face box resized directly to 256×256 | 60.0% | 26.7% | 1.70 |
| Attempt B (selected) | Detector box expanded by metadata `scale=1.7`, shifted vertically by `shiftY=-0.12`; RGB mean/std normalization | 93.3% | 53.3% | 0.80 |

Attempt B was frozen before using the held-out final data. Both old and candidate models were then run once on the same 70 images with the same detector and confidence threshold (`0.55`). All 70 had a detected face.

## Held-out final results

| Metric | Existing service | Updated service | Change |
| --- | ---: | ---: | ---: |
| Gender accuracy | 45.7% (32/70) | 81.4% (57/70) | +35.7 percentage points |
| Age-group accuracy | 24.3% (17/70) | 58.6% (41/70) | +34.3 percentage points |
| Age-group ordinal MAE | 1.93 | 0.56 | 71.1% lower |
| Face detection coverage | 100% (70/70) | 100% (70/70) | unchanged |

Age target mapping is fixed as FairFace `0–2` and `3–9` → app `0–9`; `10–19` through `50–59` map directly; `60–69` and `more than 70` → app `60+`. The original model's eight output ranges were mapped to the nearest midpoint of these same ordered seven bins (`[0,0,1,1,2,4,5,6]`). Gender labels were mapped by each model's documented class order; this also corrects the old app's reversed Korean label mapping for the original OpenCV gender output.

Candidate age accuracy by target group was 75.0% (0–9, n=16), 50.0% (10–19, n=8), 50.0% (20–29, n=8), 37.5% (30–39, n=8), 33.3% (40–49, n=6), 25.0% (50–59, n=8), and 87.5% (60+, n=16). The middle-age cohorts remain the largest observed weakness.

## Limits

This is a 70-image held-out comparison, not evidence of population-wide accuracy. Rows are selected deterministically in dataset row order after removing previously used row IDs; sampling is not randomized within a cell. The result is therefore vulnerable to sample selection and has wide uncertainty. The model card reports aggregate results on the same FairFace validation split, so this final sample is isolated from our tuning and prior local inference but is not an independent external benchmark relative to the publisher's earlier evaluation/model selection. The age labels are coarse human annotations, and apparent-age estimates can be affected by image quality, pose, lighting, makeup, and demographic representation. This evaluation does not justify decisions about individuals or establish equal performance across race, age, or other subgroups.

## Bounded preprocessing update

The existing deployed candidate used the metadata crop scale 1.7 with `shiftY=-0.12`. Two additional preprocessing candidates were tested only on the fixed 30-row tuning split. Attempt 1 used scale 1.5 and scored 26/30 gender (86.7%), 17/30 age (56.7%), and age ordinal MAE 0.70. Attempt 2 used scale 1.9 and scored 21/30 gender (70.0%), 19/30 age (63.3%), and MAE 0.63. Attempt 2 was selected for the held-out check because age estimation is the main target of this update; the 70 final rows were not used to choose or adjust the candidate.

The selected scale 1.9 was evaluated once on the same fixed 70-row final split and compared with the already recorded current-model results (no baseline rerun). Gender accuracy changed from 57/70 (81.4%) to 56/70 (80.0%), age accuracy from 41/70 (58.6%) to 44/70 (62.9%), and age ordinal MAE from 0.557 to 0.543. The unweighted mean of gender and age exact accuracy increased from 70.0% to 71.4%. All 70 rows retained face detection. Thus the update improves age accuracy and the combined mean modestly while giving up one gender-correct prediction. The ONNX weights, detector, normalization, confidence threshold, API, and UI are unchanged; only the face crop scale changes to 1.9.

FairFace is listed as CC BY 4.0 on its [Hugging Face dataset card](https://huggingface.co/datasets/HuggingFaceM4/FairFace), and the original project requests citation of Karkkainen & Joo (WACV 2021) in its [project README](https://github.com/joojs/fairface/blob/master/README.md). The 100 images remain local-only under the ignored `evaluation/data/` directory.

The sample and manifest are for this evaluation only. They are not copied into the Docker image (`.dockerignore` excludes `evaluation/`).
