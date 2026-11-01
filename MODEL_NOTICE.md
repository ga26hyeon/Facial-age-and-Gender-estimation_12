# Model and data notices

## Face attributes

The age and gender ONNX model is `face-lens` by `snakajima`, downloaded from https://huggingface.co/snakajima/face-lens at revision `d8e2c4cafb5e14a258202d8f5176973d25a244df`.

The model is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/ . The ONNX weights are used without modification. This app uses its existing OpenCV face detector box for the expanded crop instead of the model author's MediaPipe landmark crop. The model publisher's `ATTRIBUTION.csv` is included alongside the weights.

## Evaluation data

The 100 images under `evaluation/data/images/` are from the FairFace 0.25 validation split, published by Kimmo Karkkainen and Jungseock Joo under CC BY 4.0. Source: https://huggingface.co/datasets/HuggingFaceM4/FairFace and https://github.com/joojs/fairface .

Karkkainen, K. and Joo, J. “FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation.” WACV 2021, pp. 1548–1558.
