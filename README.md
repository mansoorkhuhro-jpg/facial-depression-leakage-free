# Deep Facial Features for Depression Detection: A Leakage-Free Evaluation

Code accompanying the paper *"Deep Facial Features for Depression Detection: A Leakage-Free
Evaluation"*. The notebooks compare Action Units (AUs) and three released CNN face embeddings
(DenseNet-201, ResNet, VGG-16) across four model families (averaged functionals + logistic
regression, a context-free frame-pool network, a BiGRU and a dilated TCN) on **DAIC-WOZ** and
**E-DAIC**, under one leakage-safe protocol.

## Data

**No data are included.** DAIC-WOZ and E-DAIC are distributed by the USC Institute for Creative
Technologies under an End-User License Agreement (https://dcapswoz.ict.usc.edu). Every person
who uses the data must sign their own EULA. Do not commit raw data, extracted features,
caches or the `outputs/` folder to any public location.

Set the two paths at the top of each notebook:

```python
DAIC_ROOT  = r"/path/to/DAIC-WOZ"
EDAIC_ROOT = r"/path/to/E-DAIC"
```

## Notebooks (run in order, from one folder)

| Notebook | What it does |
|---|---|
| `01_prepare_features.ipynb` | Builds subject-level feature tables and labels from both corpora (handles E-DAIC's nested folder layout, removes duplicate participants). |
| `02_feature_analysis.ipynb` | Cue correlations (FDR), leave-one-subject-out evaluation, permutation test, official split, cross-corpus transfer, grouped importance. |
| `03_temporal_models.ipynb` | Per-second sequences; functionals / frame-pool / BiGRU / TCN on AU and CNN streams; repeated cross-validation, paired bootstrap, official split, attention profiles. Set `CNN_FEATURE` to `"densenet201"`, `"vgg16"` or `"resnet"`. |
| `04_final_checks.ipynb` | Reads saved predictions: run inventory and interview-type (Wizard-of-Oz vs autonomous agent) analysis. |

## Evaluation protocol

- Splits are by participant; no person appears in both training and test.
- Scaling, PCA and hyper-parameter tuning are fitted inside training folds only.
- Repeated stratified cross-validation (5 folds × 3 repeats × 3 seeds) and the official E-DAIC
  train → dev → test split.
- 95% bootstrap confidence intervals, paired bootstrap ΔAUC against the AU baseline, Holm
  correction, permutation tests, trivial baselines, and a gender positive control.
- All seeds are fixed; results are reproducible to three decimals.

## Requirements

Python ≥ 3.10. `pip install -r requirements.txt` (install the PyTorch build that matches your
hardware). A GPU speeds up Notebook 03 but is not required.

## Citation

If you use this code, please cite the paper (citation to be added on publication) and the
datasets, as required by their licence:

- Gratch J, et al. The Distress Analysis Interview Corpus of human and computer interviews. LREC 2014, 3123–3128.
- Ringeval F, et al. AVEC 2019 workshop and challenge: State-of-mind, detecting depression with AI, and cross-cultural affect recognition. AVEC 2019, 3–12.
- DeVault D, et al. SimSensei Kiosk: A virtual human interviewer for healthcare decision support. AAMAS 2014, 1061–1068.

## Intended use

Research code only. The models are not diagnostic tools and must not be used to assess any
individual's mental health.

## License

MIT (see `LICENSE`).
