# Inter-IIT Tech Meet 15.0 Computer Vision Selection Bootcamp
## Comprehensive Reconnaissance, Data Audit, & Implementation Architecture Plan

---

## 1. Executive Summary & Problem Overview

The objective of this project is to tackle **Learning from Messy Human Labels** using the **CIFAR-10N** dataset (specifically the "worst" human label set with ~40% real crowd worker label noise). Real-world human label noise is systematic and asymmetric (e.g., confusing cats with dogs, automobiles with trucks), unlike synthetic uniform symmetric noise.

The goal is to develop a robust machine learning pipeline that:
1. Audits human label noise vs clean labels.
2. Identifies label memorization dynamics during standard training.
3. Ranks/scores training images by noise likelihood.
4. Trains robust classifiers despite 40% label noise.
5. Evaluates model performance on real vs synthetic noise.

---

## 2. Empirical Repository Audit & Dataset Statistics

All figures below have been computed directly from the project data files (`data/train_images.npy`, `data/test_images.npy`, `data/labels.csv`, `data/train_idx.txt`, `data/val_idx.txt`, `data/classes.txt`).

### A. Data Shapes, Types, and Index Audits
- **Training Image Set (`train_images.npy`)**:
  - Shape: `(50000, 32, 32, 3)`
  - Data Type: `uint8`
- **Test Image Set (`test_images.npy`)**:
  - Shape: `(10000, 32, 32, 3)`
  - Data Type: `uint8`
- **Training Subset Indices (`train_idx.txt`)**:
  - Count: `45,000` unique indices
  - Range: `0` to `49,999`
- **Validation Subset Indices (`val_idx.txt`)**:
  - Count: `5,000` unique indices
  - Range: `6` to `49,985`
- **Index Overlap**: `0` indices overlap between `train_idx` and `val_idx`.
- **Index Coverage**: `train_idx` + `val_idx` = `50,000` indices, perfectly covering all rows of `labels.csv` and `train_images.npy`.

---

### B. Label Distributions across Datasets & Subsets

| Class ID | Class Name | Clean (All 50k) | Clean (Train 45k) | Noisy (Train 45k) | Synthetic (Train 45k) | Clean (Val 5k) | Noisy (Val 5k) |
|---|---|---|---|---|---|---|---|
| **0** | airplane | 5,000 | 4,505 | 4,791 | 4,497 | 495 | 518 |
| **1** | automobile | 5,000 | 4,496 | 5,414 | 4,480 | 504 | 639 |
| **2** | bird | 5,000 | 4,504 | 4,788 | 4,602 | 496 | 554 |
| **3** | cat | 5,000 | 4,469 | 4,205 | 4,347 | 531 | 468 |
| **4** | deer | 5,000 | 4,513 | 3,636 | 4,516 | 487 | 404 |
| **5** | dog | 5,000 | 4,533 | 5,130 | 4,528 | 467 | 538 |
| **6** | frog | 5,000 | 4,503 | 3,770 | 4,559 | 497 | 433 |
| **7** | horse | 5,000 | 4,506 | 4,748 | 4,420 | 494 | 525 |
| **8** | ship | 5,000 | 4,496 | 4,160 | 4,499 | 504 | 469 |
| **9** | truck | 5,000 | 4,475 | 4,358 | 4,552 | 525 | 452 |
| **Total** | | **50,000** | **45,000** | **45,000** | **45,000** | **5,000** | **5,000** |

---

### C. Human & Synthetic Noise Rates
- **Human Noisy Label Noise Rate on Training (45,000)**:
  - Mislabeled Count: **18,109** / 45,000
  - Noise Rate: **40.2422%** (0.402422)
  - Correctly Labeled Count: **26,891** / 45,000 (59.7578%)
- **Human Noisy Label Noise Rate on Validation (5,000)**:
  - Mislabeled Count: **1,995** / 5,000
  - Noise Rate: **39.9000%** (0.399000)
- **Overall Human Noisy Label Noise Rate (50,000)**:
  - Mislabeled Count: **20,104** / 50,000
  - Noise Rate: **40.2080%** (0.402080)
- **Synthetic Noise Rate on Training (45,000)**:
  - Mislabeled Count: **18,044** / 45,000
  - Noise Rate: **40.0978%** (0.400978)

---

### D. 10x10 Confusion Matrix (Training 45k: Noisy Row vs Clean Column)

```
Clean    0(air) 1(auto) 2(bird)  3(cat) 4(deer)  5(dog) 6(frog) 7(hrs) 8(shp) 9(trk)
Noisy
0(air)   2925    191    263     180    184     150    170    179    369    180
1(auto)   293   2657    165     160    142     131    166    120    333   1247
2(bird)   314     63   2842     354    316     240    373    111     94     81
3(cat)     89    116    212    2168    184     834    350    100     65     87
4(deer)    94     80    254     194   2195     183    229    233     96     78
5(dog)    105     68    216     924    491    2620    257    289     79     81
6(frog)    85     60    222     206    174     123   2676    101     71     52
7(hrs)     75     64    147     106    683     136    124   3246     99     68
8(shp)    379     71    106      91     94      55     96     77   3076    115
9(trk)    146   1126     77      86     50      61     62     50    214   2486
```

---

### E. Top 10 Dominant Human Label Noise Confusions
1. **Noisy 1 (automobile) -> Clean 9 (truck)**: 1,247 samples (2.77% of training set)
2. **Noisy 9 (truck) -> Clean 1 (automobile)**: 1,126 samples (2.50% of training set)
3. **Noisy 5 (dog) -> Clean 3 (cat)**: 924 samples (2.05% of training set)
4. **Noisy 3 (cat) -> Clean 5 (dog)**: 834 samples (1.85% of training set)
5. **Noisy 7 (horse) -> Clean 4 (deer)**: 683 samples (1.52% of training set)
6. **Noisy 5 (dog) -> Clean 4 (deer)**: 491 samples (1.09% of training set)
7. **Noisy 8 (ship) -> Clean 0 (airplane)**: 379 samples (0.84% of training set)
8. **Noisy 2 (bird) -> Clean 6 (frog)**: 373 samples (0.83% of training set)
9. **Noisy 0 (airplane) -> Clean 8 (ship)**: 369 samples (0.82% of training set)
10. **Noisy 2 (bird) -> Clean 3 (cat)**: 354 samples (0.79% of training set)

---

## 3. Strict Rules & Data Leakage Constraints Audit

1. **Model Architecture**: MUST use exact `ResNet18` from `model.py` (3x3 stem, no maxpool, ~11.17M parameters). Trained from scratch with NO pretrained weights. `model.py` must NOT be modified.
2. **No External Data**: External datasets, pretrained models, or looking up test set labels are strictly forbidden.
3. **Clean Label Usage Restrictions**:
   - `clean_label` MAY BE USED ONLY FOR:
     1. Training the Task 1 reference model (`t1_clean_ref.csv`).
     2. Measuring diagnostics: Task 1 confusion matrix, Task 2 dynamics logging (`val_acc_clean`), and Task 3 internal validation/debugging.
   - `clean_label` MAY NOT BE USED FOR:
     - Training, filtering, relabeling, or selecting models in Tasks 2, 3, 4, or 5.
4. **Model Selection Rules**:
   - Tasks 1–4 model selection MUST use `noisy_label` on the 5,000 validation images (`val_idx.txt`).
   - Task 5 model selection MUST use `synthetic_label` on the 5,000 validation images (`val_idx.txt`).
5. **Budget Limits**:
   - Maximum **60 epochs** per network.
   - Maximum **2 networks** per method (e.g. Co-Teaching).
   - Maximum **5 Task-3 training runs** (e.g. cross-validation or loss logging runs).
   - Standard hyperparameter consistency: Optimizer, schedule, batch size, and augmentations must remain identical across baselines and methods unless the method specifically requires a modification.

---

## 4. Proposed Directory Structure & Modular Architecture

```
/Users/shouryasaxena/Desktop/Inter-IIT Tech Meet 15.0 Computer Vision Selection Bootcamp-20261005T160338Z-1-001/
├── CV Problem Statement.docx
├── PROJECT_PLAN.md                     # This comprehensive master plan
├── check_format.py                     # Submission checker script
├── model.py                            # Frozen ResNet-18 model architecture
├── results_template.json               # Template for results
├── data/
│   ├── classes.txt
│   ├── labels.csv
│   ├── test_images.npy
│   ├── train_idx.txt
│   ├── train_images.npy
│   └── val_idx.txt
├── src/                                # Core modular Python package
│   ├── __init__.py
│   ├── config.py                       # Global hyperparams & paths config
│   ├── dataset.py                      # PyTorch Dataset and augmentations
│   ├── utils.py                        # Seed setting, device management, logger
│   ├── metrics.py                      # AUROC, precision@k, accuracy counters
│   ├── trainer.py                      # Generalized training & validation loops
│   ├── task1_2_baselines.py            # Task 1 & Task 2 execution script
│   ├── task3_noise_scorer.py           # Task 3 noise score calculation
│   ├── task4_robust_trainer.py         # Task 4 robust methods (Co-Teaching, GCE, Filtered)
│   ├── task5_synthetic_runner.py       # Task 5 synthetic noise training
│   └── export_submission.py            # Generates predictions/, results.json, report.md
├── predictions/                        # Output directory for submission
│   ├── t1_clean_ref.csv
│   ├── t1_noisy_ce.csv
│   ├── t2_dynamics.csv
│   ├── t2_early_stop.csv
│   ├── t3_noise_scores.csv
│   ├── t4_coteaching.csv               # Task 4 candidate method 1
│   ├── t4_gce.csv                      # Task 4 candidate method 2
│   ├── t4_filtered_retrain.csv         # Task 4 candidate method 3
│   ├── t5_baseline_synth.csv
│   └── t5_best_synth.csv
├── code/                               # Clean code submission directory
│   ├── README.md                       # Reproduction steps
│   ├── run_all.py                      # One-click execution entry point
│   └── (copies of src/ scripts)
├── results.json                        # Formatted submission results file
└── report.md                           # Concise Q1-Q5 answers (<=100 words each)
```

---

## 5. Planned Modules & Technical Blueprint

1. **`src/config.py`**:
   - Defines standard hyperparameters:
     - Epochs: 60
     - Batch Size: 128
     - Optimizer: SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)
     - Scheduler: CosineAnnealingLR (T_max=60, eta_min=1e-3)
     - Augmentation: RandomCrop(32, padding=4, padding_mode='reflect'), RandomHorizontalFlip(), CIFAR-10 Normalize (Mean: [0.4914, 0.4822, 0.4465], Std: [0.2023, 0.1994, 0.2010]).
     - Devices: MPS if `torch.backends.mps.is_available()` else CPU.

2. **`src/dataset.py`**:
   - `CIFAR10NDataset`: Custom PyTorch Dataset reading images from `train_images.npy` or `test_images.npy` and labels from `labels.csv`.
   - Supports returns of `(index, image, label, clean_label)` for dynamics tracking.

3. **`src/task1_2_baselines.py`**:
   - **Task 1 Clean Reference Model**: Trained on `train_idx` with `clean_label` for 60 epochs. Saves `t1_clean_ref.csv`.
   - **Task 1 Noisy CE Baseline Model**: Trained on `train_idx` with `noisy_label` for 60 epochs. Saves `t1_noisy_ce.csv`.
   - **Task 2 Dynamics Tracking**: During Noisy CE training, computes every epoch:
     - `train_acc_correct`: Accuracy on 26,891 correctly labeled training samples.
     - `train_acc_mislabeled`: Accuracy on 18,109 mislabeled training samples.
     - `val_acc_noisy`: Accuracy on 5,000 validation samples (`noisy_label`).
     - `val_acc_clean`: Accuracy on 5,000 validation samples (`clean_label`).
   - Exports `t2_dynamics.csv`.
   - Selects early stopping epoch based on max `val_acc_noisy` and exports `t2_early_stop.csv`.

4. **`src/task3_noise_scorer.py`**:
   - Assigns a noise score in `[0, 1]` or continuous loss for all 45,000 training images.
   - Signal Strategy:
     - Multi-epoch Cross-Entropy loss logging + out-of-fold cross-validation prediction error.
     - Samples with high average loss across early-to-mid training epochs are far more likely to be mislabeled.
   - Evaluates score quality offline using AUROC & Precision@18109 against true clean labels.
   - Exports `t3_noise_scores.csv` (`index,score`).

5. **`src/task4_robust_trainer.py`**:
   - Implements 3 robust learning strategies (within max 2 networks & 60 epochs budget):
     1. **Co-Teaching**: Two dual ResNet-18 networks training simultaneously; each network selects samples with small loss to train the other network with forgetting rate schedule $R(t) = \min(\text{epoch}/\text{warmup} \times 0.40, 0.40)$.
     2. **Generalized Cross Entropy (GCE)**: Robust loss function $L_{q}(f(x), y) = \frac{1 - f_y(x)^q}{q}$ with $q=0.7$.
     3. **Noise Score Filtering & Retraining**: Filter out top 35-40% highest noise scores from Task 3, retrain ResNet-18 on clean-filtered subset.
   - Evaluates all on validation set using `noisy_label`.
   - Exports predictions to `t4_coteaching.csv`, `t4_gce.csv`, `t4_filtered_retrain.csv`.
   - Identifies `best_method` by highest noisy validation accuracy.

6. **`src/task5_synthetic_runner.py`**:
   - Retrains standard Cross-Entropy model on `synthetic_label` column (`t5_baseline_synth.csv`).
   - Retrains best Task 4 method on `synthetic_label` column (`t5_best_synth.csv`).
   - Evaluates using `synthetic_label` on validation set (`val_idx`).

7. **`src/export_submission.py`**:
   - Populates `results.json` matching `results_template.json`.
   - Generates `report.md` with concise (<100 words) responses to Q1-Q5.
   - Runs `check_format.py` verification to ensure 100% submission validity.

---

## 6. Training Configuration Proposal for Apple Silicon M1 (MPS)

- **Device Selection**: PyTorch `mps` (`torch.device("mps")`).
- **Batch Size**: 128 (optimal balance for M1 unified memory bandwidth and SGD convergence).
- **Optimizer**: `torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)`
- **Learning Rate Schedule**: `CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-3)`
- **Reproducibility & Seed Handling**:
  - `torch.manual_seed(42)`
  - `np.random.seed(42)`
  - `random.seed(42)`
  - `torch.mps.manual_seed(42)` (if available)

---

## 7. Experiment Budget Accounting

| Task | Run Description | # Networks | Epochs / Net | Total Epochs | Compliance Check |
|---|---|---|---|---|---|
| **Task 1** | Clean Ref Model + Noisy CE Model | 2 | 60 | 120 | Max 60 ep/net. Fully Compliant. |
| **Task 2** | Dynamics Logging & Early Stopping | 0 (Reuses Task 1 Noisy CE) | 0 | 0 | No extra training needed. |
| **Task 3** | Out-of-fold / Loss Tracking scoring | 5 | 60 | 300 | Max 5 training runs. Fully Compliant. |
| **Task 4** | Method 1: Co-Teaching | 2 | 60 | 120 | Max 2 nets per method. Fully Compliant. |
| **Task 4** | Method 2: Generalized Cross Entropy | 1 | 60 | 60 | Max 60 ep/net. Fully Compliant. |
| **Task 4** | Method 3: Filtered Subset Retrain | 1 | 60 | 60 | Max 60 ep/net. Fully Compliant. |
| **Task 5** | Synth Baseline + Synth Best Method | 3 | 60 | 180 | Max 60 ep/net. Fully Compliant. |
| **Total** | | **14** | | **840** | **100% within all Problem Statement constraints.** |

---

## 8. Planned Next Implementation Steps

1. Create modular source directory structure `src/` (`config.py`, `dataset.py`, `utils.py`, `metrics.py`, `trainer.py`).
2. Implement `src/task1_2_baselines.py` to run Task 1 (Clean Ref & Noisy CE) and log Task 2 dynamics (`t1_clean_ref.csv`, `t1_noisy_ce.csv`, `t2_dynamics.csv`, `t2_early_stop.csv`).
3. Execute Task 3 noise scoring script (`src/task3_noise_scorer.py`).
4. Execute Task 4 robust training methods (`src/task4_robust_trainer.py`).
5. Execute Task 5 synthetic noise benchmarking (`src/task5_synthetic_runner.py`).
6. Generate `results.json`, write `report.md`, populate `code/README.md`, and validate with `check_format.py`.
