# ARCH: AI-driven Renal Clustering & Histopathology model

> 🏆 **2nd Place Overall (Seoul National University Hospital Director's Award) at the Medical AI Challenge (MAIC)**
> 🥇 **1st Place on the Internal Leaderboard**
>
> This repository contains the official codebase and research pipeline for **Team Kidneyden's** submission to the MAIC. Our model, **ARCH**, was recognized for its novel approach to prognostic assessment in renal pathology, achieving the highest performance in the internal evaluation.
>
> **Internal Leaderboard (Top 5)**
> | Rank | Team | Score |
> | :---: | :--- | :---: |
> | **1st** | **Team Kidneyden** | **0.7818** |
> | 2nd | Team ---- | 0.7648 |
> | 3rd | Team ---- | 0.7219 |
> | 4th | Team ---- | 0.6083 |
> | 5th | Team ---- | 0.4253 |

## 📌 Project Overview
Chronic kidney disease (CKD) and IgA Nephropathy pose a significant global health burden. While renal biopsy remains essential for diagnosis, predicting long-term prognosis based solely on human visual assessment is challenging due to the spatial heterogeneity and complex morphological patterns of the tissue. 

**ARCH** leverages deep learning to provide objective, immediate prognostic insights from histology at the time of biopsy, enabling early and personalized clinical interventions.

## 🚀 Key Features & Novelty
* **Foundation Model Integration:** Utilizes GigaPath (pretrained on ~1.3 billion pathology images) to extract robust, high-fidelity representations from complex histological patterns, achieving ~128x WSI compression.
* **Latent Domain Clustering:** Identifies underlying dataset structures (e.g., scanner, stain, sectioning differences) in the embedding space. Applies cluster-based sampling to mitigate label noise and class imbalance during training.
* **Auxiliary Learning:** Employs a Multiple Instance Learning (TransMIL) encoder with auxiliary loss heads (Normal, Inflammation, Tumor, Necrosis) to encourage joint discrimination and improve downstream performance.

## 🧠 Methodology & Architecture

### 1. Data Preprocessing & ROI Grouping
* **WSI Downsampling:** WSIs are downsampled to 1.25x and partitioned, focusing only on the strip with the highest patch density to reduce redundancy.
* **Tissue Filtering:** The patch labeling pipeline segments ROIs (Glomerulus, Inflammation, IFTA, Medulla, Extrarenal). 
* **Noise Reduction:** Patches with `< 25%` tissue or `> 50%` extrarenal regions (which introduce staining/structural noise) are discarded.

### 2. M0/M1 Glomerulus Prediction
* Extracted 256x256 glomerulus images to link patch-level morphological findings (e.g., sclerosis, proliferation) to M0/M1 labels for slide-level decision-making.
* Explored various models (EfficientNet, ResNet18), with **DenseNet121 + Threshold Tuning** achieving the best performance.

### 3. Deep Learning Architecture (ARCH)

<img width="904" height="505" alt="ModelArchitecture" src="https://github.com/user-attachments/assets/f7c6f352-034f-43b6-892c-345b1e772e19" />

* **Feature Extraction (Frozen GigaPath):** WSI patches are categorized and passed through a frozen GigaPath model to extract rich, pathology-specific embeddings.
* **TransMIL Encoder:** The extracted features are aggregated using TransMIL to capture long-range contextual dependencies across the slide.
* **Auxiliary Loss Formulation:** Alongside the primary IgA prediction (Diagnosis Loss), the model utilizes patch-level auxiliary heads to compute an Auxiliary Loss, stabilizing the training process.

### 4. Pipeline Development Stack
* The data preprocessing, model training, and performance validation pipelines are primarily developed in **Python**, utilizing **PyTorch** for deep learning architecture implementation and **pandas** for structured clinical data and metric evaluation.

## 📊 Performance
The model was evaluated using a Stratified 5-Fold Cross-Validation strategy, balancing both the IgA ratio and Cluster ratio across folds to prevent overfitting.

* **AUROC Scores across folds:** `0.770` | `0.724` | `0.743` | `0.804` | `0.800`
* **M0/M1 Prediction (DenseNet121 + Threshold Tuning):** AUROC `0.878`, Macro F1 `0.780`

## 💡 Clinical Significance
1. **Objective Risk Stratification:** AI-based analysis of histologic features reduces inter-observer variability and improves consistency in prognostic assessment.
2. **Precision Medicine:** Integrates tissue-level risk prediction into standard care pathways, discovering novel prognostic patterns beyond conventional scoring systems.
3. **Scalability:** Automated analysis enables rapid, high-throughput risk assessment, even in settings with limited pathology expertise.

## 👥 Team 키드니든 (Kidneyden)
* **Younghoon Na** (SNU Biomedical Science)
  * Data Preprocessing, AI Training Pipeline Design, Validation & Performance Analysis
* **Yoongeol Lee** (SNU Biomedical Science)
  * Data Preprocessing, Model Development & Optimization, Validation & Performance Analysis
* **Hyun Keun Ahn** (SNUH Dermatology)
  * Problem Definition & Strategy, Presentation & Documentation, Clinical Insight & Interpretation
