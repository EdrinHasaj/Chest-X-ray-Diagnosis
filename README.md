Group Name: Disease Decoders

Team Member Names: Edrin Hasaj, Abdullah Siddiqui, Ibrahim Youssef, Haris Aljic

# The Machine Vision Problem we are Addressing:
Our goal is to employ a deep learning model to recognize patterns in chest X-ray images for the purpose of detecting thoracic diseases. This process involves identifying the presence of 14 thoracic diseases such as pneumonia, cardiomegaly, and more.

## Table of Contents

- [Dataset](#dataset)  
- [Dataset Challenges](#dataset-challenges)  
  - [Class Imbalance Overview](#class-imbalance-overview)  
  - [Multi-label Disease Co-occurence](#multi-label-disease-co-occurence)  
  - [Noisy Data](#noisy-data)  
- [Data Preprocessing](#data-preprocessing)  
  - [Patient-Level Splitting](#patient-level-splitting)  
  - [Transformations Applied](#transformations-applied)  
- [Model Architectures Explored](#model-architectures-explored)  
- [Singular Model Results](#singular-model-results)  
- [Gamma Correction Augmentation](#gamma-correction-augmentation)  
- [Ensemble Modeling](#ensemble-modeling)  
  - [Uniform Weighted Average](#1-uniform-weighted-average)  
  - [Differential Evolution + Forward Greedy (Ours)](#2-differential-evolution-de--forward-greedy-selection-novel)  
  - [Ensemble AUROC Results](#ensemble-auroc-results)  
  - [AUROC Comparison Chart](#auroc-comparison-chart)  
- [Model Interpretability with Grad-CAM](#model-interpretability-with-grad-cam)  
  - [Why Heatmaps?](#why-heatmaps)  
  - [Heatmap Visualizations (Lung Nodule Sample)](#heatmap-visualizations-lung-nodule-sample)  
  - [Insights from Grad-CAM](#insights-from-grad-cam)  
- [Limitations](#limitations)  
- [Scientific Poster](#scientific-poster)  
- [Individual Contributions](#individual-contributions)  
- [Notebook Overview](#notebook-overview)  
- [References](#references)



## Dataset

**Name**: ChestX-ray14 dataset

**Source**: The dataset was created by National Institute of Health (NIH) and can be downloaded from Kaggle (National Institutes of Health, 2018).

**Size**: 112,120 frontal-view chest X-rays from 30,805 unique patients

**Labels**: 14 disease conditions + 1 “No Finding” label 

![Chest X-ray Samples](https://github.com/EdrinHasaj/CSC490H5-Project/blob/main/figures/intro.png)


## Dataset Challenges
This project focuses on building a deep learning system to classify thoracic diseases from chest X-rays using the ChestX-ray14 dataset. We address challenges like:

### Class Imbalance Overview

| Disease Class           | Image Count | Dataset Share (%) |
|-------------------------|-------------|-------------------|
| No Finding              | 60,361      | 53.84%            |
| Infiltration            | 19,894      | 17.74%            |
| Effusion                | 13,317      | 11.88%            |
| Atelectasis             | 11,559      | 10.31%            |
| Nodule                  | 6,331       | 5.65%             |
| Mass                    | 5,782       | 5.16%             |
| Pneumothorax            | 5,302       | 4.73%             |
| Consolidation           | 4,667       | 4.16%             |
| Pleural Thickening      | 3,385       | 3.02%             |
| Cardiomegaly            | 2,776       | 2.48%             |
| Emphysema               | 2,516       | 2.24%             |
| Edema                   | 2,303       | 2.05%             |
| Fibrosis                | 1,686       | 1.50%             |
| Pneumonia               | 1,431       | 1.28%             |
| Hernia                  | 227         | 0.20%             |

>  The dataset is highly imbalanced. "No Finding" accounts for over **half** of all labels, while critical conditions like Hernia and Pneumonia occur in **less than 2%** of images.

### Multi-label Disease Co-occurence
![Chest X-ray Samples](https://github.com/EdrinHasaj/CSC490H5-Project/blob/main/figures/coocurrencematrix.png)

### Noisy Data
![Chest X-ray Samples](https://github.com/EdrinHasaj/CSC490H5-Project/blob/main/figures/noisysample.png)

## Data Preprocessing

To ensure high-quality input for model training, we applied a series of preprocessing steps and maintained patient-level separation during dataset splitting.

### Patient-Level Splitting

Many patients in the ChestX-ray14 dataset have **multiple X-ray images**. To prevent **data leakage** and overly optimistic performance metrics, we split the dataset by **unique Patient IDs** into:

- **70% Training**
- **10% Validation**
- **20% Testing**

This guarantees that images from the same patient never appear in both the training and evaluation sets.

---

### Transformations Applied

The following image preprocessing steps were applied to improve model robustness and performance:

- **Greyscale Conversion**  
  Converted single-channel grayscale images to 3-channel to match input requirements of most pretrained CNN architectures.

- **Random Horizontal Flip**  
  Introduced left-right variability to encourage spatial invariance.

- **Random Rotation (±15°)**  
  Helps the model generalize to slight changes in patient positioning.

- **Resizing to `224 × 224`**  
  Standardized input size for compatibility and computational efficiency.

- **Normalization**  
  Applied ImageNet mean and standard deviation to pixel values for better training convergence.

> All transformations were implemented using `torchvision.transforms` and were applied consistently across training and evaluation phases (except randomness during inference).

## Model Architectures Explored

- **CNN Models**: VGG19, DenseNet121  
  <br><sub><b>Figure 1:</b> DenseNet121 Architecture Diagram</sub>  
  ![DenseNet121](https://github.com/EdrinHasaj/CSC490H5-Project/blob/main/figures/D121.drawio_2.png)

- **Hybrid CNN + Transformer Models**: MaxViT, CoAtNet, ConvNeXt  
  <br><sub><b>Figure 2:</b> MaxViT Architecture Diagram</sub>  
  ![MaxViT](https://github.com/EdrinHasaj/CSC490H5-Project/blob/main/figures/MaxViT.drawio_5.png)

- **Pure Transformer**: Swin Transformer

---

## Singular Model Results


| Model        | Type                | Params (M) | Best AUROC | 4-Fold CV AUROC (±SD) |
|--------------|---------------------|------------|------------|------------------------|
| VGG19        | CNN                 | 143.7      | 0.8124     | 80.07 (±0.55%)         |
| DenseNet121  | Dense CNN           | 7.97       | 0.8316     | 82.89 (±0.16%)         |
| MaxViT       | CNN + Transformer   | 124.5      | 0.8385     | 83.60 (±0.20%)         |
| CoAtNet      | Hybrid Transformer  | 73.9       | **0.8411** | 83.26 (±9.21%)         |
| ConvNeXt     | Conv-inspired CNN   | 229.8      | 0.8359     | 83.44 (±0.07%)         |
| Swin         | Pure Transformer    | 228.6      | 0.8312     | 82.13 (±0.24%)         |


> All pretrained models were imported using the [`timm`](https://github.com/huggingface/pytorch-image-models) library.


## Gamma Correction Augmentation

To enhance image quality and boost classification performance, especially for underrepresented conditions, we applied **Gamma Correction** as a data augmentation technique.

Gamma Correction is a non-linear transformation that adjusts the brightness and contrast of an image using a tunable parameter, helping highlight subtle features that may otherwise be missed in noisy or low-contrast X-ray scans.

### Visual Example of Gamma Transformation

![Gamma Correction Example](https://github.com/EdrinHasaj/CSC490H5-Project/blob/main/figures/gammacorrect.png)

---

### Impact on Classification Performance (ConvNeXt Model)

| Condition              | Prevalence | AUROC (Non-Gamma → Gamma) |        
|------------------------|------------|----------------------------|
| No Finding             | 53.84%     | 78.92 → 78.85 (**-0.07**)  |
| **Infiltration**       | 17.74%     | 71.48 → **72.12 (+0.64)** |
| Effusion               | 11.88%     | 88.57 → 88.32 (**-0.25**)  |
| **Atelectasis**        | 10.31%     | 82.11 → **82.58 (+0.47)** |
| Nodule                 | 5.65%      | 79.31 → 77.94 (**-1.37**)  |
| **Mass**               | 5.16%      | 85.34 → **85.61 (+0.27)** |
| **Pneumothorax**       | 4.73%      | 86.88 → **87.25 (+0.37)** |
| **Consolidation**      | 4.16%      | 79.80 → **80.73 (+0.93)** |
| **Pleural Thickening** | 3.02%      | 81.77 → **82.41 (+0.64)** |
| Cardiomegaly           | 2.48%      | 90.09 → 89.39 (**-0.70**)  |
| Emphysema              | 2.24%      | 93.35 → 92.53 (**-0.82**)  |
| **Edema**              | 2.05%      | 89.89 → **90.38 (+0.49)** |
| **Fibrosis**           | 1.50%      | 82.29 → **82.79 (+0.50)** |
| **Pneumonia**          | 1.28%      | 74.84 → **76.61 (+1.77)** |
| **Hernia**             | 0.20%      | 89.26 → **90.38 (+1.12)** |
| **Overall AUROC**      | –          | 83.59 → **83.86 (+0.27)** |

---

**Gamma Correction proved highly effective**, especially for rare conditions like Pneumonia (+1.77%), Hernia (+1.12%), and Fibrosis (+0.50%).
>This increased overall AUROC in two models, ConvNext (+0.27) and MaxVit (+0.22)


## Ensemble Modeling

To enhance model performance and stability, we implemented ensembling strategies to combine predictions from multiple models. We evaluated all **57 non-singleton combinations** across six top-performing architectures:

- VGG19
- DenseNet121
- MaxViT
- CoAtNet
- ConvNeXt
- Swin Transformer

We explored two main ensemble approaches:

---

### 1. Uniform Weighted Average

This method assigns equal weight to each model in the ensemble:

`ŷᵢ = (1 / K) ∑ₖ pᵢₖ`

It’s simple, effective, and works well when models are relatively strong and diverse. However, it doesn’t differentiate based on individual model performance.

---

### 2. Differential Evolution (DE) + Forward Greedy Selection (Novel)

To further boost AUROC, we propose a **novel greedy-weighted ensemble**:

- **Forward Greedy Selection**: Iteratively adds the next best-performing model to the current ensemble.
- **Differential Evolution (DE)**: Optimizes the weights at each step to maximize the **mean AUROC** over all classes.

---

#### Pseudocode: Forward Greedy + DE Strategy

```python
def forward_greedy_de(models, val_preds, val_labels):
    selected_models = []
    remaining = list(models)
    best_score = 0.0
    best_weights = None

    while remaining:
        best_candidate = None
        candidate_score = best_score

        for model in remaining:
            current_ensemble = selected_models + [model]
            weights = differential_evolution(current_ensemble, val_preds, val_labels)
            ensemble_preds = weighted_sum(current_ensemble, weights, val_preds)
            score = compute_auroc(ensemble_preds, val_labels)

            if score > candidate_score:
                candidate_score = score
                best_candidate = model
                best_weights = weights

        if best_candidate:
            selected_models.append(best_candidate)
            remaining.remove(best_candidate)
            best_score = candidate_score
        else:
            break

    return selected_models, best_weights, best_score
```

### Results Summary

| **Ensemble Method**           | **AUROC** | **Competitive Best** | **Optimal Model Weights** |
|-------------------------------|-----------|------------------------|----------------------------|
| Uniform Weighted Average      | 0.8562    | 0.8532                 | MaxViT: 0.20, CoAtNet: 0.20, DenseNet121: 0.20, Swin: 0.20, ConvNeXt: 0.20, VGG19: 0.00 |
| DE + Forward Greedy (Ours)    | **0.8565**| **0.8543**             | MaxViT: 0.1663, ConvNeXt: 0.1877, DenseNet121: 0.2052, CoAtNet: 0.1524, Swin: 0.1747, VGG19: 0.1138 |

>  Our **greedy DE ensemble** slightly outperforms both the uniform average and competitive SynthEnsemble benchmark, while also offering interpretable weighting per model.

---

### AUROC Comparison Chart

![AUROC Comparison](https://github.com/EdrinHasaj/CSC490H5-Project/blob/main/figures/ensembleauroc.png)

*Visualizing AUROC scores of our best ensemble.*

>  The ensemble approach consistently improved overall classification performance by leveraging model diversity and adaptive weighting.

## Model Interpretability with Grad-CAM

To enhance interpretability and ensure clinical relevance, we implemented **Grad-CAM heatmaps** to visualize where each model focused its attention when predicting disease presence on chest X-rays.

Grad-CAM is widely used in medical imaging to provide visual explanations by highlighting important regions contributing to a model’s prediction. For instance, it's been used in past research to detect COVID-19 abnormalities in chest radiographs.

---

### Why Heatmaps?

- Helps validate that the model is learning **medically relevant features**
- Reveals attention **differences between models**
- Provides insight into **ensemble model performance**

We generated Grad-CAM heatmaps for **each individual model** and the **ensemble model** on a lung nodule example — a rare and subtle condition.

---

### Heatmap Visualizations (Lung Nodule Sample)

<table>
  <tr>
    <td align="center"><img src="https://github.com/EdrinHasaj/CSC490H5-Project/raw/main/figures/vggnodule.png" width="300"/><br><b>VGG</b></td>
    <td align="center"><img src="https://github.com/EdrinHasaj/CSC490H5-Project/raw/main/figures/coatnetnodule.png" width="300"/><br><b>CoAtNet</b></td>
    <td align="center"><img src="https://github.com/EdrinHasaj/CSC490H5-Project/raw/main/figures/convnextnodule.png" width="300"/><br><b>ConvNeXt</b></td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/EdrinHasaj/CSC490H5-Project/raw/main/figures/densnetnodule.png" width="300"/><br><b>DenseNet121</b></td>
    <td align="center"><img src="https://github.com/EdrinHasaj/CSC490H5-Project/raw/main/figures/swinnodule.png" width="300"/><br><b>Swin</b></td>
    <td align="center"><img src="https://github.com/EdrinHasaj/CSC490H5-Project/raw/main/figures/maxvitnodule.png" width="300"/><br><b>MaxViT</b></td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/EdrinHasaj/CSC490H5-Project/raw/main/figures/noduleensemble.png" width="300"/><br><b>Ensemble</b></td>
    <td align="center"><img src="https://github.com/EdrinHasaj/CSC490H5-Project/raw/main/figures/nodule.png" width="300"/><br><b>Original X-ray</b></td>
    <td align="center"><img src="https://github.com/EdrinHasaj/CSC490H5-Project/raw/main/figures/nodulereference.png" width="300"/><br><b>Reference</b></td>
  </tr>
</table>

---

### Insights from Grad-CAM

- **DenseNet121** and **MaxViT** showed strong central activation near the suspected nodule.
- **Swin** and **ConvNeXt** displayed more diffuse or scattered attention patterns.
- The **ensemble Grad-CAM**, a novel contribution, effectively fused these attention maps:
  - Highlighted relevant nodule areas more precisely
  - Filtered out irrelevant zones
  - Reduced individual model noise and bias

>  This fused visualization reflects how radiologists cross-reference cues and provides an interpretable justification for the ensemble's superior AUROC performance.

---

## Limitations

Despite our efforts, two key challenges remained unresolved. First, the issue of **class imbalance** proved difficult to fully mitigate. While our gamma correction augmentation improved performance on rare classes, traditional methods such as over-sampling, under-sampling, and class-weighted loss functions often resulted in degraded performance. Second, due to **limited computational resources**, we were unable to perform comprehensive hyperparameter tuning across all models. This deterred our ability to explore larger architectures, deeper ensembles, and fine-tuned optimization, which may have further boosted performance.

## Scientific Poster
![Poster](https://github.com/EdrinHasaj/CSC490H5-Project/blob/main/figures/CSC490_Final_Poster.png?raw=true)

## Individual Contributions

### Edrin Hasaj
- Researched and documented deep learning models and data augmentation techniques  
- Handled image resizing, preprocessing, and model implementation
- Reasearch, implemented and performed TTA for all singular models 
- Performed hyperparameter tuning and trained multiple model variants  
- Explored and implemented ensemble methods including Differential Evolution  
- Generated Grad-CAM heatmaps for individual models and the ensemble to support interpretability

### Abdullah Siddiqui
- Set up the model training and testing environment  
- Focused on singular model exploration, training, and evaluation
- Experimented with different libraries and singular model implementation
- Ran cross-validation experiments to assess generalization performance

### Ibrahim Youssef
- Researched benchmark scores from prior studies for comparison  
- Collaborated on advanced data preprocessing techniques with Haris  
- Assisted with hyperparameter tuning and running model training scripts

### Haris Aljic
- Applied gamma correction to all singular models and evaluated its impact  
- Built detailed tables comparing AUROC results across models and settings  
- Trained multiple models and co-developed the cross-validation script  
- Contributed to preprocessing strategies for improving robustness

## Notebook Overview

| **Folder / File**            | **Contents**                                                                 |
|-----------------------------|------------------------------------------------------------------------------|
| `data_visuals.ipynb`        | Notebook for data exploration and visualization.                            |
| `figures/`                  | Saved plots and images (e.g., Grad-CAM, AUROC curves).                      |
| `metrics/`                  | Metric comparisons across Normal, Gamma-Corrected, and TTA predictions.     |
| `model_cross_validation/`   | Contains 6 notebooks for cross-validating each individual model.            |
| `model_ensembling/`         | - `model_ensembling_de_greedy.ipynb`: Differential Evolution + Greedy ensemble strategy.  |
|                             | - `model_ensembling_weighted.ipynb`: Uniform weighted ensemble across models.             |
|                             | - `model_ensembling_heatmaps.ipynb`: Grad-CAM heatmap generation for ensemble outputs.    |
| `model_training/`           | - `training_1/`: Baseline training scripts.                                 |
|                             | - `training_2/`: Includes Gamma correction + CLAHE augmentation.            |
|                             | - `training_fastai/`: Training using fastai.                                |
|                             | - `training_gamma/`: Scripts for testing various gamma values.              |
| `tta_singular_models/`      | Notebook for running Test Time Augmentation (TTA) on each individual model. |



## References

1. **NIH ChestX-ray14 Dataset**  
   Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). *ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases*. IEEE CVPR.  
   [Link to Dataset on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)

2. **SynthEnsemble: Ensemble Learning for Chest X-ray Multi-label Classification**  
   Ashraf, H., Chen, Z., & Lin, H. (2023). *SynthEnsemble: An Empirical Study of Ensemble Learning for Chest X-ray Multi-label Classification*.  
