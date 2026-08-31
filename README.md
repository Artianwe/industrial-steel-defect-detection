# Industrial Steel Surface Defect Detection

## Project Overview

This project develops a semantic segmentation system for detecting and
localizing surface defects in industrial steel images.

The project uses a ResNet-18 encoder with a U-Net-style decoder and
evaluates a defect-aware image sampling strategy designed to improve
learning of rare and small defect classes.

## Problem

The baseline model achieved a strong overall score but showed severe
class imbalance behavior: it was highly effective at background
segmentation while missing most defect pixels, particularly for rare
classes.

The diagnostic evaluation showed:

-   Baseline defect mean IoU: **0.43107**
-   Baseline mean IoU: **0.53870**
-   Class 1 IoU: **0.27414**
-   Class 2 IoU: **0.35474**
-   Class 3 IoU: **0.55595**
-   Class 4 IoU: **0.53944**

This motivated a targeted sampling experiment rather than changing the
model architecture.

## Method

### Preprocessing

-   Images are resized to **256 × 1024**.
-   Images use bilinear interpolation.
-   Segmentation masks use nearest-neighbor interpolation.
-   Images are normalized with ImageNet mean and standard deviation.
-   RLE annotations are decoded into a five-class mask:
    -   0: Background
    -   1--4: Defect classes

### Model

The segmentation network uses:

-   ResNet-18 encoder
-   U-Net-style decoder
-   Skip connections
-   Five output classes

### Defect-aware sampling

The final experiment changes the probability with which training images
are selected.

Sampling weights emphasize:

1.  Images containing Class 1 defects
2.  Images containing Class 2 defects
3.  Images containing small total defect areas

Weights are capped to avoid excessive sampling bias.

Importantly, this strategy does **not** duplicate image files. It
changes image selection probability during training.

## Experiments

### Baseline

The original model/checkpoint was independently verified on the same
validation set.

**Verified results:**

  Metric               Baseline
  ------------------ ----------
  Mean IoU              0.53870
  Mean Dice             0.67076
  Defect Mean IoU       0.43107
  Defect Mean Dice      0.59236

### Defect-aware sampling

The sampling experiment was trained for a maximum of 30 epochs with
early stopping.

Training stopped at epoch 19 after five consecutive epochs without
improvement.

The best checkpoint was epoch 14.

**Final verified results:**

  Metric                     Final
  ------------------ -------------
  Mean IoU             **0.56248**
  Mean Dice            **0.69555**
  Defect Mean IoU      **0.46076**
  Defect Mean Dice     **0.62333**

### Per-class final results

  Class              IoU      Dice   Precision    Recall
  ------------ --------- --------- ----------- ---------
  Background     0.96937   0.98445     0.97796   0.99102
  Defect 1       0.31089   0.47432     0.56444   0.40902
  Defect 2       0.40933   0.58089     0.61057   0.55396
  Defect 3       0.55105   0.71055     0.79577   0.64183
  Defect 4       0.57175   0.72754     0.86881   0.62578

## Baseline vs Final

  Metric               Baseline     Final         Change
  ------------------ ---------- --------- --------------
  Mean IoU              0.53870   0.56248       +0.02378
  Defect Mean IoU       0.43107   0.46076   **+0.02969**
  Mean Dice             0.67076   0.69555       +0.02479
  Defect Mean Dice      0.59236   0.62333   **+0.03097**

The defect mean IoU increased by approximately **6.9% relative to the
baseline**.

## Qualitative evaluation

Representative validation examples were evaluated using:

**Original image → Ground truth → Baseline prediction → Final sampling
prediction**

The final model demonstrated substantial improvements on difficult/rare
defects. Representative example-level IoU changes included:

-   Class 1: 0.000 → 0.664
-   Class 1: 0.000 → 0.708
-   Class 2: 0.000 → 0.462
-   Class 4: 0.000 → 0.785
-   Class 3: 0.166 → 0.264

These examples support the quantitative finding that the sampling
strategy improves defect detection rather than simply optimizing
background segmentation.

## Final conclusion

The completed project demonstrates a practical approach to severe class
imbalance in industrial defect segmentation.

Rather than repeatedly changing the network architecture, the project:

1.  established a baseline,
2.  diagnosed its failure mode,
3.  introduced a targeted defect-aware sampling strategy,
4.  trained and independently verified the resulting model,
5.  compared quantitative metrics, confusion matrices, and qualitative
    predictions.

The final model improves both overall segmentation and, more
importantly, defect-focused performance.

## Project status

**COMPLETE**

The trained checkpoints, verification outputs, and qualitative
comparison results should be retained as the final project artifacts.

## Suggested portfolio description

> Developed an industrial steel surface defect segmentation pipeline
> using a ResNet-18 U-Net architecture. Diagnosed severe class-imbalance
> and rare-defect under-detection in the baseline model, then designed a
> defect-aware sampling strategy emphasizing rare and small defects.
> Independently verified the final model and improved defect mIoU from
> 0.431 to 0.461 while improving overall mIoU from 0.539 to 0.562.
