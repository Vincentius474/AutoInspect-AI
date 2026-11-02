# AutoInspect AI - Model Comparison

## Dataset
- Training images: 1,175
- Validation images: 248
- Test images: 208
- Classes: Low, Medium, High

## Experiment 1: Custom CNN
- Architecture: Four convolutional blocks
- Optimizer: AdamW
- Learning rate: 0.0005
- Batch size: 16
- Maximum epochs: 50

Validation metrics were not recorded in the available experiment results.

## Experiment 2: MobileNetV3-Small
- Architecture: MobileNetV3-Small
- Pretrained weights: ImageNet
- Optimizer: AdamW
- Classifier learning rate: 0.001
- Fine-tuning learning rate: 0.0001
- Batch size: 16
- Warmup epochs: 5
- Maximum fine-tuning epochs: 25

### Held-out test results

| Metric | Score |
|---|---:|
| Accuracy | 0.6394 |
| Balanced accuracy | 0.6396 |
| Macro precision | 0.6411 |
| Macro recall | 0.6396 |
| Macro F1 | 0.6400 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Low | 0.7031 | 0.6618 | 0.6818 | 68 |
| Medium | 0.5352 | 0.5429 | 0.5390 | 70 |
| High | 0.6849 | 0.7143 | 0.6993 | 70 |

Confusion matrix (rows are actual classes; columns are predicted classes):

| Actual \ Predicted | Low | Medium | High |
|---|---:|---:|---:|
| Low | 45 | 18 | 5 |
| Medium | 14 | 38 | 18 |
| High | 5 | 15 | 50 |

## Observations
- Medium damage was the hardest class for this checkpoint (test F1: 0.5390).
- A comparable Custom CNN result is not available, so the results do not
  establish which architecture performed better or whether transfer learning
  improved performance.
- The held-out test results alone are not sufficient to assess overfitting;
  comparable training and validation histories are needed.
