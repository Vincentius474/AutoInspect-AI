# AutoInspect AI - Model Comparison

## Dataset
- Training images: 1,175
- Validation images: 248
- Test images: 208
- Classes: Low, Medium, High

## Experiment 1: Custom CNN
- Architecture: Four convolutional blocks
- Optimizer: AdamW
- Learning rate: 0.001
- Batch size: 16
- Maximum epochs: 40

Validation accuracy:
Validation macro F1:

## Experiment 2: MobileNetV3-Small
- Architecture: MobileNetV3-Small
- Pretrained weights: ImageNet
- Optimizer: AdamW
- Classifier learning rate: 0.001
- Fine-tuning learning rate: 0.0001
- Batch size: 16
- Warmup epochs: 5
- Maximum fine-tuning epochs: 25

Validation accuracy:
Validation macro F1:

## Observations
- Which model performed better?
- Which classes were hardest to classify?
- Did transfer learning improve performance?
- Was there evidence of overfitting?
