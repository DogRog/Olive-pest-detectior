# 🫒 Olive Pest Detector

A machine learning project for detecting and classifying olive tree pests and diseases using computer vision techniques. This project implements both classification and object detection approaches to identify common olive tree ailments.

## 🎯 Overview

This project addresses the critical need for automated detection of olive tree pests and diseases, which can significantly impact olive production. The system can identify six major categories of olive tree problems:

- **Anthracnose** - A fungal disease causing leaf spots and fruit rot
- **Black Scale** - Insect pest that weakens trees
- **Olive Peacock Spot** - Fungal disease affecting leaves
- **Psyllid** - Small insects that damage leaves and shoots
- **Russet Mite** - Microscopic pests causing leaf damage
- **Tuberculosis/Olive Knot** - Bacterial disease causing galls

## 📊 Performance

| Method | 1-shot | 2-shot | 3-shot | 5-shot | 10-shot |
|--------|--------|--------|--------|--------|---------|
| Prototypical Network | 67.1% | 74.5% | 78.8% | 83.2% | 87.5% |
| Matching Network | 68.3% | 72.7% | 77.1% | 81.8% | 86.1% |
| Fine-Tuning | 61.5% | 69.8% | 73.4% | 78.9% | 84.3% |

> Results using EfficientNetV2-S backbone

## 🛠️ Installation

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended)

### Setup

1. **Clone the repository**

   ```bash
   git clone [<repository-url>](https://github.com/DogRog/Olive-pest-detectior.git)
   cd "Olive Pest Detector"
   ```

2. **Install using uv (recommended)**

   ```bash
   pip install uv
   uv sync
   ```

3. **Or install using pip**

   ```bash
   pip install -e .
   ```

### Dependencies

The project uses modern ML libraries:

- **PyTorch & TorchVision** - Deep learning framework
- **TIMM** - Pre-trained vision models
- **Albumentations** - Image augmentations
- **scikit-learn** - ML
- **Matplotlib & Seaborn** - Visualization

### Dataset

- Original dataset: [Olive Tree Diseases Dataset](https://universe.roboflow.com/arina-fay/olive-tree-diseases) by Arina Fay on Roboflow Universe

## 📁 Project Structure

```text
olive-pest-detector/
├── code/
│   ├── experiments/           # Jupyter notebooks for experiments
│   │   ├── classification.ipynb
│   │   └── FSOD.ipynb
│   ├── analysis/             # Data analysis notebooks
│   │   ├── dataset_EDA.ipynb
│   │   └── results_viz.ipynb
│   └── data_preparation/     # Data preprocessing scripts
│       └── create_classification_dataset.py
├── data/
│   ├── classification_dataset/  # Organized by class folders
│   └── object_detection_dataset/ # COCO format dataset
├── results/
│   ├── classification/         # Classification results & plots
│   └── object_detection/      # Object detection results
└── pyproject.toml             # Project configuration
```

## 🔬 Usage

### 1. Data Preparation

Convert COCO format annotations to classification dataset:

```bash
python code/data_preparation/create_classification_dataset.py
```

### 2. Training & Evaluation

#### Classification Experiments

Open and run the classification notebook:

```bash
jupyter notebook code/experiments/classification.ipynb
```

The notebook includes:

- Few-shot learning with Prototypical Networks
- Matching Networks implementation
- Fine-tuning approaches
- Multiple backbone architectures
- Comprehensive evaluation metrics

#### Object Detection

Run FSOD (Few-Shot Object Detection) experiments:

```bash
jupyter notebook code/experiments/FSOD.ipynb
```

### 3. Analysis & Visualization

Explore the dataset and results:

```bash
jupyter notebook code/analysis/dataset_EDA.ipynb
jupyter notebook code/analysis/results_viz.ipynb
```

## 📈 Methodology

### Classification Approach

1. **Few-shot Learning**: Prototypical and Matching Networks for learning from few examples
2. **Transfer Learning**: Fine-tuning pre-trained models on olive pest data
3. **Multiple Backbones**: Comparison of EfficientNet, ResNet, and Vision Transformers

### Object Detection Approach

- **COCO Format**: Standard annotation format for precise localization
- **Few-Shot Object Detection**: Adapting detection models with minimal examples
- **Multiple Architectures**: Testing different detection frameworks

### Evaluation Metrics

- **Classification Accuracy**: Percentage of correctly classified images
- **Confidence Intervals**: Statistical significance of results
- **t-SNE Visualization**: Feature space analysis
- **mAP Scores**: Object detection performance metrics
