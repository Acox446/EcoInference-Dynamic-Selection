# EcoInference-Dynamic-Selection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> Dynamic model selection strategies for Green AI — optimizing the trade-off between inference accuracy and energy consumption.

## Overview

**EcoInference** implements two dynamic model selection strategies that intelligently route inference requests through a pool of models with varying complexity and energy costs. Instead of always using the largest, most accurate model, these strategies aim to use the *minimum necessary* computational resources to achieve reliable predictions.

This approach is inspired by the principles of **Green AI**, which advocates for developing machine learning systems that are environmentally sustainable without significantly compromising performance.

## Key Features

- **Cascading Strategy**: Sequential evaluation from lightweight to heavy models, stopping early when confidence is sufficient
- **Routing Strategy**: ML-based router that predicts the optimal model for each input in a single decision
- **Energy Monitoring**: Real-time energy consumption tracking using [CodeCarbon](https://github.com/mlco2/codecarbon)
- **Model Pool**: Pre-configured models ranging from Decision Trees to CNNs
- **Configurable**: YAML-based configuration for easy experimentation

## How It Works

### Model Pool

The system includes 5 models ordered by complexity and energy cost:

| Model | Type | Accuracy | Energy (J/sample) |
|-------|------|----------|-------------------|
| Tiny | Decision Tree | 69.5% | 0.000005 |
| Small | Logistic Regression | 84.0% | 0.000013 |
| Medium | Random Forest | 83.8% | 0.000006 |
| Large | MLP Neural Network | 87.0% | 0.000042 |
| Extra Large | CNN | 89.6% | 0.000142 |

### Cascading Strategy

```
Input → [Tiny] → confident? → YES → Return prediction
                    ↓ NO
              [Medium] → confident? → YES → Return prediction
                    ↓ NO
              [Large] → confident? → YES → Return prediction
                    ↓ NO
              [Extra Large] → Return prediction
```

The cascade evaluates models sequentially from lightest to heaviest. If a model's confidence exceeds its threshold, the prediction is returned immediately, saving energy by avoiding heavier models.

### Routing Strategy

```
Input → [Router] → decision → [Selected Model] → Return prediction
```

A lightweight router (Logistic Regression) is trained to predict which model will correctly classify each input using the minimum energy. The router learns from an "oracle" that knows the optimal assignment for each sample.

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/EcoInference-Dynamic-Selection.git
cd EcoInference-Dynamic-Selection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow codecarbon pyyaml
```

## Usage

### 1. Train and Benchmark Individual Models

First, train all models and measure their individual performance:

```bash
python scripts/main_benchmark.py
```

This will:
- Download Fashion-MNIST dataset
- Train all 5 models (Tiny → Extra Large)
- Measure accuracy and energy consumption
- Save models to `saved_models/`
- Generate `benchmark_results.csv` and `benchmark_plot.png`

### 2. Train the Router

Train the routing model on the validation set:

```bash
python scripts/train_router.py
```

### 3. Run Cascading Strategy

Evaluate the cascading strategy on the test set:

```bash
python scripts/run_cascading.py
```

### 4. Run Routing Strategy

Evaluate the routing strategy on the test set:

```bash
python scripts/run_routing.py
```

## Project Structure

```
EcoInference-Dynamic-Selection/
├── configs/
│   ├── experiments.yaml    # Strategy settings, thresholds, active models
│   └── models.yaml         # Model hyperparameters, energy costs
├── scripts/
│   ├── main_benchmark.py   # Train & benchmark all models
│   ├── train_router.py     # Train the routing classifier
│   ├── run_cascading.py    # Evaluate cascading strategy
│   └── run_routing.py      # Evaluate routing strategy
├── src/
│   ├── __init__.py
│   ├── base_model.py       # Abstract base class for models
│   ├── config.py           # Configuration loader
│   ├── data_loader.py      # Fashion-MNIST data loading
│   ├── energy.py           # Energy measurement (CodeCarbon wrapper)
│   ├── model_pool.py       # Model implementations
│   └── strategies/
│       ├── cascading.py    # Cascading strategy implementation
│       └── routing.py      # Routing strategy implementation
├── saved_models/           # Trained model checkpoints
├── logs/                   # Energy tracking logs
├── benchmark_results.csv   # Individual model benchmarks
└── README.md
```

## Configuration

### Experiments Configuration (`configs/experiments.yaml`)

```yaml
# Active models (order: light → heavy)
active_models:
  - "Tiny"
  - "Medium"
  - "Large"
  - "Extra"

# Cascading thresholds (must match active_models length)
cascading:
  thresholds: [0.9, 0.8, 0.7, 0.0]  # Last is 0.0 to ensure response
  presets:
    aggressive: [0.95, 0.90, 0.85, 0.0]
    balanced: [0.9, 0.8, 0.7, 0.0]
    conservative: [0.8, 0.7, 0.6, 0.0]
```

### Models Configuration (`configs/models.yaml`)

```yaml
# Energy costs in Joules per sample
energy_costs:
  Tiny: 0.00000539
  Small: 0.00001260
  Medium: 0.00000550
  Large: 0.00004192
  Extra: 0.00014224

# Model hyperparameters
model_params:
  tiny:
    max_depth: 5
  large:
    epochs: 5
    hidden_units: 128
  # ...
```

## Dataset

This implementation uses **Fashion-MNIST**, a dataset of Zalando's article images:
- 60,000 training images
- 10,000 test images
- 10 classes (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- 28×28 grayscale images

## Dependencies

- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `matplotlib` & `seaborn` - Visualization
- `scikit-learn` - Classical ML models
- `tensorflow` - Neural network models
- `codecarbon` - Energy consumption tracking
- `pyyaml` - Configuration parsing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


```
