# Iris ML Pipeline

A production-ready machine learning pipeline for training and inference on the Iris dataset using PyTorch. The project uses Docker for containerization with separate environments for training and inference.

## Project Structure

```python
.
├── core/                           # Core utilities and shared code
│   ├── dataset.py                  # Shared dataset object
│   ├── model.py                    # Shared model object
│   └── utils.py                    # Shared utility functions
├── data/                           # Data directory
│   ├── raw/                        # Raw data (iris.csv)
│   ├── processed/                  # Processed data (train.csv, inference.csv)
│   └── predictions/                # Model predictions (predictions.csv)
├── inference/                      # Inference module
│   ├── Dockerfile.prod             # Production Dockerfile for inference
│   └── inference.py                # Inference script
├── models/                         # Trained models
├── scripts/                        # Utility scripts
│   └── prepare_data.py             # Data preparation script
├── tests/                          # Unit tests
│   ├── test_data.py                # Data preparation tests
│   ├── test_training.py            # Training tests
│   ├── test_inference.py           # Inference tests
│   └── test_utils.py               # Utility tests
├── training/                       # Training module
│   ├── Dockerfile.dev              # Development Dockerfile for training
│   └── train.py                    # Training script
├── .gitignore                      # Git ignore file
├── Dockerfile.dev.dockerignore     # Git ignore file
├── Dockerfile.prod.dockerignore    # Git ignore file
├── pyproject.toml                  # Project configuration
├── requirements.txt                # Python dependencies
├── settings.json                   # Project settings and default variables
└── README.md                       # This file
```

## Features

- **Dockerized Pipeline**: Separate containers for training and inference
- **Automated Testing**: Comprehensive test suite with pytest
- **Data Processing**: Automated data preparation and splitting
- **PyTorch Model**: Neural network implementation for Iris classification
- **Production Ready**: Containerized deployment for both training and inference

## Prerequisites

- Docker
- NVIDIA Docker (optional, for GPU support)

## Getting Started

### 1. Training Pipeline

Build and run the training Docker container:

```bash
# Build the training image
docker build -f training/Dockerfile.dev -t iris-training .
```

Windows Powershell:

```powershell
# Run the training container.
# --gpus all activates cuda usage. Optional but suggested. Only works with nvidia gpus.
docker run --rm --gpus all `
  -v ${PWD}\data:/app/data `
  -v ${PWD}\models:/app/models `
  iris-training
######################
# or
######################
# Run the training container.
docker run --rm `
  -v ${PWD}\data:/app/data `
  -v ${PWD}\models:/app/models `
  iris-training
```

Linux / Mac:

```bash
# Run the training container.
# --gpus all activates cuda usage. Optional but suggested. Only works with nvidia gpus.
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  iris-training
######################
# or
######################
# Run the training container.
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  iris-training
```

The training pipeline will:

1. Install dependencies
2. Prepare the data using `scripts/prepare_data.py`
3. Run tests to validate the setup
4. Train the model using `training/train.py`
5. Save the trained model to `models/iris_model.pt`

### 2. Inference Pipeline

After training is complete, copy the trained model and run inference:

```bash
# Build the training image
docker build -f inference/Dockerfile.prod -t iris-inference .
```

Windows Powershell:

```powershell
# Run the training container.
# --gpus all activates cuda usage. Optional but suggested. Only works with nvidia gpus.
docker run --rm --gpus all `
  -v ${PWD}\data:/app/data `
  -v ${PWD}\models:/app/models `
  iris-inference
######################
# or
######################
# Run the training container.
docker run --rm `
  -v ${PWD}\data:/app/data `
  -v ${PWD}\models:/app/models `
  iris-inference
```

Linux / Mac:

```bash
# Run the training container.
# --gpus all activates cuda usage. Optional but suggested. Only works with nvidia gpus.
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  iris-inference
######################
# or
######################
# Run the training container.
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  iris-inference
```

The inference pipeline will:

1. Load the trained model
2. Run tests to validate the inference setup
3. Generate predictions on the inference dataset
4. Save predictions to `data/predictions/predictions.csv`

## Docker Images

### Training Image (`Dockerfile.dev`)

- **Base Image**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
- **Purpose**: Data preparation, model training, and validation
- **Components**: Core modules, training scripts, data preparation, tests

### Inference Image (`Dockerfile.prod`)

- **Base Image**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
- **Purpose**: Production inference on new data
- **Components**: Core modules, inference scripts, trained model, tests

## Development

### Local Setup

1. Create a virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

3. Prepare data:

    ```bash
    python -m scripts.prepare_data
    ```

4. Run tests:

    ```bash
    pytest tests/test_data.py tests/test_training.py tests/test_utils.py
    ```

5. Train the model:

    ```bash
    python -m training.train
    ```

6. Run tests:

    ```bash
    pytest tests/
    ```

7. Run inference:

    ```bash
    python -m inference.inference
    ```

## Testing

The project includes comprehensive tests for all components:

- `test_data.py`: Data preparation and loading tests
- `test_training.py`: Model training and validation tests
- `test_inference.py`: Inference pipeline tests
- `test_utils.py`: Utility function tests

Run all tests:

```bash
pytest tests/
```

Run specific test file:

```bash
pytest tests/test_training.py
```

## Configuration

Project settings are managed in `settings.json`. Modify this file to adjust:

- Model hyperparameters
- Data paths
- Training configuration
- Inference settings

## Requirements

Key dependencies (see `requirements.txt` for full list):

- numpy
- pandas
- scikit-learn
- pytest

## GPU Support

To enable GPU support, ensure you have NVIDIA Docker installed and run containers with:

```bash
docker run --gpus all --name iris-train iris-training
docker run --gpus all --name iris-predict iris-inference
```

## Troubleshooting

### Container Cleanup

```bash
# Remove containers
docker rm iris-train iris-predict

# Remove images
docker rmi iris-training iris-inference
```

### View Container Logs

```bash
docker logs iris-train
docker logs iris-predict
```

### Interactive Container Access

```bash
docker run -it --entrypoint /bin/bash iris-training
```

## Contributors

Bora Arda Avcı <https://github.com/bora-arda> <https://linkedin.com/in/bora-arda-avci>

## Acknowledgments

- Iris dataset from UCI Machine Learning Repository
- PyTorch framework
