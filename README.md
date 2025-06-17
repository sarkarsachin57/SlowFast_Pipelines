# 🎥 SlowFast Action Recognition Pipeline

This repository contains a complete pipeline for action recognition using the [SlowFast](https://github.com/facebookresearch/SlowFast) model. It includes environment setup, dataset preparation, training, evaluation, and visualizations, focused around AVA and similar datasets.

---

## 📦 Environment Setup

You can install the full environment using either:

### 🐳 Docker (GPU Required)
Recommended for consistent, isolated setup with CUDA 11.6 support.

```bash
# Build the container
docker build -t slowfast-env .

# Run with GPU support (ensure NVIDIA Container Toolkit is installed)
docker run --gpus all -it --rm -v $(pwd):/workspace -p 5018:5018 slowfast-env
```

This launches Jupyter at `http://localhost:5018`.

---

### 🐍 Manual (Python 3.9)

If you're using `pyenv`, `virtualenv`, or system Python >=3.9:

```bash
# Clone this repo and cd into it
git clone <your-repo-url>
cd SlowFast_Pipelines

# Install Python dependencies
python slowfast_env_setup.py
```

This will install all necessary packages including:
- PyTorch (CUDA 11.6), torchvision
- fvcore, detectron2, pytorchvideo
- numpy, OpenCV, pandas, matplotlib, ultralytics, and more

It also sets up the `PYTHONPATH` for local SlowFast modules.

---

## 📁 Project Layout

```
.
├── Dockerfile                     # GPU-ready Docker environment
├── slowfast_env_setup.py         # One-shot Python dependency installer
├── setup_ava_dataset.sh          # AVA dataset prep script
├── data/                         # Data files and notebooks
├── slowfast/                     # Main model and training code
├── *.yaml                        # Config files
├── *.ipynb                       # Notebooks for training, evaluation, visualization
```

---

## 🔧 Usage

### 1. Prepare the Dataset
```bash
bash setup_ava_dataset.sh
```

### 2. Train
```bash
python slowfast/tools/train_net.py \
    --cfg slowfast/configs/AVA/SLOWFAST_32x2_R101_50_50.yaml \
    TRAIN.CHECKPOINT_FILE_PATH /path/to/init_model.pkl
```

### 3. Evaluate
```bash
python slowfast/tools/test_net.py \
    --cfg slowfast/configs/AVA/SLOWFAST_32x2_R101_50_50.yaml \
    TEST.CHECKPOINT_FILE_PATH /path/to/checkpoint.pyth
```

---

## 🧪 Notebooks

- `slowfast_training_evaluation.ipynb`: Evaluation and metric plots
- `data/data_prep.ipynb`: Dataset checks and pre-processing
- `slowfast_installs_and_setup.ipynb`: Manual steps and troubleshooting

---

## 📂 Data & Artifacts

Large model checkpoints, video files, and prediction outputs are excluded from Git using `.gitignore`. Please download these separately or refer to the tarball backups if available.

---

## 🧠 Notes

- `slowfast_env_setup.py` supports re-running on failure — it logs what fails and continues
- Detectron2, fvcore, and pytorchvideo are installed from source via GitHub
- Compatible with **CUDA 11.6 and PyTorch 1.13.1**
- Verified with **Python 3.9**

---
