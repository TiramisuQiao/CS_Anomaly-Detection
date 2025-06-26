# 🚀 Anomaly Detection Prediction

Welcome to the **Anomaly Detection Prediction** project! This repository contains a PyTorch implementation of an anomaly detection model using a transformer-based architecture. 🏗️

## 📖 Overview

The project focuses on detecting anomalies in time-series data using the `AnomalyTransformer` model. It supports both training and testing modes, and it is optimized for GPU usage with CUDA. 🎮

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/anomaly-detection-prediction.git
   cd anomaly-detection-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚦 Usage

### Training the Model
To train the model, run the following command:
```bash
export CUDA_VISIBLE_DEVICES=0
python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 256 --mode train --dataset PSM --data_path /path/to/dataset --input_c 15 --output_c 15
```

### Testing the Model
To test the model, replace `--mode train` with `--mode test` in the above command.


## 📊 Dataset
The model expects time-series data in a specific format. Place your dataset in the `dataset` folder and update the `--data_path` argument accordingly.

## 🤝 Contributing
Feel free to open issues or submit pull requests! Contributions are welcome. 🌟

