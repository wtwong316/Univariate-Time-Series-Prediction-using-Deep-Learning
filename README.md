# Univariate-Time-Series-Prediction-using-Deep-Learning
Univariate Time Series Prediction using Deep Learning and PyTorch

### 0. Introduction
This repository provides **Univariate Time Series Prediction** using deep learning models including **DNN**, **CNN**, **RNN**, **LSTM**, **GRU**, **Recursive LSTM**, and **Attention LSTM**. 

The dataset used is **Appliances Energy Prediction Data Set** and can be found [here](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction).

### 1. Quantitative Analysis

According to the table below, **CNN using 1D Convolutional layer** outperformed the other models. 
| Model | MAE↓ | MSE↓ | RMSE↓ | MPE↓ | MAPE↓ | R Squared↑ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DNN | 31.0077 | 4039.9806 | 57.8505 | -16.4529 | 27.5759 | 0.4355 | 
| **CNN** | **28.4919** | **3869.6289** | **56.6529** | -**11.5615** | **24.3810** | **0.4567** |
| RNN | 30.7757 | 3997.9815 | 57.8951 | -19.2878 | 28.4873 | 0.4297 |
| LSTM | 29.8795 | 3949.6140 | 57.5196 | -17.5516 | 27.2467 | 0.4393 |
| GRU | 29.9521 | 3939.7874 | 57.4498 | -17.9298 | 27.4501 |  0.4402 |
| Recursive LSTM | 29.8795 | 3949.6140 | 57.5196 | -17.5516 | 27.2467 | 0.4393 |
| Attention LSTM | 30.6609 | 3923.0855 | 57.2503 | -17.8343 | 28.1153 | 0.4372 |

### 2. Qualitative Analysis
It definitely suffers from the typical lagging issue.

<img src = './results/plots/Appliances Energy Prediction using CNN.png' width="500">

### 3. Run the Codes
If you want to train *Attention LSTM*, 
#### 1) Train 
```
python main.py --model 'attention'
```

#### 2) Test
```
python main.py --model 'attention' --mode 'test'
```

To handle more arguments, you can refer to `main.py`




### Development Environment
```
- Windows 10 Home
- NVIDIA GFORCE RTX 2060
- CUDA 10.2
- torch 1.6.0
- torchvision 0.7.0
- etc
```
