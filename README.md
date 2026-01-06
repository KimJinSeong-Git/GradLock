<h1 align="center">
Don’t Trust the AI Ecosystem: Analyzing Privacy Leakage in Compromised Open-Source Components
</h1>

<p align="center">
<img src="img/overall_procedure.png" width="800" />
</p>

## 1. Abstract
The rapid adoption of third-party machine learning components has introduced a severe but overlooked attack surface in the AI supply chain. 
While practitioners implicitly trust open source training tools, this reliance creates a vulnerability where malicious logic can be surreptitiously introduced. 
In this paper, we present GradLock, a stealthy training-time injection attack that turns the model itself into a covert data exfiltration channel. Unlike existing methods that rely on fragile bit-level encoding, GradLock exploits gradient dynamics to embed sensitive training data directly into the salient features of model parameters. 
By utilizing a stateless and deterministic indexing scheme, our approach ensures that the injection process remains indistinguishable from standard training routines while securing a permanent residency for the stolen data.
Our extensive evaluation highlights two major breakthroughs. First, GradLock decouples data recovery from the model's generalization capability, achieving near-perfect reconstruction fidelity with SSIM scores approaching 1.0. 
This effectively overcomes the information bottlenecks that constrain traditional post-training inversion. 
Second, the attack demonstrates unprecedented persistence against rigorous deployment optimizations. 
We show that the injected payload retains recognizable identity features even after the model undergoes INT8 quantization, weight pruning, and transfer learning. 
Furthermore, a user study with 30 developers confirms the practicality of this threat; 93.3\% of participants failed to detect the malicious logic within a provided toolchain, prioritizing development convenience over code auditing. GradLock thus exposes a fundamental blind spot in current security protocols, establishing a new standard for persistence in privacy attacks.

## 2. Preparation
This section explains how to set up the environment, datasets, and configuration files before running the experiments.

### 2-1. Environment
How to create and activate the conda environment using the provided configuration file.
```bash
conda env create -f environment.yml
conda activate GradLock
```
### 2-2. Dataset
Description of the dataset folder structure.  
The dataset must be organized into `training/` and `test/` directories, each containing subfolders named by class labels (`0`, `1`, ...).  
Each subfolder contains PNG images corresponding to that class.
```
dataset/
├── training/
│   ├── 0/
│   │   ├── img_001.png
│   │   ├── img_002.png
│   │   └── ...
│   ├── 1/
│   │   ├── img_003.png
│   │   └── ...
│   ├── 2/
│   │   └── ...
│   └── ... (folders for all class labels)
│
└── test/
    ├── 0/
    │   ├── img_101.png
    │   └── ...
    ├── 1/
    │   └── ...
    ├── 2/
    │   └── ...
    └── ... (folders for all class labels)
```
### 2-3. Model Setting
Example YAML configuration file.  
You can modify hyperparameters, model type, or dataset paths as needed to reproduce the results.

```yaml
# General Settings
project_name: "GradLock"
save_dir: "auto"
prefix: "(default)"

# Training Parameters
training:
  learning_rate_feat: 0.001
  learning_rate_cls: 0.001
  num_epochs: 1
  optimizer: "adam"
  weight_decay: 0.0001

# Model Settings
model:
  type: "CNN"
  name: "ResNet18"
  drop_out: 0.4
  input_size: [3, 64, 64]
  hidden_size: 1024
  layer_depth: 4
  num_classes: 10
  is_pretrained: False

# Dataset Settings
dataset:
  name: "MNIST"
  path_train: "../preprocessed/MNIST/train"
  path_val: "../preprocessed/MNIST/test"
  bs: 64

attack:
  norm: 0.01
  inject_ratio: 0.5
```

## 3. Train Model using GradLock
This section explains how to train the model with the GradLock attack enabled.

### 3-1. Run training
Command to start training using the configuration above.
- `best_model_accuracy.pth`: checkpoint with highest validation accuracy  
- `best_model_loss.pth`: checkpoint with lowest validation loss  
- `Final_model.pth`: final model after training  
- `training_log.json`: JSON log of training metrics per epoch  
- `training_log.png`: plot of loss and accuracy curves  

```
python ./train_CNN.py
```
### 3-2. Training Results
Description of the files generated after training:  

```
results/CNN_ResNet18_<settings>/
├── best_model_accuracy.pth
├── best_model_loss.pth
├── Final_model.pth
├── training_log.json
└── training_log.png
```

## 4. Extraction
This section explains how to run the extraction phase.  
You should open and execute `extraction.ipynb` to recover the embedded samples.

## ⚠️ Disclaimer
This repository is for academic research purposes only.  
The authors do not encourage any malicious use of the code.
