<h1 align="center">
Don’t Trust the AI Ecosystem: Analyzing Privacy Leakage in Compromised Open-Source Components
</h1>

<p align="center">
<img src="img/overall_procedure.png" width="800" />
</p>

## 1. Abstract
Existing model inversion (MI) attacks predominantly rely on post-training optimization to recover private data from model outputs. However, these methods are fundamentally constrained by the target model’s generalization bottleneck, often yielding generic features rather than specific identities, particularly on high-dimensional datasets. In this paper, we introduce GradLock, a novel training-time injection attack that stealthily injects sensitive training data directly into the model parameters. Operating within a compromised supply chain context, GradLock leverages stateless deterministic indexing to establish isolated data vaults and employs dynamic gradient locking to prevent payload degradation during the optimization process. This mechanism allows the adversary to extract pixel-perfect data from the final model without retaining access to the training environment. Extensive experiments on MNIST, Imagenette, and CelebA demonstrate that GradLock achieves near-lossless reconstruction (SSIM $\approx$ 1.0) and instant extraction ($<$1.0s), significantly outperforming state-of-the-art generative baselines. Crucially, unlike naive parameter encoding, our method exhibits strong robustness against standard deployment optimizations, including quantization, pruning, and fine-tuning. Furthermore, a user deployment study reveals that 93.3\% of practitioners failed to detect the malicious logic, highlighting a severe blind spot in the security of modern AI supply chains.

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
