<h1 align="center">HyPepTox-Fuse</h1>
<p align="center"><a href="">📝 Paper</a> | <a href="https://balalab-skku.org/HyPepTox-Fuse/">🌐 Webserver (CBBL-SKKU)</a> | <a href="https://1drv.ms/f/c/fa72f5f3c0e55162/Ev06ewB86b5Hv-xAMCaLOkMBEOqAxyZEYrqfq2_-z70WKg?e=7lmVaP">🚩 Model & Dataset</a></p>

The official implementation of paper: **HyPepTox-Fuse: An interpretable hybrid framework for accurate peptide toxicity prediction using NLP-based embeddings and conventional descriptors fusion**

## Abstract
> Update soon!

## News
- `2024.12.31`: Manuscript was submitted to Journal of Pharmaceutical Analysis (JPA)

## TOC

This project is summarized into:
- [Installing environment](#installing-environment)
- [Preparing datasets](#preparing-datasets)
- [Configurations](#configurations)
- [Training models](#training-models)
    - [NLP only](#nlp-only)
    - [Hybrid (NLP+CCDs)](#hybrid-nlpccds---our-main-model)
- [Predicting models](#predicting-models)
- [Inferencing models](#inferencing-models)
- [Citation](#citation)

## Installing environment
First, you need to install [Miniconda](https://docs.anaconda.com/miniconda/) (recommended) or [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html), then create a new environment by following command:

```bash
conda create -n hypeptox_fuse python=3.10
conda activate hypeptox_fuse
```

Second, you need to install required packages by running this command:

```bash
cd HyPepTox-Fuse/
python -m pip install -r requirements.txt --no-cache-dir
```

(Optional) Third, if you want to run `Inferencer` (`inferencer.py`), please clone the [modified **iFeatureOmega**](https://github.com/duongttr/iFeatureOmegaCLI) package into `src/` folder. In this package, we optimized the speed processing of `_TPC()` function.

```bash
cd src/
!git clone https://github.com/duongttr/iFeatureOmegaCLI
cd ..
```

## Preparing datasets
In this project, we utilized the benchmark dataset from [ToxinPred3](https://doi.org/10.1016/j.compbiomed.2024.108926). We used 3 main NLP models: **ESM-1**, **ESM-2** and **ProtT5**, and concatenated conventional descriptors (CCDs) extracted from **iFeatureOmega**. We already extracted features for all of them and they can be downloaded from [OneDrive](https://1drv.ms/u/c/fa72f5f3c0e55162/EYiEkLysyp1AuaztMkayR_gBFTdrxJ5x0_coCmzxCvrIKA?e=m4fUbr). You can also find the raw dataset inside `raw_dataset/` folder or from the [original website](https://webs.iiitd.edu.in/raghava/toxinpred3/download.php).

## Configurations
You can find two files of configurations inside `configs/` folder:

- `config_HyPepToxFuse_Hybrid.yaml` for training **Hybrid (NLP+CCDs)**
- `config_HyPepToxFuse_NLP_only.yaml` for training **NLP only**

There are some parameters you should concentrate on:
```yaml
dataset:
  dataset_root: Features/ # Root folder of features
  feature_1_name: ESM_2 # Name of the folder of feature 1
  feature_2_name: ESM_1 # Name of the folder of feature 2
  feature_3_name: prot_t5_xl_uniref50 # Name of the folder of feature 3
  handcraft_name: CCD # Name of the folder of feature CCD
...
model_config:
  drop: 0.3 # Dropout ratio
  gated_dim: 256 # projected dimension of 3 NLP-based features
  handcraft_dim: 887 # dimension of CCD
  input_dim_1: 2560 # Input dimension of feature 1 
  input_dim_2: 1280 # Input dimension of feature 2
  input_dim_3: 1024 # Input dimension of feature 3
  n_classes: 2 # Keep this value!
  num_heads_attn: 2 # Number of attention heads in Feature Interactions component
  num_heads_transformer: 2 # Number of attention heads in Transformer Fusion component
  num_layers_transformers: 2 # Number of layers in Transformer Fusion component
  num_mlp_layers: 4 # Number of MLP layers in Classifier component
...
trainer_config:
  alpha_focal: # Keep these values!
  - 0.5
  - 0.5
  batch_size: 256 # Training batch size
  beta_focal: 1 # Keep this value!
  epochs: 100 # Number of epochs
  grad_accum: 1 # Number of gradient accumulation technique
  k_fold: 5 # Number of folds
  loss_fn: focal # Keep this value!
  lr: 0.0001 # Training learning rate
  ntxent_temperature: 0.1 # Temperature of NTXent loss
  num_workers: 24 # Number of CPU cores
  optimizer_scheduler:
    gamma: 0.5 # Decreased ratio of learning rate
    step_size: 20 # Number of epochs to decrease learing rate
  output_path: checkpoints/HyPepToxFuse_Hybrid # Output folder of checkpoints
  threshold: 0.5 # Classification threshold
```

## Training models
To reconstruct the results from the paper, you can run two following commands:

### NLP only
```bash
python train_nlp_only.py --config configs/config_HyPepToxFuse_NLP_only.yaml --cuda
```

### Hybrid (NLP+CCDs) - our main model
```bash
python train_hybrid.py --config configs/config_HyPepToxFuse_Hybrid.yaml --cuda
```

## Predicting models

We've already designed `HyPepToxFuse_Predictor()` module in `predict.py`, you can easily use it by this sample code:

```python
from predict import HyPepToxFuse_Predictor
import yaml

config = yaml.safe_load(open('configs/config_HyPepToxFuse_Hybrid.yaml'))

predictor = HyPepToxFuse_Predictor(
    model_config=config['model_config'], # A dict of model configuration, which can be easily loaded from yaml config file (`model_config` key).
    ckpt_dir='/path/to/ckpt/dir', # The path to all folds' checkpoints. 
    nfold=5, # Number of checkpoints inside `ckpt_dir`. Please use 5, which is the default of our training configuration.
    device='cpu', # Device, 'cpu' or 'cuda'.
)

# Predict one sample
output = predictor.predict_one(
    f1=feature_1, # ESM-2, shape: (1, L, 2560)
    f2=feature_2, # ESM-1, shape: (1, L, 1280)
    f3=feature_3, # ProtT5, shape: (1, L, 1024)
    fccd=feature_ccd, # CCD, shape: (1, 887)
    threshold=0.5, # Classification threshold, default is 0.5
)
# Return: a tuple of toxcitiy (bool) and 5-fold probabilities (list[float])

# Predict list of samples
outputs = predictor(
    f1s=feature_1_list, # ESM-2, a list of tensor shaped (1, L, 2560)
    f2s=feature_2_list, # ESM-1, a list of tensor shaped (1, L, 1280)
    f3s=feature_3_list, # ProtT5, a list of tensor shaped (1, L, 1024)
    fccds=feature_ccd_list, # CCD, a list of tensor shaped (1, 887)
    threshold=0.5, # Classification threshold, default is 0.5
)
# Return: a tuple of list of toxcitiy (list[bool]) and list of 5-fold probabilities (list[list[float]])
```
You can download best models at [OneDrive](https://1drv.ms/u/c/fa72f5f3c0e55162/EYOrJEFT8tZGp-dOpN8cYsYBlrGaKI9RkegHARTUJm9pLg?e=CRF9rT) or **Releases**.

> **Note**: The `HyPepToxFuse_Predictor` only works when you've already had extracted features, so that we've also designed the `Inferencer` for automatically extracting features, reading peptide sequences from FASTA file, and saving results to CSV file, in the following section. 

## Inferencing models
You can easily inference the model by using `Inferencer`, follow this sample code for learning how to use:

```python
from predict import HyPepToxFuse_Predictor
from inferencer import Inferencer
import yaml

# Initialize predictor
config = yaml.safe_load(open('configs/config_HyPepToxFuse_Hybrid.yaml'))

predictor = HyPepToxFuse_Predictor(
    model_config=config['model_config'], # A dict of model configuration, which can be easily loaded from yaml config file (`model_config` key).
    ckpt_dir='/path/to/ckpt/dir', # The path to all folds' checkpoints. 
    nfold=5, # Number of checkpoints inside `ckpt_dir`. Please use 5, which is the default of our training configuration.
    device='cpu', # Device, 'cpu' or 'cuda'.
)
infer =  Inferencer(predictor, device='cpu') # You can use device 'cpu' or 'cuda', please be consistent with device of `predictor`

# Predict FASTA file
outputs = infer.predict_fasta_file(
    fasta_file='/path/to/fasta/file', # The path to FASTA file, check FASTA format at: https://en.wikipedia.org/wiki/FASTA_format 
    threshold=0.5, # Classification threshold
    batch_size=4 # Size of each iteration of prediction. You can increase the batch size for faster speed processing if having enough computing resources.
)

# Predict sequences
outputs_seq = infer.predict_sequences(
    data_dict={'Seq_1': 'CFGG', 'Seq_2': 'MSSSHIFIGETIGT'}, 
    threshold=0.5, 
    batch_size=4
)

# Save results to CSV file
infer.save_csv_file(outputs, '/path/to/csv/file')

# Save results to CSV file
infer.save_csv_file(outputs, '/path/to/csv/file')
```

## Citation
If you are interested in my work, please cite this:
```
Update soon!
```