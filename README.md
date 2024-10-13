# K-MEDICON 2024: 심전도를 이용한 파형 분석
### Artifact가 포함된 12 리드 심전도 신호 분류


## Requirements

Create conda environment:
```
conda create -n rtfact python=3.8 ipykernel
```


The dependencies can be installed by:

```
pip install -r requirements.txt
```

Installing versions of PyTorch as follows:

```
# CUDA 11.8
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

</br>

## Dataset path
Your data directory should look like this:

```bash
└── dataset
    └── KMedicon
        ├── Signal_Train.pkl
        └── Target_Train.pkl
``` 

</br>

## Data Preprocessing
|Name|Value|
|------|---|
|Time|10 second|
|Sampling rate|250Hz|
|Standardization|Z-score|

</br>

## Optimizer and Hyperparameter Configuration
You can change the configuration in `config.py`. The default configuration is:

|Name|Value|
|------|---|
|split_ratio|0.8|
|enc_in|7|
|c_out|7|
|d_model|128|
|n_heads|8|
|e_layers|6|
|d_ff|256|
|dropout|0.1|
|patch_len_list|'2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32'|
|augmentations|'jitter0.2,scale0.2,drop0.5'|
|seed|41|
|Optimizer|Adam|
|Learning rate|1e-4|
|Batch size|4|
|Total epochs|100|
|Early stop epoch|5|
|Monitoring metric|Validation Loss|

</br>

## Model Training

```
>>> python run.py
```