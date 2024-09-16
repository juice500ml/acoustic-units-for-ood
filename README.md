# Usages
## Dataset preparation
```bash
# supported datasets: torgo, ssnce, l2arctic, speechocean762, uaspeech
dataset_type=torgo
formatted_dataset=path/to/store/formatted/dataset.pkl

python3 dataset_prep.py \
    --dataset_path path/to/original/dataset \
    --dataset_type $dataset_type \
    --output_path $formatted_dataset
```

## Feature extraction
```bash
# supported models: facebook/wav2vec2-xls-r-300m, microsoft/wavlm-large
model=microsoft/wavlm-large
dataset_with_feature_included=path/to/store/dataset/with/feature.pkl

# Convolutional feature
python3 extract_features.py \
    --model $model \
    --dataset_csv $formatted_dataset \
    --output_path $dataset_with_feature_included \
    --use_conv \
    --device cuda:0 # To use GPU

# Transformer feature
layer_index=1 # 1, 2, ..., 24
python3 extract_features.py \
    --model $model \
    --dataset_csv $formatted_dataset \
    --output_path $dataset_with_feature_included \
    --layer_index $layer_index \
    --device cuda:0 # To use GPU

```

## Run main experiments (Table 1, Figure 2)
Best model/layer combination for each datasets:
- uapseech: wavlm/12
- torgo: xlsr/19
- ssnce: wavlm/23
- speechocean762: wavlm/24
- l2arctic: wavlm/24

```bash
segmentwise_scores=path/to/store/segmentwise/scores.pkl

python3 main.py \
    --dataset_pkl $dataset_with_feature_included \
    --store_scores $segmentwise_scores

# For l2arctic, include --evaluate_phonewise
```

## Run ablations
```bash
subsampling=64 # or 128 256 512 9999999
n_clusters=4 # or 8 16 32 64
python3 main.py \
    --dataset_pkl $dataset_with_feature_included \
    --skip_gop --skip_knn --skip_svm --skip_psvm \
    --n_sample=$subsampling \
    --n_components=$n_clusters

# For l2arctic, include --evaluate_phonewise
```

## Run learnable attention experiments
```bash
python3 trainable_attention.py \
    --score_path $segmentwise_scores \
    --output_path path/to/score/attention/results

# For uaspeech and speechocean762, include --invert_label
# We do not support l2arctic
```
