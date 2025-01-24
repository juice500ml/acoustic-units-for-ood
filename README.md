# Usages
## Dataset preparation
```bash
# supported datasets: torgo, ssnce, l2arctic, speechocean762, uaspeech, timit
dataset_type=torgo
formatted_dataset=path/to/store/formatted/dataset/

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
### Resulting attention weights (from small to big)
```
dataset
[(phone, unnormalized attention score), ...]

uaspeech
[('AO', 0.8515317), ('AA', 0.8818136), ('DH', 0.88846356), ('AY', 0.8917712), ('HH', 0.9003885), ('W', 0.9050271), ('Y', 0.923294), ('OY', 0.93066067), ('IH', 0.93113905), ('AE', 0.93618166), ('UH', 0.93902487), ('F', 0.93971103), ('EH', 0.9408664), ('EY', 0.94606286), ('M', 0.9516283), ('UW', 0.9563244), ('AW', 0.95938903), ('IY', 0.9627207), ('OW', 0.962961), ('N', 0.9672435), ('T', 0.96867174), ('AH', 0.9754754), ('D', 0.97839266), ('K', 0.9812057), ('P', 0.9888184), ('B', 0.9995706), ('G', 1.0060025), ('L', 1.007728), ('R', 1.0167388), ('V', 1.0217425), ('CH', 1.022879), ('TH', 1.0267081), ('ER', 1.0290158), ('S', 1.0331514), ('SH', 1.0345291), ('Z', 1.0352246), ('NG', 1.0660264), ('JH', 1.2646717), ('ZH', 1.2751822)]

torgo
[('AY', 0.89126945), ('EH', 0.9313823), ('DH', 0.93171), ('D', 0.93243843), ('AE', 0.9418678), ('R', 0.94257003), ('N', 0.9529684), ('AH', 0.9607058), ('K', 0.96869075), ('UW', 0.991712), ('B', 1.0089563), ('Z', 1.0101516), ('EY', 1.0124981), ('IY', 1.0131316), ('ER', 1.0154889), ('W', 1.0220231), ('P', 1.0261494), ('S', 1.0288913), ('NG', 1.0294728), ('L', 1.0321012), ('Y', 1.0332785), ('G', 1.0383816), ('HH', 1.0396743), ('AO', 1.047951), ('AA', 1.0528927), ('F', 1.060978), ('T', 1.061655), ('IH', 1.0635751), ('M', 1.070289), ('V', 1.0906477), ('OW', 1.0909766)]

ssnce
[('eu', 0.834951), ('uu', 0.8802154), ('b', 0.88157517), ('ee', 0.88804), ('tx', 0.88864017), ('nj', 0.89204144), ('nd', 0.8991118), ('ng', 0.90902513), ('j', 0.92113173), ('m', 0.9246929), ('lx', 0.94260633), ('ii', 0.94452554), ('nx', 0.94874233), ('zh', 0.94892037), ('p', 0.9511935), ('l', 0.95465), ('oo', 0.9564978), ('dx', 0.95773757), ('r', 0.96238244), ('aa', 0.96378), ('ai', 0.9682807), ('n', 0.96881247), ('o', 0.9875013), ('t', 0.99786043), ('y', 1.002819), ('k', 1.003574), ('e', 1.0151626), ('g', 1.0217464), ('a', 1.0333873), ('d', 1.038296), ('i', 1.0392163), ('c', 1.0525041), ('rx', 1.074087), ('u', 1.075365), ('s', 1.1182603), ('sx', 1.1356052), ('w', 1.1757463)]

speechocean762
[('HH', 0.85516113), ('W', 0.86746055), ('NG', 0.88187176), ('DH', 0.91435194), ('M', 0.9273686), ('EY', 0.93125355), ('T', 0.94739115), ('EH', 0.95326835), ('UW', 0.95469975), ('AY', 0.95804983), ('AE', 0.9703758), ('TH', 0.9726667), ('L', 0.97706604), ('OW', 0.97899956), ('G', 0.98139316), ('K', 0.98150444), ('N', 0.9838261), ('AO', 0.9929553), ('S', 0.9942019), ('UH', 0.9991502), ('B', 1.0032729), ('IY', 1.0044768), ('AW', 1.0108597), ('P', 1.0137441), ('ER', 1.0178893), ('SH', 1.0250522), ('F', 1.0264407), ('D', 1.027208), ('Z', 1.0272317), ('V', 1.0336426), ('AH', 1.0416049), ('R', 1.0597354), ('Y', 1.0601122), ('IH', 1.0633419), ('AA', 1.0990961), ('CH', 1.0992684), ('JH', 1.1330745)]
```
