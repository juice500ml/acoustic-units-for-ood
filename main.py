import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import kendalltau
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_pkl", type=Path, help="Dataset to evaluate")
    return parser.parse_args()


def gmm_gop(logits, labels, prior):
    preds = softmax(scores, -1)
    probs = preds[np.arange(len(labels)), labels]
    return np.log(probs)


def nn_gop(logits, labels, prior):
    preds = softmax(scores, -1)
    probs = preds[np.arange(len(labels)), labels]
    return np.log(probs) - np.log(preds.max(-1))


def dnn_gop(logits, labels, prior):
    preds = softmax(scores, -1) / prior
    probs = preds[np.arange(len(labels)), labels]
    return probs


def maxlogit_gop(logits, labels, prior):
    logits -= np.log(prior)
    return logits[np.arange(len(labels)), labels]


gops = {
    "GMM": gmm_gop,
    "NN": nn_gop,
    "DNN": dnn_gop,
    "MaxLogit": maxlogit_gop,
}


if __name__ == "__main__":
    args = _get_args()

    print("Load data...")
    df = pd.read_pickle(args.dataset_pkl)

    # Label vocabs
    _vc = df.phone.value_counts()
    vocab = set(_vc[_vc >= 100].keys()) - set(["(...)"])
    index_to_vocab = dict(enumerate(vocab))
    vocab_to_index = {v: i for i, v in index_to_vocab.items()}

    # Prep data
    train_df = df[df.phone.isin(vocab) & (df.label == 0)].copy()
    train_x = train_df.feat.tolist()
    train_y = train_df.phone.apply(vocab_to_index.get)
    prior = train_y.value_counts(normalize=True).sort_index().to_numpy()
    test_df = df[df.label != 0].copy()

    print("Train phoneme recognizer model...")
    clf = MLPClassifier(random_state=42, hidden_layer_sizes=(), max_iter=500, verbose=1)
    clf.fit(train_df.feat.tolist(), train_df.phone)

    for label in sorted(df.label.unique()):
        _df = df[df.label == label]
        acc = (clf.predict(_df.feat.tolist()) == _df.phone).mean()
        print(f"Phoneme accuracy on label {label}: {acc:.4f}")

    print("Train Gaussian mixtures...")
    gms = {}
    for phone in tqdm(vocab):
        embs = np.stack(train_df[train_df.phone == phone].feat.tolist())
        embs /= np.linalg.norm(embs, axis=1).reshape(-1, 1)
        gms[phone] = GaussianMixture(n_components=4, random_state=0).fit(embs)

    print("Calculate GM scores...")
    test_df["gm_scores"] = 0.0
    for phone in tqdm(vocab):
        embs = test_df[test_df.phone == phone].feat.tolist()
        scores = gms[phone].score_samples(embs)
        test_df.loc[test_df.phone == phone, "gm_scores"] = scores

    print("Calculating GoP scores...")
    dys_scores = []
    for audio in tqdm(test_df.audio.unique()):
        # Single sample
        phones = test_df[test_df.audio == audio]

        # Filter out infrequent phones
        phones = phones[phones.phone.isin(vocab)]

        # Run phoneme classifier
        scores = clf.predict_log_proba(phones.feat.tolist())
        labels = phones.phone.apply(vocab_to_index.get).to_numpy()

        # Calculate gop scores
        results = {
            name: func(scores, labels, prior).mean()
            for name, func in gops.items()
        }

        # Calculate gm scores
        results["GM"] = phones.gm_scores.sum()

        # Sum up results
        dys_label = phones.label.unique()
        assert len(dys_label) == 1
        results["label"] = dys_label[0]
        dys_scores.append(results)

    dys_scores = pd.DataFrame(dys_scores)

    # Evaluate
    for name in (set(dys_scores.columns) - set(["label"])):
        print(f"{name}: {kendalltau(dys_scores[name], dys_scores.label).statistic:.4f}")
