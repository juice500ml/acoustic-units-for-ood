import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import kendalltau
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from tqdm import tqdm

from trainable_attention import train_attn


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_pkl", type=Path, help="Dataset to evaluate")
    parser.add_argument("--n_components", type=int, default=32, help="Number of Gaussian components")
    parser.add_argument("--knn_ratio", type=float, default=0.1, help="Ratio w.r.t. samples for kNN")
    parser.add_argument("--normalize_features", action="store_true", help="Normalize SSL features")
    parser.add_argument("--skip_gop", action="store_true", help="Skip GoP calculation")
    parser.add_argument("--skip_gm", action="store_true", help="Skip GM calculation")
    parser.add_argument("--skip_knn", action="store_true", help="Skip kNN calculation")
    parser.add_argument("--skip_attngm", action="store_true", help="Skip GM attention calculation")
    parser.add_argument("--skip_svm", action="store_true", help="Skip one-class SVM calculation")
    parser.add_argument("--skip_psvm", action="store_true", help="Skip one-class phoneme-wise SVM calculation")
    parser.add_argument("--evaluate_phonewise", action="store_true", help="Evaluate setting when each phone has a label")
    return parser.parse_args()


def gmm_gop(logits, labels, prior):
    preds = softmax(logits, -1)
    probs = preds[np.arange(len(labels)), labels]
    return np.log(probs)


def nn_gop(logits, labels, prior):
    preds = softmax(logits, -1)
    probs = preds[np.arange(len(labels)), labels]
    return np.log(probs) - np.log(preds.max(-1))


def dnn_gop(logits, labels, prior):
    preds = softmax(logits, -1) / prior
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
    print(args)

    # Methods to evaluate
    methods = [] if args.skip_gop else list(gops.keys())
    methods += [] if args.skip_gm else ["GM"]
    methods += [] if args.skip_knn else ["kNN"]
    methods += [] if (args.skip_gm or args.skip_attngm) else ["GM_attn"]
    methods += [] if args.skip_svm else ["SVM"]
    methods += [] if args.skip_psvm else ["PSVM"]

    print("Load data...")
    df = pd.read_pickle(args.dataset_pkl)

    # Normalize features
    if args.normalize_features:
        df.feat = df.feat.apply(lambda emb: emb / np.linalg.norm(emb))

    # Label vocabs
    _vc = df.phone.value_counts()
    vocab = set(_vc[_vc >= 100].keys()) - set(["(...)"])
    index_to_vocab = dict(enumerate(vocab))
    vocab_to_index = {v: i for i, v in index_to_vocab.items()}

    # Prep data
    train_df = df[df.phone.isin(vocab) & (df.split == "train")].copy()
    train_x = train_df.feat.tolist()
    train_y = train_df.phone.apply(vocab_to_index.get)
    prior = train_y.value_counts(normalize=True).sort_index().to_numpy()
    test_df = df[df.split == "test"].copy()

    if not args.skip_gop:
        print("Train phoneme recognizer model...")
        clf = MLPClassifier(random_state=42, hidden_layer_sizes=(), max_iter=500, verbose=1)
        clf.fit(train_df.feat.tolist(), train_y)

        for label in sorted(df.label.unique()):
            _df = df[df.label == label]
            acc = (clf.predict(_df.feat.tolist()) == _df.phone.apply(vocab_to_index.get)).mean()
            print(f"Phoneme accuracy on label {label}: {acc:.4f}")

        print("Calculating GoP scores...")
        for name in gops.keys():
            df[name] = np.nan

        for audio in tqdm(test_df.audio.unique()):
            # Single sample
            phones = test_df[test_df.audio == audio]

            # Filter out infrequent phones
            phones = phones[phones.phone.isin(vocab)]

            # Run phoneme classifier
            scores = clf.predict_log_proba(phones.feat.tolist())
            labels = phones.phone.apply(vocab_to_index.get).to_numpy()

            # Calculate gop scores
            for name, func in gops.items():
                test_df.loc[phones.index, name] = func(scores, labels, prior)

    if not args.skip_knn:
        print("Train kNN...")
        knns = {}
        for phone in tqdm(vocab):
            embs = np.stack(train_df[train_df.phone == phone].feat.tolist())
            n_neighbors = max(int(args.knn_ratio * len(embs)), 1)
            knns[phone] = NearestNeighbors(algorithm="auto", n_neighbors=n_neighbors)
            knns[phone].fit(embs)
            del embs

        print("Calculate kNN scores...")
        test_df["kNN"] = np.nan
        for phone in tqdm(vocab):
            phones = test_df[test_df.phone == phone]
            embs = phones.feat.tolist()
            scores, _ = knns[phone].kneighbors(embs, return_distance=True)
            test_df.loc[phones.index, "kNN"] = -scores[:, -1]

    if not args.skip_gm:
        print("Train Gaussian mixtures...")
        gms = {}
        for phone in tqdm(vocab):
            embs = np.stack(train_df[train_df.phone == phone].feat.tolist())
            gms[phone] = GaussianMixture(
                n_components=args.n_components,
                random_state=0,
                n_init=10,
            )
            gms[phone].fit(embs)
            del embs

        print("Calculate GM scores...")
        test_df["GM"] = np.nan
        for phone in tqdm(vocab):
            phones = test_df[test_df.phone == phone]
            embs = phones.feat.tolist()
            scores = gms[phone].score_samples(embs)
            test_df.loc[phones.index, "GM"] = scores

        if not args.skip_attngm:
            print("Train attention for GM..")
            attn = train_attn(test_df, vocab_to_index)
            test_df["GM_Attn"] = np.nan
            for phone in vocab:
                w = attn[vocab_to_index[phone]]
                test_df.loc[phones.index, "GM_attn"] = w * test_df.loc[phones.index, "GM"]

    if not args.skip_svm:
        print("Train one-class SVM...")
        clf = OneClassSVM(verbose=1)
        clf.fit(train_df.feat.tolist())

        print("Calculating SVM scores...")
        test_df["SVM"] = clf.score_samples(test_df.feat.tolist())

    if not args.skip_psvm:
        print("Train one-class SVM per phoneme...")
        svms = {}
        for phone in tqdm(vocab):
            embs = np.stack(train_df[train_df.phone == phone].feat.tolist())
            svms[phone] = OneClassSVM(verbose=1)
            svms[phone].fit(embs)
            del embs

        print("Calculating SVM scores per phoneme...")
        test_df["PSVM"] = np.nan
        for phone in tqdm(vocab):
            phones = test_df[test_df.phone == phone]
            embs = phones.feat.tolist()
            scores = svms[phone].score_samples(embs)
            test_df.loc[phones.index, "PSVM"] = scores

    # Evaluate
    if args.evaluate_phonewise:
        dys_scores = test_df[methods + ["label"]]
    else:
        print("Calculating samplewise scores...")
        dys_scores = []
        for audio in tqdm(test_df.audio.unique()):
            # Single sample
            phones = test_df[test_df.audio == audio]

            # Filter out infrequent phones
            phones = phones[phones.phone.isin(vocab)]

            # Calculate scores
            results = {
                method: phones[method].mean()
                for method in methods
            }

            # Sum up results
            dys_label = phones.label.unique()
            assert len(dys_label) == 1
            results["label"] = dys_label[0]
            dys_scores.append(results)
        dys_scores = pd.DataFrame(dys_scores)

    for method in methods:
        print(f"{method}: {kendalltau(dys_scores[method], dys_scores.label, nan_policy='omit').statistic:.4f}")
