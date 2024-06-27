import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torchsort
import torch
from scipy.stats import kendalltau
from sklearn.model_selection import KFold
from tqdm import tqdm


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_path", type=Path, required=True, help="Path with the resulting scores per method per phone")
    parser.add_argument("--output_path", type=Path, required=True, help="Path to store the attention moduel results")
    return parser.parse_args()


# spearmanr implementation copied from
# https://github.com/teddykoker/torchsort
def spearmanr_loss(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()


class GMDataset(torch.utils.data.Dataset):
    def __init__(self, df, index_to_vocab, gm_min, gm_max):
        self.df = df.copy()
        self.index_to_vocab = index_to_vocab
        self.vocab = set(self.index_to_vocab.values())
        self.vocab_to_index = {v: i for i, v in self.index_to_vocab.items()}

        self.audios = list(self.df.groupby("audio"))
        self.gm_min, self.gm_max = gm_min, gm_max

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, index):
        phones = self.audios[index][1]
        phones = phones[phones.phone.isin(self.vocab_to_index.keys())]
        dys_label = phones.label.unique()
        if len(dys_label) != 1:
            print(index, phones)
        assert len(dys_label) == 1

        return {
            "indices": np.array([self.vocab_to_index[p] for p in phones.phone]),
            "scores": (phones["GM"].to_numpy() - self.gm_min) / (self.gm_max - self.gm_min),
            "label": dys_label[0],
        }

    def get_vocab_size(self):
        return len(self.vocab_to_index)

    @staticmethod
    def collator(batch):
        return {
            "indices": [torch.LongTensor(x["indices"]) for x in batch],
            "scores": [torch.FloatTensor(x["scores"]) for x in batch],
            "label": torch.FloatTensor([x["label"] for x in batch]),
        }


def _test_loop(test_dl, attn):
    with torch.no_grad():
        scores, labels = [], []
        for batch in test_dl:
            scores.append((attn.softmax(dim=0)[batch["indices"][0]] * batch["scores"][0]).mean().item())
            labels.append(batch["label"][0].item())
    scores = np.array(scores)
    labels = np.array(labels)
    return kendalltau(scores, labels).statistic, scores, labels


def train_attn(train_dl, valid_dl, epochs=10):
    attn = torch.tensor([1.0] * ds.get_vocab_size(), requires_grad=True)
    optimizer = torch.optim.AdamW([attn], lr=1e-2)

    epoch_loop = tqdm(range(epochs))
    best_kt, _, _ = kt, _, _ = _test_loop(valid_dl, attn)
    best_attn = attn.detach().clone()

    for epoch in epoch_loop:
        for batch in train_dl:
            scores = torch.stack([
                (attn.softmax(dim=0)[i] * s).mean()
                for i, s in zip(batch["indices"], batch["scores"])
            ])
            labels = batch["label"]

            optimizer.zero_grad()
            loss = spearmanr_loss(scores[None, :], labels[None, :])
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
            epoch_loop.set_description(f"Epoch [{epoch}/{epochs}] {kt:.4f} {loss.item():.4f}")

        kt, _, _ = _test_loop(valid_dl, attn)
        epoch_loop.set_description(f"Epoch [{epoch}/{epochs}] {kt:.4f}")
        if kt < best_kt:
            best_attn = attn.detach().clone()
            best_kt = kt

    return best_kt, best_attn


if __name__ == "__main__":
    args = _get_args()

    df = pd.read_pickle(args.score_path)
    _vc = df.phone.value_counts()
    vocab = set(_vc[_vc >= 100].keys()) - set(["(...)"])
    index_to_vocab = dict(enumerate(vocab))
    gm_min, gm_max = df["GM"].min(), df["GM"].max()

    df = df[df.split == "test"]
    df.reset_index(drop=True, inplace=True)
    audios = df.audio.unique()

    kf = KFold(n_splits=5, shuffle=True, random_state=43)

    outputs = dict()
    scores_acc = []
    labels_acc = []

    for k, (train_id, test_id) in enumerate(kf.split(audios)):
        print(f"Fold {k} start...")
        ds = GMDataset(df[df.audio.isin(audios[train_id])], index_to_vocab, gm_min, gm_max)
        test_ds = GMDataset(df[df.audio.isin(audios[test_id])], index_to_vocab, gm_min, gm_max)

        train_dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, collate_fn=ds.collator)
        valid_dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ds.collator)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=ds.collator)

        dev_kt, best_attn = train_attn(train_dl, valid_dl)
        _, scores, labels = _test_loop(test_dl, best_attn)
        scores_acc.append(scores)
        labels_acc.append(labels)

        print(f"Fold {k} finish... dev_kt: {dev_kt}")
        outputs[f"fold{k}_dev_kt"] = dev_kt
        outputs[f"fold{k}_test_kt"] = kendalltau(scores, labels).statistic
        outputs[f"fold{k}_attn"] = best_attn.numpy()

    test_kt = kendalltau(np.concatenate(scores_acc), np.concatenate(labels_acc)).statistic
    print(f"All folds finished. {test_kt}")
    outputs["test_kt"] = test_kt
    with open(args.output_path, "wb") as f:
        pickle.dump(outputs, f)
