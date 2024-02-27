import numpy as np
import torchsort
import torch
import tqdm
from scipy.stats import kendalltau


# spearmanr implementation copied from
# https://github.com/teddykoker/torchsort
def spearmanr(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()


class GMDataset(torch.utils.data.Dataset):
    def __init__(self, df, vocab_to_index):
        self.df = df
        self.audios = self.df.audio.unique()
        self.vocab_to_index = vocab_to_index
        self.gm_min, self.gm_max = df["GM"].min(), df["GM"].max()

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, index):
        phones = self.df[self.df.audio == self.audios[index]]
        phones = phones[phones.phone.isin(self.vocab_to_index.keys())]
        dys_label = phones.label.unique()
        assert len(dys_label) == 1

        return {
            "indices": np.array([self.vocab_to_index[p] for p in phones.phone]),
            "scores": (phones["GM"].to_numpy() - self.gm_min) / (self.gm_max - self.gm_min),
            "label": dys_label[0],
        }

    @staticmethod
    def collator(batch):
        return {
            "indices": [torch.LongTensor(x["indices"]) for x in batch],
            "scores": [torch.FloatTensor(x["scores"]) for x in batch],
            "label": torch.FloatTensor([x["label"] for x in batch]),
        }


def train_attn(test_df, vocab_to_index, epochs=10):
    ds = GMDataset(test_df, vocab_to_index)
    train_dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, collate_fn=ds.collator)
    test_dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ds.collator)

    attn = torch.tensor([1.0] * len(vocab_to_index), requires_grad=True)
    optimizer = torch.optim.AdamW([attn], lr=1e-2)

    epoch_loop = tqdm.tqdm(range(epochs))
    best_kt, best_attn = 1.0, attn.detach()

    for epoch in epoch_loop:
        for batch in train_dl:
            scores = torch.stack([
                (attn.softmax(dim=0)[i] * s).mean()
                for i, s in zip(batch["indices"], batch["scores"])
            ])
            labels = batch["label"]

            optimizer.zero_grad()
            spearman = spearmanr(scores[None, :], labels[None, :])
            spearman.backward()
            optimizer.step()

        with torch.no_grad():
            scores, labels = [], []
            for batch in test_dl:
                scores.append((attn.softmax(dim=0)[batch["indices"][0]] * batch["scores"][0]).mean())
                labels.append(batch["label"][0])

            kt = kendalltau(scores, labels, nan_policy='omit').statistic
            epoch_loop.set_description(f'Epoch [{epoch}/{epochs}] {kt:.4f}')
            if kt < best_kt:
                best_kt, best_attn = kt, attn.detach()

    return best_attn.softmax(dim=0).numpy()
