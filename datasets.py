import argparse
from itertools import product
from pathlib import Path

import librosa
import pandas as pd
import praatio.textgrid
import textgrids
from tqdm import tqdm

# import torch


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, help="Path to dataset.")
    parser.add_argument("--dataset_type", type=str, choices=["ssnce", "torgo", ])
    parser.add_argument("--output_path", type=Path, help="Output csv folder")
    return parser.parse_args()


def get_vocab(df):
    vocabs = df["phone"].unique()
    index_to_vocab = dict(enumerate(sorted(vocabs)))
    vocab_to_index = {v: i for i, v in index_to_vocab.items()}
    return index_to_vocab, vocab_to_index


def _prepare_ssnce(ssnce_path: Path):
    labels = ("0_healthy", "1_mild", "2_moderate", "3_severe")
    rows = []
    for label in labels:
        index, label_name = label.split("_")
        index = int(index)

        for audio_path in tqdm((ssnce_path / label).glob("*.wav")):
            grid_path = audio_path.with_suffix(".TextGrid")
            grid = praatio.textgrid.openTextgrid(grid_path, includeEmptyIntervals=False)
            audio_len = librosa.get_duration(path=audio_path, sr=16000)
            for entry in grid.getTier("phonemes").entries:
                if entry.label in ("#", "SIL", ):
                    label = "(...)"
                else:
                    label = entry.label
                if entry.end <= audio_len:
                    rows.append({
                        "audio": audio_path,
                        "label": index,
                        "label_name": label_name,
                        "min": entry.start,
                        "max": entry.end,
                        "phone": label,
                    })
                else:
                    break
            rows[-1]["max"] = audio_len
    return pd.DataFrame(rows)


def _prepare_torgo(torgo_path: Path):
    labels = ("0_healthy", "1_mild", "2_moderate", "3_severe")
    rows = []
    for label in labels:
        index, label_name = label.split("_")
        index = int(index)

        for audio_path in tqdm((torgo_path / label).glob("*.wav")):
            grid_path = audio_path.with_suffix(".TextGrid")
            grid = praatio.textgrid.openTextgrid(grid_path, includeEmptyIntervals=False)

            for entry in grid.getTier("phones").entries:
                label = entry.label.strip()
                if "0" <= label[-1] <= "9":
                    label = label[:-1]
                if label in ("", "#", "@", "sil", "sp"):
                    label = "(...)"
                else:
                    label = label.upper()
                rows.append({
                    "audio": audio_path,
                    "label": index,
                    "label_name": label_name,
                    "min": entry.start,
                    "max": entry.end,
                    "phone": label,
                })

            audio_length = librosa.get_duration(path=audio_path, sr=16000)
            if entry.end < audio_length:
                rows.append({
                    "audio": audio_path,
                    "label": index,
                    "label_name": label_name,
                    "min": entry.end,
                    "max": audio_length,
                    "phone": "(...)",
                })
    return pd.DataFrame(rows)


# class PhoneRecognitionDataset(torch.utils.data.Dataset):
#     """
#     * What should be saved in CSV file?
#     We read the "audio" column and "label" column via pandas.read_csv.
#     "audio" column should contain the absolute path to the audio, and "label" should be integer: 0, 1, 2.
#     """

#     def __init__(
#         self,
#         df: pd.DataFrame,
#         sample_rate: int = 16000
#     ):
#         self.df = df
#         self.audios = df.audio.unique()
#         self.sample_rate = sample_rate

#     def __len__(self):
#         return len(self.audios)

#     def __getitem__(self, i):
#         x, _ = librosa.load(self.audios[i], sr=self.sample_rate, mono=True)
#         y = self.df[self.df["audio"] == self.audios[i]]

#         return x, y[["phone", "min", "max"]]


if __name__ == "__main__":
    args = _get_args()
    _prepare = {"ssnce": _prepare_ssnce, "torgo": _prepare_torgo, }[args.dataset_type]
    df = _prepare(args.dataset_path)
    csv_path = args.output_path / f"{args.dataset_type}.csv.gz"
    df.to_csv(csv_path, index=False, compression="gzip")
