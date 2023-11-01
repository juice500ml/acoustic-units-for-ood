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
    parser.add_argument("--dataset_path", type=Path, help="Path to dataset")
    parser.add_argument("--dataset_type", type=str, choices=["ssnce", "torgo", "l2arctic", ])
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
        split = "train" if index == 0 else "test"

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
                        "split": split,
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
        split = "train" if index == 0 else "test"

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
                    "split": split,
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
                    "split": split,
                })
    return pd.DataFrame(rows)


def _prepare_l2arctic(l2arctic_path: Path):
    rows = []
    def _remove_stray(p):
        p = "".join([c for c in p if c not in "0123456789 _)`"]).upper()
        if p in ("SIL", "SP", "SPN"):
            p = "(...)"
        return p

    def _clean_phone(p):
        p = p.split(",")
        return _remove_stray(p[0]), int(len(p) != 1)

    for grid_path in l2arctic_path.glob("*/annotation/*.TextGrid"):
        audio_path = str(grid_path).replace("/annotation/", "/wav/").replace(".TextGrid", ".wav")
        grid = praatio.textgrid.openTextgrid(grid_path, includeEmptyIntervals=True)
        speaker = grid_path.parent.parent.name
        if speaker == "suitcase_corpus":
            continue
        if speaker in ("NJS", "TLV", "TNI", "ZHAA", "TXHC", "YKWK"):
            split = "test"
        else:
            split = "train"
        for entry in grid.getTier("phones").entries:
            emin = entry.xmin if hasattr(entry, "xmin") else entry.start
            emax = entry.xmax if hasattr(entry, "xmax") else entry.end
            phone = entry.text if hasattr(entry, "text") else entry.label
            phone, label = _clean_phone(phone)
            if not phone:
                continue
            if label == 1 and split == "train":
                split = "train_excluded"
            rows.append({
                "audio": audio_path,
                "speaker": speaker,
                "split": split,
                "min": emin,
                "max": emax,
                "phone": phone,
                "label": label,
                "label_name": ["correct", "wrong"][label],
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    args = _get_args()
    _prepare = {"ssnce": _prepare_ssnce, "torgo": _prepare_torgo, "l2arctic": _prepare_l2arctic, }[args.dataset_type]
    df = _prepare(args.dataset_path)
    csv_path = args.output_path / f"{args.dataset_type}.csv.gz"
    df.to_csv(csv_path, index=False, compression="gzip")
