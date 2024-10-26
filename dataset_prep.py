import argparse
import os
import json
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import praatio.textgrid
import kaldi_io
import torch
from tqdm import tqdm
import os
from datasets import load_dataset
from data.timit import TIMITConverter


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, help="Path to dataset")
    parser.add_argument("--dataset_type", type=str, choices=["ssnce", "torgo", "l2arctic", "speechocean762", "uaspeech", "timit"])
    parser.add_argument("--output_path", type=Path, help="Output csv folder")
    return parser.parse_args()


def get_vocab(df):
    vocabs = df["phone"].unique()
    index_to_vocab = dict(enumerate(sorted(vocabs)))
    vocab_to_index = {v: i for i, v in index_to_vocab.items()}
    return index_to_vocab, vocab_to_index


def _prepare_uaspeech(uaspeech_path: Path):
    label_info = {
        (0, "healthy"): "CF02 CF03 CF04 CM04 CM05 CM06 CM08 CM09 CM10 CM12 CM13".split(),
        (1, "verylow"): "F03 M01 M04 M12".split(),
        (2, "low"): "F02 M07 M16".split(),
        (3, "mid"): "F04 M05 M11".split(),
        (4, "high"): "M08 M09 M10 M14".split(),
    }
    speaker_info = {spk: label for label, spks in label_info.items() for spk in spks}

    rows = []
    for p in tqdm(uaspeech_path.glob("noisereduced/*/*.TextGrid")):
        if p.stem.split("_")[3] != "M6":
            continue
        grid = praatio.textgrid.openTextgrid(p, includeEmptyIntervals=True)
        audio_path = p.with_suffix(".wav")
        audio_len = librosa.get_duration(path=audio_path, sr=16000)

        speaker = p.parent.name
        index, label_name = speaker_info[speaker]
        split = "train" if index == 0 else "test"

        for entry in grid.getTier("phones").entries:
            label = entry.label
            if label in ("", "spn"):
                continue
            if "0" <= label[-1] <= "9":
                label = label[:-1]

            if entry.end <= audio_len:
                rows.append({
                    "audio": audio_path,
                    "label": index, # 0: healthy
                    "label_name": label_name,
                    "min": entry.start,
                    "max": entry.end,
                    "phone": label,
                    "split": split,
                })

    return pd.DataFrame(rows)


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


def _prepare_speechocean762(speechocean762_path: Path):
    def _read_scp(scp_path: Path):
        files = {}
        for line in open(scp_path).readlines():
            key, fname = line.strip().split("\t")
            files[key] = fname
        return files

    def _read_alignment(alignment_path: Path):
        original_working_dir = os.getcwd()
        os.chdir(alignment_path.parent.parent.parent)
        alignment_path = Path(*alignment_path.parts[2:])
        alignments = {
            key: alignment
            for key, alignment in kaldi_io.read_vec_int_ark(str(alignment_path))
        }
        os.chdir(original_working_dir)
        return alignments

    def _read_feat(feat_path: Path):
        original_working_dir = os.getcwd()
        os.chdir(feat_path.parent.parent.parent)
        feat_path = Path(*feat_path.parts[2:])
        feats = defaultdict(list)
        for key_index, feat in kaldi_io.read_vec_flt_scp(str(feat_path)):
            key, index = key_index.split(".")
            assert len(feats[key]) == int(index)
            feats[key].append(np.array(feat))
        os.chdir(original_working_dir)
        return feats

    def _remove_stray(p):
        p = "".join([c for c in p if c not in "0123456789"]).upper()
        return p


    meta = json.load(open(speechocean762_path / "resource" / "scores.json"))

    fnames = {}
    fnames.update(_read_scp(speechocean762_path / "train" / "wav.scp"))
    fnames.update(_read_scp(speechocean762_path / "test" / "wav.scp"))

    alignments = {}
    for alignment_path in speechocean762_path.glob("exp/ali_*/ali-phone.*.gz"):
        alignments.update(_read_alignment(alignment_path))

    feats = {}
    for feat_path in speechocean762_path.glob("exp/gop_*/feat.scp"):
        feats.update(_read_feat(feat_path))

    rows = []
    for key, scores in tqdm(meta.items()):
        audio = speechocean762_path / fnames[key]
        max_seconds = librosa.get_duration(path=audio)
        alignment = alignments[key]
        phones = [_remove_stray(p) for w in scores["words"] for p in w["phones"]]
        assert len(phones) == len(feats[key])

        # Calculate end_seconds
        labels, counts = torch.unique_consecutive(
            torch.LongTensor(alignment), return_counts=True)
        end_seconds = np.cumsum(counts.numpy()) / counts.sum().item() * max_seconds
        end_seconds = np.insert(end_seconds, 0, 0.0)

        phone_index = 0
        for index, label in enumerate(labels):
            if label != 1: # Ignore SIL
                rows.append({
                    "audio": speechocean762_path / fnames[key],
                    "label": scores["total"],
                    "label_name": str(scores["total"]),
                    "min": end_seconds[index],
                    "max": end_seconds[index + 1],
                    "phone": phones[phone_index],
                    "split": "train" if scores["total"] >= 9.0 else "test",
                    "feat": feats[key][phone_index],
                })
                phone_index += 1
        assert phone_index == len(phones)

    return pd.DataFrame(rows)


def _prepare_timit(timit_path: Path):
    timit = load_dataset('timit_asr', data_dir=timit_path, trust_remote_code=True)
    converter = TIMITConverter(timit_mapping_path='data')

    rows = []
    for split in ['train', 'test']:
        for utterance in tqdm(timit[split]):
            audio_path = utterance['audio']['path']
            speaker = utterance['speaker_id']
            alignment = utterance['phonetic_detail']
            for phn, start, stop in zip(alignment['utterance'], alignment['start'], alignment['stop']):
                # phn could be mapped to sil, which is removed
                arpa_phones = converter.convert(phn)
                if len(arpa_phones) == 0:
                    continue
                arpa_phone = arpa_phones[0]

                rows.append({
                    "audio": audio_path,
                    "speaker": speaker,
                    "min": start / 16000,
                    "max": stop / 16000,
                    "phone": arpa_phone,
                    "split": split
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    args = _get_args()
    _prepare = {
        "ssnce": _prepare_ssnce,
        "torgo": _prepare_torgo,
        "l2arctic": _prepare_l2arctic,
        "speechocean762": _prepare_speechocean762,
        "uaspeech": _prepare_uaspeech,
        "timit": _prepare_timit,
    }[args.dataset_type]
    df = _prepare(args.dataset_path)
    csv_path = args.output_path / f"{args.dataset_type}.original.pkl"
    df.to_pickle(csv_path)
