import argparse
import pickle
from pathlib import Path

import librosa
import pandas as pd
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from tqdm import tqdm


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/wavlm-large", help="Huggingface model name.")
    parser.add_argument("--dataset_csv", type=Path, help="Dataset to inference")
    parser.add_argument("--output_path", type=Path, help="Output pkl path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _get_args()

    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)

    df = pd.read_csv(args.dataset_csv)
    data = {}
    for path in tqdm(df.audio.unique()):
        x, _ = librosa.load(path, sr=16000, mono=True)
        outputs = model(**processor(raw_speech=[x], sampling_rate=16000, padding=False, return_tensors="pt"))
        data[path] = outputs.last_hidden_state.detach().numpy()[0]

    with open(args.output_path, "wb") as f:
        pickle.dump(data, f)
