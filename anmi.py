import argparse
from itertools import product
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


_natural_classes = {
    'aa': ('open_back_unrounded', 'open_back_unrounded'),
    'ae': ('open_front_unrounded', 'open_front_unrounded'),
    'ao': ('open-mid_back_rounded', 'open-mid_back_rounded'),
    'ah': ('open-mid_back_unrounded', 'open-mid_back_unrounded'),
    'aw': ('open_front_unrounded', 'near-close_near-back_rounded'),
    'ay': ('open_front_unrounded', 'near-close_near-front_unrounded'),
    'b': ('bilabial_plosive', 'bilabial_plosive'),
    'ch': ('palato-alveolar_sibilant-affricate', 'palato-alveolar_sibilant-affricate'),
    'd': ('alveolar_plosive', 'alveolar_plosive'),
    'dh': ('dental_non-sibilant-fricative', 'dental_non-sibilant-fricative'),
    'dx': ('alveolar_flap', 'alveolar_flap'),
    'eh': ('close-mid_front_unrounded', 'close-mid_front_unrounded'),
    'er': ('open-mid_central_unrounded', 'open-mid_central_unrounded'),
    'ey': ('close-mid_central_unrounded', 'near-close_near-front_unrounded'),
    'f': ('labio-dental_non-sibilant-fricative', 'labio-dental_non-sibilant-fricative'),
    'g': ('velar_plosive', 'velar_plosive'),
    'hh': ('glottal_non-sibilant-fricative', 'glottal_non-sibilant-fricative'),
    'ih': ('close_front_unrounded', 'close_front_unrounded'),
    'iy': ('close_front_unrounded', 'close_front_unrounded'),
    'jh': ('palato-alveolar_sibilant-affricate', 'palato-alveolar_sibilant-affricate'),
    'k': ('velar_plosive', 'velar_plosive'),
    'l': ('alveolar_lateral-approximant', 'alveolar_lateral-approximant'),
    'm': ('bilabial_nasal', 'bilabial_nasal'),
    'n': ('alveolar_nasal', 'alveolar_nasal'),
    'ng': ('velar_nasal', 'velar_nasal'),
    'ow': ('close-mid_back_rounded', 'near-close_near-back_rounded'),
    'oy': ('close-mid_back_rounded', 'near-close_near-front_unrounded'),
    'p': ('bilabial_plosive', 'bilabial_plosive'),
    'r': ('alveolar_approximant', 'alveolar_approximant'),
    's': ('alveolar_sibilant-fricative', 'alveolar_sibilant-fricative'),
    'sh': ('palato-alveolar_sibilant-fricative', 'palato-alveolar_sibilant-fricative'),
    't': ('alveolar_plosive', 'alveolar_plosive'),
    'th': ('dental_non-sibilant-fricative', 'dental_non-sibilant-fricative'),
    'uh': ('near-close_near-back_rounded', 'near-close_near-back_rounded'),
    'uw': ('close_back_rounded', 'close_back_rounded'),
    'v': ('labio-dental_non-sibilant-fricative', 'labio-dental_non-sibilant-fricative'),
    'w': ('labio-velar_approximant', 'labio-velar_approximant'),
    'y': ('palatal_approximant', 'palatal_approximant'),
    'z': ('alveolar_sibilant-fricative', 'alveolar_sibilant-fricative'),
}


def hubert_metrics(phone, cluster):
    uniq_phone = np.unique(phone)
    phone_to_idx = {p: i for i, p in enumerate(uniq_phone)}

    uniq_cluster = np.unique(cluster)
    cluster_to_idx = {c: i for i, c in enumerate(uniq_cluster)}

    # joint distrb
    joint = np.zeros((len(uniq_phone), len(uniq_cluster)))
    for p, c in zip(phone, cluster):
        joint[phone_to_idx[p]][cluster_to_idx[c]] += 1
    joint /= len(phone)

    pz = joint.sum(axis=0)
    py = joint.sum(axis=1)
    z_star = np.argmax(joint, axis=1)
    y_star = np.argmax(joint, axis=0)

    # phone purity
    pp = sum(
        joint[y_star[j]][j]
        for j in range(len(uniq_cluster))
    )

    # cluster purity
    cp = sum(
        joint[i][z_star[i]]
        for i in range(len(uniq_phone))
    )

    # pnmi
    mi = sum(
        joint[i][j] * np.log(joint[i][j] / (py[i] * pz[j]))
        for i, j in product(range(len(uniq_phone)), range(len(uniq_cluster)))
        if joint[i][j] > 0
    )
    hy = sum(
        -py[i] * np.log(py[i])
        for i in range(len(uniq_phone))
        if py[i] > 0
    )
    pnmi = mi / hy

    return pp, cp, pnmi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_pkl", type=Path, help="Dataset to evaluate")
    parser.add_argument("--n_clusters", type=int, default=32, help="k in kmeans")
    parser.add_argument("--natural_class", action=argparse.BooleanOptionalAction, help="Use natural classes as environment. If not, use phoneme.")
    parser.add_argument("--allophone", action=argparse.BooleanOptionalAction, help="Directly use allophone.")
    args = parser.parse_args()

    df = pd.read_pickle(args.dataset_pkl)

    if "torgo" in args.dataset_pkl.name:
        _vc = df.phone.value_counts()
        vocab = set(_vc[_vc >= 100].keys()) - set(["(...)"])
        df = df[df.phone.isin(vocab) & (df.split == "train")].copy()
        df.phone = df.phone.apply(lambda x: x.lower())

    indices, envs = [], []

    if args.allophone:
        df["env"] = df["allophone"]
    else:
        for _, _df in df.groupby("audio"):
            _df = _df.sort_values("min")
            index, phone = _df.index.tolist(), _df.phone.tolist()

            indices.append(index[0])
            if args.natural_class:
                envs.append(f"word-initial+{_natural_classes[phone[1]][0]}")
            else:
                envs.append(f"sil_{phone[1]}")
            for i in range(1, len(phone) - 1):
                indices.append(index[i])
                if args.natural_class:
                    envs.append(f"{_natural_classes[phone[i-1]][1]}+{_natural_classes[phone[i+1]][0]}")
                else:
                    envs.append(f"{phone[i-1]}_{phone[i+1]}")
            indices.append(index[-1])
            if args.natural_class:
                envs.append(f"{_natural_classes[phone[-2]][1]}_word-final")
            else:
                envs.append(f"{phone[-2]}_sil")

        df["env"] = None
        df.loc[indices, "env"] = envs

    scores = []
    for p in df.phone.unique():

        _df = df[df.phone == p]
        x = _df.feat.tolist()
        y = _df.env.to_numpy()

        km = KMeans(n_clusters=args.n_clusters)
        c = km.fit_predict(x)
        scores.append(hubert_metrics(y, c))
    scores = np.array(scores)

    print(scores.mean(0).tolist())
