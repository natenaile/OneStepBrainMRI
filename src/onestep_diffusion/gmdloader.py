import os
import csv
import random

import nibabel
import numpy as np
import torch
from torch.utils.data import Dataset

import time as _time
def _robust_load(path, attempts=5, wait=2.0):

    last = None
    for i in range(attempts):
        try:
            return nibabel.load(str(path))
        except Exception as e:
            last = e
            print(f"[loader] read failed ({type(e).__name__}) on {path} "
                  f"- retry {i+1}/{attempts}", flush=True)
            _time.sleep(wait * (i + 1))
    raise last

EPS = 1e-6

# Taken from the BASELINE, never the target: the target does not exist
# at inference time.
MASK_THRESHOLD = 0.10

NATIVE_SHAPE = (113, 137, 113)
PAD_SHAPE    = (120, 144, 120)

import os as _os
# 0.22 = median 99.5th percentile of |change| over the training set.
# Must match at inference: future = baseline + CHANGE_SCALE * output.
# The ~0.5% of voxels that overshoot [-1,1] are kept, not clipped -
# they carry the strongest atrophy signal.
CHANGE_SCALE = float(_os.environ.get("CHANGE_SCALE", "0.22"))

DT_REF = float(_os.environ.get("DT_REF", "48.0"))

FLIP_AXIS = 0

COND_VEC_DIM = 6


def to_minus1_plus1(arr):

    return arr * 2.0 - 1.0


def load_nifti_volume(path):

    return _robust_load(path).get_fdata().astype(np.float32)


def pad_to(arr, shape=PAD_SHAPE):

    out = np.zeros(shape, dtype=arr.dtype)
    d, h, w = arr.shape
    pd, ph, pw = [(shape[i] - arr.shape[i]) // 2 for i in range(3)]
    out[pd:pd + d, ph:ph + h, pw:pw + w] = arr
    return out


def crop_from_pad(arr, native=NATIVE_SHAPE, padded=PAD_SHAPE):

    pd, ph, pw = [(padded[i] - native[i]) // 2 for i in range(3)]
    return arr[pd:pd + native[0], ph:ph + native[1], pw:pw + native[2]]


def load_pairs_csv(csv_path, split, scans_dir, intervals=None,
                   exclude_subjects=None, zero_cond_slots=None):

    out, skipped, seen_dt, excluded = [], 0, {}, 0
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] != split:
                continue
            dt = int(float(r["dt_months"]))
            if intervals is not None and dt not in intervals:
                skipped += 1
                continue
            if exclude_subjects and r["subject"] in exclude_subjects:
                excluded += 1
                continue

            inp = os.path.join(scans_dir, os.path.basename(r["input_file"].replace("\\", "/")))
            tgt = os.path.join(scans_dir, os.path.basename(r["target_file"].replace("\\", "/")))

            dx = int(float(r["dx_code"]))
            onehot = [0.0, 0.0, 0.0]
            onehot[dx] = 1.0
            vec = onehot + [
                float(r["apoe4"]),
                float(r["age_at_base_norm"]),
                float(r["dt_months"]) / DT_REF,
            ]
            if zero_cond_slots:
                for i in zero_cond_slots:
                    vec[i] = 0.0

            out.append({
                "folder_name": f'{r["pair_id"]}_{r["subject"]}_'
                               f'{r["input_session"]}_to_{r["target_session"]}',
                "input_file":  inp,
                "target_file": tgt,
                "change_file": None,
                "dt_months":   dt,
                "cond_vec":    np.asarray(vec, dtype=np.float32),
            })
            seen_dt[dt] = seen_dt.get(dt, 0) + 1
    return out, skipped, seen_dt, excluded


class GMDDataset(Dataset):

    def __init__(
        self,
        test_flag=False,
        regression=False,
        single_contrast=True,
        device="cluster",
        targets_dict=None,
        type_slices="all",
        list_patients="",
        random_seed=42,
        conditioned=True,
        tier1_csv=None,

        pairs_csv=None,
        split=None,
        scans_dir=None,
        intervals=None,
        exclude_subjects_csv=None,
        exclude_split="train",
        zero_cond_slots=None,
        augment_flip=False,
        predict_target=False,
    ):
        super().__init__()
        self.test_flag = test_flag

        self.augment_flip = bool(augment_flip) and not test_flag

        self.predict_target = bool(predict_target)

        random.seed(random_seed)

        if pairs_csv:
            if split is None:
                raise ValueError("[GMDDataset] CSV mode needs split=train/val/test")
            pairs_csv = os.path.expanduser(pairs_csv)
            if not os.path.exists(pairs_csv):
                raise FileNotFoundError(f"[GMDDataset] pairs csv not found: {pairs_csv}")
            if scans_dir is None:
                scans_dir = os.path.join(os.path.dirname(os.path.abspath(pairs_csv)), "scans")
            if isinstance(intervals, str) and intervals.strip():
                intervals = [int(x) for x in intervals.split(",")]
            if intervals is not None:
                intervals = set(int(x) for x in intervals)

            excl = None
            if exclude_subjects_csv:
                excl = set()
                with open(os.path.expanduser(exclude_subjects_csv), newline="") as _f:
                    for _r in csv.DictReader(_f):
                        if _r.get("split") == exclude_split:
                            excl.add(_r["subject"])
                print(f"[GMDDataset] excluding {len(excl)} subjects from "
                      f"{exclude_subjects_csv} (split={exclude_split})")
            if isinstance(zero_cond_slots, str) and zero_cond_slots.strip():
                zero_cond_slots = [int(x) for x in zero_cond_slots.split(",")]

            self.database, skipped, seen_dt, n_excl = load_pairs_csv(
                pairs_csv, split, scans_dir, intervals, excl, zero_cond_slots)
            if not self.database:
                raise RuntimeError(
                    f"[GMDDataset] no pairs for split={split} in {pairs_csv}")

            missing = [e for e in self.database[:20]
                       if not os.path.exists(e["input_file"])]
            if missing:
                raise FileNotFoundError(
                    f"[GMDDataset] scans not found under {scans_dir}, e.g. "
                    f"{missing[0]['input_file']}")

            print(f"[GMDDataset] CSV mode: {len(self.database)} pairs "
                  f"(split={split}) from {pairs_csv}")
            print(f"[GMDDataset]   scans_dir = {scans_dir}")
            print(f"[GMDDataset]   per-interval: "
                  f"{{{', '.join(f'{k}:{v}' for k, v in sorted(seen_dt.items()))}}}")
            if skipped:
                print(f"[GMDDataset]   skipped {skipped} pairs outside "
                      f"intervals={sorted(intervals)}")
            if n_excl:
                print(f"[GMDDataset]   dropped {n_excl} pairs from excluded "
                      f"subjects (contamination guard)")
            if zero_cond_slots:
                print(f"[GMDDataset]   cond_vec slots {zero_cond_slots} forced "
                      f"to 0.0")
            print(f"[GMDDataset]   cond_vec = [CN,MCI,Dem,apoe4,age_norm,"
                  f"dt/{DT_REF:g}]  CHANGE_SCALE={CHANGE_SCALE}")
            print(f"[GMDDataset]   augment_flip = {self.augment_flip} "
                  f"(L-R flip on axis {FLIP_AXIS}, p=0.5)")
            return

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        e = self.database[idx]

        do_flip = bool(self.augment_flip and torch.rand(()).item() < 0.5)

        # Applied to the NATIVE array before pad_to: padding is asymmetric
        # (3 voxels one side, 4 the other), so flipping the padded array
        # shifts the volume one voxel relative to the crop window, silently.
        def _aug(a):
            return np.ascontiguousarray(np.flip(a, axis=FLIP_AXIS)) if do_flip else a

        base_arr = pad_to(_aug(load_nifti_volume(e["input_file"])))
        gm_mask  = torch.tensor((base_arr > MASK_THRESHOLD),
                                dtype=torch.float32).unsqueeze(0)
        baseline = torch.tensor(to_minus1_plus1(base_arr),
                                dtype=torch.float32).unsqueeze(0)

        if e["change_file"] is not None:
            change_arr = pad_to(_aug(load_nifti_volume(e["change_file"])))
        elif e.get("target_file") and os.path.exists(e["target_file"]):
            tgt_arr = pad_to(_aug(load_nifti_volume(e["target_file"])))
            change_arr = tgt_arr - base_arr
        else:
            change_arr = None

        if self.predict_target:

            if change_arr is not None:
                tgt_native = base_arr + change_arr
            else:
                tgt_native = base_arr
            change = torch.tensor(to_minus1_plus1(tgt_native),
                                  dtype=torch.float32).unsqueeze(0)
        elif change_arr is not None:
            change = torch.tensor(change_arr / CHANGE_SCALE,
                                  dtype=torch.float32).unsqueeze(0)
        else:
            change = torch.zeros_like(baseline)

        cond_vec = torch.tensor(e["cond_vec"], dtype=torch.float32)

        out_dict = {
            "y": 1,
            "cond_image":  baseline,
            "cond_vec":    cond_vec,
            "input_slice": baseline,
            "gm_mask":     gm_mask,
        }

        image = change
        label = change
        return image, out_dict, 1, label, e["folder_name"]

    def get_patient_numbers(self):
        return [e["folder_name"] for e in self.database]

    def get_id(self, x):
        return self.database[x]["folder_name"]
