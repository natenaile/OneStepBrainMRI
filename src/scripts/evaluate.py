import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from skimage.metrics import structural_similarity as ssim_fn

REGIONS = {
    "Entorhinal cortex":       [116, 117],
    "Hippocampus":              [47, 48],
    "Parahippocampal gyrus":  [170, 171],
    "Amygdala":                 [31, 32],
    "Posterior cingulate":    [166, 167],
    "Precuneus":              [168, 169],
    "Thalamus":                 [59, 60],
}

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True,
                help="directory of predictions written by the sampler")
ap.add_argument("--csv", required=True,
                help="pairs_tier1.csv for the dataset that was evaluated")
ap.add_argument("--scans", required=True,
                help="directory holding the .nii.gz scans")
ap.add_argument("--atlas", required=True,
                help="Neuromorphometrics atlas at 1.5 mm, matching the "
                     "data geometry (113x137x113)")
ap.add_argument("--run", default="", help="label for the printed header")
ap.add_argument("--max_pairs", type=int, default=0)
a = ap.parse_args()

PRED = Path(a.pred)
THR = 0.10

atlas = nib.load(a.atlas).get_fdata().astype(int)
rois = {k: np.isin(atlas, v) for k, v in REGIONS.items()}

df = pd.read_csv(a.csv)
df = df[df.split == "test"].copy()
if a.max_pairs:
    df = df.head(a.max_pairs)

scan = lambda p: nib.load(str(Path(a.scans) / Path(str(p).replace("\\", "/")).name)
                          ).get_fdata(dtype=np.float32)


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else np.nan

rows = []
for i, (_, r) in enumerate(df.iterrows()):
    hits = list(PRED.glob(f"{r.pair_id}_*"))
    f = hits[0] / "predicted_change_mwp1sub.nii.gz" if hits else None
    if f is None or not f.exists():
        continue
    base = scan(r.input_file)
    true = scan(r.target_file)
    pchg = nib.load(str(f)).get_fdata().astype(np.float32)
    pred = base + pchg
    tchg = true - base

    gm = base > THR
    dr = float(true[gm].max() - true[gm].min())
    d = {"dt": int(r.dt_months), "subj": r.subject}

    d["dP"]          = pearson(pchg[gm], tchg[gm])
    d["mse"]         = float(((pchg - tchg)[gm] ** 2).mean())
    d["mae"]         = float(np.abs(pchg - tchg)[gm].mean())
    d["pearson_img"] = pearson(pred[gm], true[gm])
    d["psnr"]        = float(10 * np.log10(dr ** 2 / max(d["mse"], 1e-12)))
    d["ssim_gm"]     = float(ssim_fn(true * gm, pred * gm, data_range=dr))

    for name, roi in rois.items():
        m = roi & gm
        if int(m.sum()) < 50:
            for k in ("dP", "mse", "mae"):
                d[f"{k}|{name}"] = np.nan
            continue
        p, t = pchg[m], tchg[m]
        d[f"dP|{name}"]  = pearson(p, t)
        d[f"mse|{name}"] = float(((p - t) ** 2).mean())
        d[f"mae|{name}"] = float(np.abs(p - t).mean())

    rows.append(d)
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(df)}")

M = pd.DataFrame(rows)
if M.empty:
    raise SystemExit(f"no predictions found under {PRED}")
print(f"\nn = {len(M)} pairs from {M.subj.nunique()} subjects")
DTS = sorted(M.dt.unique())
SCALE = {"mse": 1e3, "mae": 1e3}


def cell(g, col, nd):
    v = g[col].values * SCALE.get(col.split("|")[0], 1.0)
    return f"{v.mean():.{nd}f} ± {v.std():.{nd}f}"

GLOBAL = [("Delta-Pearson", "dP", 3), ("Pearson (img)", "pearson_img", 3),
          ("MSE x1e-3", "mse", 3), ("MAE x1e-3", "mae", 3),
          ("PSNR (dB)", "psnr", 3), ("SSIM (GM)", "ssim_gm", 3)]

B = "=" * 108
print(f"\n{B}\nTABLE 1  -  GLOBAL, BY INTERVAL   (mean ± SD)\n{B}")
hdr = ["Interval"] + [n for n, _, _ in GLOBAL]
out = []
for dt in DTS:
    g = M[M.dt == dt]
    out.append([f"{dt//12} y"] + [cell(g, c, nd) for _, c, nd in GLOBAL])
out.append(["All"] + [cell(M, c, nd) for _, c, nd in GLOBAL])
T1 = pd.DataFrame(out, columns=hdr)
print(T1.to_string(index=False))
T1.to_csv(PRED / "table1_global.csv", index=False)

REG = [("Delta-Pearson", "dP", 3), ("MSE x1e-3", "mse", 3), ("MAE x1e-3", "mae", 3)]
for j, name in enumerate(REGIONS, start=2):
    print(f"\n{B}\nTABLE {j}  -  {name.upper()}   (change space, mean ± SD)\n{B}")
    out = []
    for dt in DTS:
        g = M[M.dt == dt]
        out.append([f"{dt//12} y"] + [cell(g, f"{c}|{name}", nd) for _, c, nd in REG])
    out.append(["All"] + [cell(M, f"{c}|{name}", nd) for _, c, nd in REG])
    T = pd.DataFrame(out, columns=["Interval"] + [n for n, _, _ in REG])
    print(T.to_string(index=False))
    slug = name.lower().replace(" ", "_").replace(".", "")
    T.to_csv(PRED / f"table{j}_{slug}.csv", index=False)

M.to_csv(PRED / "per_pair_all.csv", index=False)
print(f"\nsaved {1+len(REGIONS)} tables -> {PRED}/table*.csv")
print(f"per-pair values              -> {PRED}/per_pair_all.csv")
