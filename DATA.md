# Data preparation

## Expected inputs

The model operates on **modulated grey-matter density maps** (`mwp1`) produced
by CAT12, in MNI152 space:

| property | value |
|---|---|
| resolution | 1.5 mm isotropic |
| shape | 113 × 137 × 113 |
| value range | approximately [0, 1.2] |
| CAT12 version | 12.8.1 (cross-sectional pipeline) |

Because the maps are modulated, summing intensities within a region gives that
region's grey-matter volume.

Volumes are zero-padded to 120 × 144 × 120 inside the loader so each dimension
divides by 8 for the three downsampling stages, and cropped back before saving.
No resampling or interpolation is applied.

---

## Directory layout

```
<data_root>/
    pairs_tier1.csv      one row per baseline/follow-up pair
    scans/               one .nii.gz per session, shared across pairs
```

Each scan is stored **once** in `scans/` and referenced by every pair it
participates in. The same M00 scan is the baseline for a subject's 24, 36
and 48 month pairs. The change map is computed on the fly as
`target − input`.

Only the basename is taken from `input_file` and `target_file` and joined to
`scans_dir`, so a CSV written on Windows works unchanged on a cluster.

---

## `pairs_tier1.csv`

| column | meaning |
|---|---|
| `pair_id` | unique identifier for the pair |
| `subject` | subject identifier; the split is by subject |
| `split` | `train`, `val` or `test` |
| `dt_months` | follow-up interval in months |
| `input_file` | baseline scan |
| `target_file` | follow-up scan |
| `dx_code` | 0 = CN, 1 = MCI, 2 = Dementia |
| `apoe4` | **number** of APOE4 alleles: 0, 1 or 2, not a binary flag |
| `age_at_base_norm` | baseline age, z-scored (see below) |

The covariate vector passed to the network is six-dimensional:

```
[CN, MCI, Dementia, apoe4, age_at_base_norm, dt_months / 48]
```

The first three are a one-hot encoding of `dx_code`. The interval enters as a
continuous value, which is what allows one model to cover every horizon, including intervals never seen during training.

---

## Normalisation constants

Two constants are applied by the loader. **Neither should be re-estimated on a
new dataset if you are using weights trained on ADNI.**

**`CHANGE_SCALE = 0.22`** is the median 99.5th percentile of the absolute change
over the ADNI training set. The change map is divided by it during training,
and inference reconstructs the follow-up as
`baseline + CHANGE_SCALE × output`. A different constant rescales every
prediction, with no error and no visible symptom. It is read from the
`CHANGE_SCALE` environment variable and defaults to 0.22.

The roughly 0.5% of voxels that overshoot [−1, 1] after scaling are kept rather
than clipped, because they carry the strongest atrophy signal.

**Age normalisation.** `age_at_base_norm` was z-scored with the ADNI training
mean of 74.5186 years and standard deviation of 6.6485:

```
age_at_base_norm = (age_at_base − 74.5186) / 6.6485
```

Re-normalising on another cohort would place ages on a scale the model has
never seen. If a new cohort is younger or older than ADNI, its normalised ages
should sit systematically below or above zero. That shift is real information and should not be removed.

---

## Splitting

The split is by **subject**, never by pair. One subject can contribute many
pairs, and mixing them across splits would leak. The training configs assume
the split is already encoded in the `split` column.

---

## Applying the model to a new cohort

The geometry and processing must match. Verify before running inference:

```python
import nibabel as nib
img = nib.load("<one scan>.nii.gz")
print(img.shape, img.header.get_zooms())
d = img.get_fdata()
print(d.min(), d.max(), d.mean())
```

Expect `(113, 137, 113)`, `(1.5, 1.5, 1.5)`, and a maximum near 1.2. A
mismatch means the data was processed differently and the model's inputs would
be out of distribution.

Intervals outside the 24–120 month training range are accepted, since `dt` is continuous, but should be reported separately, since predictions there are
extrapolation rather than interpolation.
