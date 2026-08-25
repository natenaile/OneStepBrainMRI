from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import nibabel as nib
import numpy as np
import torch as th
import yaml

sys.path.append(".")
sys.path.append("..")

from onestep_diffusion import logger
from onestep_diffusion.conditioned_model import ConditionedModel
from onestep_diffusion.gmdloader import (
    CHANGE_SCALE,
    MASK_THRESHOLD,
    NATIVE_SHAPE,
    PAD_SHAPE,
    GMDDataset,
    crop_from_pad,
    pad_to,
)
from utils.script_util_gmd import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def _select_device() -> th.device:
    if th.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        th.cuda.set_device(local_rank)
        return th.device(f"cuda:{local_rank}")
    return th.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _extract_state_dict(obj: Any) -> Mapping[str, th.Tensor]:

    if isinstance(obj, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            value = obj.get(key)
            if isinstance(value, Mapping) and value:
                obj = value
                break

    if not isinstance(obj, Mapping) or not obj:
        raise TypeError("Checkpoint does not contain a non-empty state_dict mapping.")

    state: Dict[str, th.Tensor] = {}
    for key, value in obj.items():
        if not isinstance(value, th.Tensor):
            continue
        clean_key = key[7:] if key.startswith("module.") else key
        state[clean_key] = value

    if not state:
        raise TypeError("No tensor parameters were found in the checkpoint.")
    return state


def _atomic_save_nifti(data: np.ndarray, reference: nib.Nifti1Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(data.astype(np.float32, copy=False), reference.affine, header)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".nii.gz", dir=str(destination.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        nib.save(image, str(temporary))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_pickle(obj: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_name, destination)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _append_jsonl(destination: Path, record: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(record), sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _validate_args(args: argparse.Namespace) -> None:
    required_paths = {
        "tier1_csv": Path(args.tier1_csv),
        "model_path": Path(args.model_path),
    }
    for name, path in required_paths.items():
        if not str(path):
            raise ValueError(f"{name} is empty.")
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    expected_shape = (int(args.image_size_d), int(args.image_size_h), int(args.image_size_w))
    if expected_shape != tuple(PAD_SHAPE):
        raise ValueError(
            f"Inference shape {expected_shape} does not match loader PAD_SHAPE={PAD_SHAPE}."
        )
    if int(args.cond_vec_dim) != 6:
        raise ValueError(f"Expected cond_vec_dim=6, got {args.cond_vec_dim}.")
    if bool(args.learn_sigma):
        raise ValueError("The trained checkpoint uses learn_sigma=false.")
    if int(args.generation_number) < 1:
        raise ValueError("generation_number must be >= 1.")
    if int(args.max_samples) < 0:
        raise ValueError("max_samples must be >= 0.")


def _required_outputs(output_dir: Path, folder_name: str, n_gen: int) -> Iterable[Path]:
    pair_dir = output_dir / folder_name
    yield pair_dir / "predicted_change_mwp1sub.nii.gz"
    yield pair_dir / "predicted_target_mwp1sub.nii.gz"
    if n_gen > 1:
        yield pair_dir / "uncertainty_mwp1sub.nii.gz"


def _already_complete(output_dir: Path, folder_name: str, n_gen: int) -> bool:
    return all(path.is_file() and path.stat().st_size > 0
               for path in _required_outputs(output_dir, folder_name, n_gen))


def _tensor_stats(x: th.Tensor) -> Dict[str, float]:
    y = x.detach().float()
    return {
        "min": float(y.min().item()),
        "max": float(y.max().item()),
        "mean": float(y.mean().item()),
        "std": float(y.std().item()),
    }


def main() -> None:
    args = create_argparser().parse_args()
    _validate_args(args)

    device = _select_device()
    output_dir = Path(args.results_dir).expanduser().resolve() / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.configure(str(output_dir))

    n_gen = int(args.generation_number) if bool(args.probability_map) else 1
    shape = (int(args.image_size_d), int(args.image_size_h), int(args.image_size_w))
    progress_path = output_dir / "progress.jsonl"
    failure_path = output_dir / "failures.jsonl"

    logger.log(f"Device: {device}")
    logger.log(f"Output directory: {output_dir}")
    logger.log(f"Sampling shape: {shape}; generations per pair: {n_gen}")
    logger.log(
        f"diffusion_steps={args.diffusion_steps}, "
        f"timestep_respacing={args.timestep_respacing!r}, "
        f"clip_denoised={args.clip_denoised}, eta={args.eta}"
    )

    dataset = GMDDataset(
        test_flag=True,
        single_contrast=True,
        device=args.device,
        conditioned=True,
        tier1_csv=args.tier1_csv or None,
        pairs_csv=args.pairs_csv,
        split=getattr(args, "eval_split", "") or "test",
        scans_dir=getattr(args, "scans_dir", "") or None,
        intervals=getattr(args, "eval_intervals", "") or None,
        exclude_subjects_csv=getattr(args, "exclude_subjects_csv", "") or None,
        exclude_split=getattr(args, "exclude_split", "") or "train",
        zero_cond_slots=getattr(args, "zero_cond_slots", "") or None,
        predict_target=bool(getattr(args, "predict_target", False)),
    )

    if len(dataset) == 0:
        raise RuntimeError("The inference dataset contains zero valid pairs.")

    requested = len(dataset) if int(args.max_samples) == 0 else min(int(args.max_samples), len(dataset))
    logger.log(f"Dataset pairs: {len(dataset)}; selected for this run: {requested}")

    dataloader = th.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
    )

    model_kwargs_config = args_to_dict(args, model_and_diffusion_defaults().keys())
    model_kwargs_config["cond_channels"] = 1
    base_model, diffusion = create_model_and_diffusion(**model_kwargs_config)
    model = ConditionedModel(base_model)

    checkpoint_path = Path(args.model_path).expanduser().resolve()
    logger.log(f"Loading checkpoint strictly: {checkpoint_path}")
    checkpoint_obj = th.load(str(checkpoint_path), map_location="cpu")
    state_dict = _extract_state_dict(checkpoint_obj)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint/model mismatch. Verify num_channels=96, num_res_blocks=1, "
            "channel_mult=1,2,4,8, cond_vec_dim=6, class_cond=true, and "
            "learn_sigma=false. Original error:\n" + str(exc)
        ) from exc

    model.to(device)
    if bool(args.use_fp16):
        if device.type != "cuda":
            raise RuntimeError("use_fp16=true requires CUDA.")
        model.convert_to_fp16()
    model.eval()

    parameter_count = sum(p.numel() for p in model.parameters())
    logger.log(f"Strict checkpoint load passed; parameters: {parameter_count:,}")
    logger.log(f"Effective reverse steps: {diffusion.num_timesteps}")

    run_metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "device": str(device),
        "torch_version": th.__version__,
        "cuda_version": th.version.cuda,
        "dataset_size": len(dataset),
        "selected_samples": requested,
        "effective_diffusion_steps": int(diffusion.num_timesteps),
        "generation_number": n_gen,
        "change_scale": float(CHANGE_SCALE),
        "mask_threshold": float(MASK_THRESHOLD),
        "native_shape": list(NATIVE_SHAPE),
        "padded_shape": list(PAD_SHAPE),
        "config": vars(args),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, sort_keys=True, default=str)

    def model_fn(x: th.Tensor, t: th.Tensor, **kwargs: Any) -> th.Tensor:
        return model(x, t, **kwargs)

    completed_this_run = 0
    skipped = 0
    failed = 0
    measured_times = []
    compact_index: Dict[str, Dict[str, Any]] = {}

    _entry_by_folder = {e["folder_name"]: e
                        for e in getattr(dataset, "database", [])}

    def _load_baseline_native(folder_name):

        _e = _entry_by_folder.get(folder_name)
        if not (_e and _e.get("input_file")):
            raise KeyError(f"no csv entry with input_file for {folder_name}")
        _p = Path(_e["input_file"])
        return nib.load(str(_p)).get_fdata(dtype=np.float32)

    for index, batch in enumerate(dataloader):
        if index >= requested:
            break

        _, out_dict, _, _, folder_names = batch
        folder_name = str(folder_names[0])

        if bool(args.resume) and _already_complete(output_dir, folder_name, n_gen):
            skipped += 1
            logger.log(f"[{index + 1}/{requested}] SKIP complete: {folder_name}")
            continue

        pair_start = time.perf_counter()
        logger.log(f"[{index + 1}/{requested}] Sampling: {folder_name}")

        try:
            baseline = out_dict["cond_image"].to(device, non_blocking=True)
            cond_vec = out_dict["cond_vec"].to(device, non_blocking=True)

            if tuple(baseline.shape) != (1, 1, *shape):
                raise ValueError(f"Unexpected baseline shape {tuple(baseline.shape)}")
            if tuple(cond_vec.shape) != (1, 6):
                raise ValueError(f"Unexpected cond_vec shape {tuple(cond_vec.shape)}")
            if not th.isfinite(baseline).all():
                raise ValueError("Baseline contains NaN or Inf.")
            if not th.isfinite(cond_vec).all():
                raise ValueError("Condition vector contains NaN or Inf.")

            conditioning = {
                "y": th.ones(1, dtype=th.long, device=device),
                "cond_image": baseline,
                "cond_vec": cond_vec,
            }

            baseline_for_recon = (_load_baseline_native(folder_name)
                                  if getattr(args, "predict_target", False) else None)
            generated = []
            for generation in range(n_gen):

                seed = int(args.random_seed) + index * 100_003 + generation
                _seed_everything(seed)

                def _sample_with(cond):
                    with th.inference_mode():
                        return diffusion.ddim_sample_loop(
                            model_fn,
                            (1, 1, *shape),
                            clip_denoised=bool(args.clip_denoised),
                            model_kwargs=cond,
                            device=device,
                            progress=bool(args.progress),
                            eta=float(args.eta),
                        )

                if th.cuda.is_available():
                    th.cuda.synchronize()
                _t_gpu0 = time.perf_counter()
                sample = _sample_with(conditioning)
                if th.cuda.is_available():
                    th.cuda.synchronize()
                _gpu_s = time.perf_counter() - _t_gpu0

                if bool(getattr(args, "tta_flip", False)):
                    base_np = baseline[0, 0].float().cpu().numpy()
                    base_flip = pad_to(crop_from_pad(base_np)[::-1].copy())
                    cond_flip = dict(conditioning)
                    cond_flip["cond_image"] = th.from_numpy(base_flip).to(
                        device, dtype=baseline.dtype)[None, None]
                    s_flip = _sample_with(cond_flip)
                    s_flip_np = s_flip[0, 0].float().cpu().numpy()
                    s_unflip = pad_to(crop_from_pad(s_flip_np)[::-1].copy())
                    sample = (sample
                              + th.from_numpy(s_unflip).to(device, dtype=sample.dtype)
                                  [None, None]) / 2.0

                if tuple(sample.shape) != (1, 1, *shape):
                    raise ValueError(f"Unexpected sample shape {tuple(sample.shape)}")
                if not th.isfinite(sample).all():
                    raise FloatingPointError("Sample contains NaN or Inf.")

                change_pad = sample[0, 0].float().cpu().numpy().astype(np.float32)

                if getattr(args, "predict_target", False):
                    target_native = crop_from_pad((change_pad + 1.0) / 2.0)
                    change_native = target_native - baseline_for_recon
                else:
                    change_native = crop_from_pad(change_pad * float(CHANGE_SCALE))
                if tuple(change_native.shape) != tuple(NATIVE_SHAPE):
                    raise ValueError(f"Unexpected cropped shape {change_native.shape}")
                generated.append(change_native)

            generations = np.stack(generated, axis=0)
            change_mean = generations.mean(axis=0, dtype=np.float32)
            change_std = generations.std(axis=0, dtype=np.float32) if n_gen > 1 else np.zeros_like(change_mean)

            _entry = _entry_by_folder.get(folder_name)
            if not (_entry and _entry.get("input_file")):
                raise KeyError(f"no csv entry with input_file for {folder_name}")
            input_path = Path(_entry["input_file"])
            input_image = nib.load(str(input_path))
            baseline_native = input_image.get_fdata(dtype=np.float32)
            if tuple(baseline_native.shape) != tuple(NATIVE_SHAPE):
                raise ValueError(
                    f"Native baseline shape is {baseline_native.shape}; expected {NATIVE_SHAPE}."
                )
            if not np.isfinite(baseline_native).all():
                raise ValueError("Native baseline contains NaN or Inf.")

            if bool(args.zero_outside_gm):
                gm_native = baseline_native > float(MASK_THRESHOLD)
                change_mean = np.where(gm_native, change_mean, 0.0).astype(np.float32)
                change_std = np.where(gm_native, change_std, 0.0).astype(np.float32)

            target_pred = (baseline_native + change_mean).astype(np.float32)
            if bool(args.clip_target):
                target_pred = np.clip(target_pred, 0.0, 1.0).astype(np.float32)

            if not np.isfinite(change_mean).all() or not np.isfinite(target_pred).all():
                raise FloatingPointError("Final reconstructed arrays contain NaN or Inf.")

            pair_dir = output_dir / folder_name
            pair_dir.mkdir(parents=True, exist_ok=True)
            change_path = pair_dir / "predicted_change_mwp1sub.nii.gz"
            target_path = pair_dir / "predicted_target_mwp1sub.nii.gz"
            uncertainty_path = pair_dir / "uncertainty_mwp1sub.nii.gz"

            _atomic_save_nifti(change_mean, input_image, change_path)
            _atomic_save_nifti(target_pred, input_image, target_path)
            if n_gen > 1:
                _atomic_save_nifti(change_std, input_image, uncertainty_path)

            elapsed = time.perf_counter() - pair_start
            measured_times.append(elapsed)
            completed_this_run += 1
            remaining_unseen = requested - (index + 1)
            eta_seconds = float(np.mean(measured_times[-10:])) * remaining_unseen if measured_times else 0.0

            stats = {
                "folder_name": folder_name,
                "status": "complete",
                "index": index + 1,
                "total_selected": requested,
                "seconds": round(elapsed, 3),
                "eta_hours": round(eta_seconds / 3600.0, 3),
                "change_min": float(change_mean.min()),
                "change_max": float(change_mean.max()),
                "change_mean": float(change_mean.mean()),
                "change_std": float(change_mean.std()),
                "target_min": float(target_pred.min()),
                "target_max": float(target_pred.max()),
                "condition": cond_vec[0].detach().float().cpu().tolist(),
                "files": {
                    "predchange": str(change_path),
                    "predtarget": str(target_path),
                    "uncertainty": str(uncertainty_path) if n_gen > 1 else None,
                },
            }
            _append_jsonl(progress_path, stats)
            compact_index[folder_name] = stats
            logger.log(
                f"[{index + 1}/{requested}] SAVED {folder_name} in {elapsed / 60:.2f} min "
                f"(sampling {_gpu_s:.3f} s); "
                f"change=[{stats['change_min']:.5f}, {stats['change_max']:.5f}]; "
                f"ETA≈{eta_seconds / 3600:.2f} h"
            )

            del sample, generated, generations, baseline, cond_vec
            if device.type == "cuda":
                th.cuda.empty_cache()

        except Exception as exc:
            failed += 1
            failure = {
                "folder_name": folder_name,
                "status": "failed",
                "index": index + 1,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            _append_jsonl(failure_path, failure)
            logger.log(f"FAILED {folder_name}: {type(exc).__name__}: {exc}")
            if not bool(args.continue_on_error):
                raise

    if compact_index:
        _atomic_pickle(compact_index, output_dir / "cond_change_predictions.pkl")

    logger.log(
        f"Inference finished: completed_now={completed_this_run}, "
        f"skipped={skipped}, failed={failed}, selected={requested}."
    )
    logger.log(f"Outputs: {output_dir}")


def create_argparser() -> argparse.ArgumentParser:
    config_path = "./src/scripts/config/config_inference_cond_gmd_3d.yaml"
    for i, token in enumerate(sys.argv):
        if token == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
        elif token.startswith("--config="):
            config_path = token.split("=", 1)[1]

    defaults: Dict[str, Any] = {
        "data_dir": "",
        "results_dir": "",
        "run_id": "cond_change",
        "model_path": "",
        "tier1_csv": "",
        "clip_denoised": False,
        "use_fp16": False,
        "eta": 0.0,
        "image_size_d": 120,
        "image_size_h": 144,
        "image_size_w": 120,
        "generation_number": 1,
        "probability_map": False,
        "max_samples": 0,
        "device": "cluster",
        "resume": True,
        "progress": False,
        "continue_on_error": False,
        "zero_outside_gm": True,
        "clip_target": False,
        "num_workers": 0,
    }
    defaults.update(model_and_diffusion_defaults())

    defaults.update(dict(
        pairs_csv="",
        scans_dir="",
        eval_split="test",
        eval_intervals="",

        exclude_subjects_csv="",
        exclude_split="train",
        zero_cond_slots="",

        predict_target=False,

        tta_flip=False,
    ))

    config = Path(config_path)
    if not config.exists():
        raise FileNotFoundError(f"Inference config not found: {config}")
    with config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    unknown = sorted(set(cfg) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    defaults.update(cfg)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(config))
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
