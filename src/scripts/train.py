import sys
import os
import json
import argparse

import torch as th
import torch.nn as nn

sys.path.append("..")
sys.path.append(".")

from onestep_diffusion.gmdloader import GMDDataset
from onestep_diffusion import dist_util, logger
from onestep_diffusion.resample import create_named_schedule_sampler
from onestep_diffusion.train_util import TrainLoop

from utils.script_util_gmd import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import yaml


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(args.results_dir)

    save_model_config(args)

    logger.log("Creating BASELINE-CONDITIONED model and diffusion...")

    md_kwargs = args_to_dict(args, model_and_diffusion_defaults().keys())
    md_kwargs["cond_channels"] = 1
    base_model, diffusion = create_model_and_diffusion(**md_kwargs)

    from onestep_diffusion.conditioned_model import ConditionedModel
    model = ConditionedModel(base_model)
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion, maxt=1000
    )

    logger.log("Creating data loader...")

    _common = dict(
        test_flag=False,
        regression=False,
        single_contrast=True,
        device=args.device,
        conditioned=True,
        tier1_csv=getattr(args, "tier1_csv", None) or None,
    )
    ds = GMDDataset(
        pairs_csv=args.pairs_csv,
        split="train",
        scans_dir=getattr(args, "scans_dir", "") or None,
        intervals=getattr(args, "train_intervals", "") or None,
        augment_flip=bool(getattr(args, "augment_flip", False)),
        zero_cond_slots=getattr(args, "zero_cond_slots", "") or None,
        predict_target=bool(getattr(args, "predict_target", False)),
        **_common,
    )
    import torch.distributed as _dist
    _world = _dist.get_world_size() if _dist.is_initialized() else 1
    train_sampler = (th.utils.data.distributed.DistributedSampler(ds, shuffle=True)
                     if _world > 1 else None)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_datal = None
    if getattr(args, "data_dir_val", "") and args.val_interval > 0:

        ds_val = GMDDataset(
            pairs_csv=args.pairs_csv,
            split="val",
            scans_dir=getattr(args, "scans_dir", "") or None,
            intervals=getattr(args, "val_intervals", "") or None,
            zero_cond_slots=getattr(args, "zero_cond_slots", "") or None,
            predict_target=bool(getattr(args, "predict_target", False)),
            **_common,
        )
        val_datal = th.utils.data.DataLoader(
            ds_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        logger.log(f"Validation set: {len(ds_val)} volumes "
                   f"(val every {args.val_interval} steps)")

    logger.log(f"Dataset size: {len(ds)} volumes")
    logger.log("Starting training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        val_data=val_datal,
        val_interval=args.val_interval,
        max_train_hours=args.max_train_hours,
        grad_clip=getattr(args, "grad_clip", 1.0),
    ).run_loop()


def save_model_config(args):

    config = {
        "image_size":             args.image_size,
        "num_channels":           args.num_channels,
        "num_res_blocks":         args.num_res_blocks,
        "attention_resolutions":  args.attention_resolutions,
        "learn_sigma":            args.learn_sigma,
        "class_cond":             args.class_cond,
        "diffusion_steps":        args.diffusion_steps,
        "noise_schedule":         args.noise_schedule,
        "dataset":                args.dataset,
        "in_channels":            1,
        "out_channels":           2,
        "batch_size":             args.batch_size,
        "lr":                     args.lr,
    }
    os.makedirs(args.results_dir, exist_ok=True)
    config_path = os.path.join(args.results_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.log(f"Model config saved to {config_path}")

CSV_MODE_DEFAULTS = dict(
    pairs_csv="",
    scans_dir="",
    train_intervals="",
    val_intervals="",

    augment_flip=False,
)


def create_argparser():
    defaults = dict(
        data_dir="",
        data_dir_val="",
        val_interval=0,
        max_train_hours=0.0,
        grad_clip=1.0,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=10,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset="gmd",
        results_dir="./results/ddpm",
        device="cluster",
        single_contrast=True,
        zero_cond_slots="",
        predict_target=False,
        config="./src/scripts/config/train_main.yaml",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(CSV_MODE_DEFAULTS)

    import sys
    config_path = defaults["config"]
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
        elif arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
    defaults["config"] = config_path

    with open(defaults["config"], "r") as f:
        config = yaml.safe_load(f)
        defaults.update(config)

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
