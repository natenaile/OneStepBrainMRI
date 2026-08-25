import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from onestep_diffusion import dist_util, logger
from onestep_diffusion.fp16_util import MixedPrecisionTrainer
from onestep_diffusion.nn import update_ema
from onestep_diffusion.resample import UniformSampler

INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        dataset='gmd',
        val_data=None,
        val_interval=0,
        max_train_hours=0.0,
        grad_clip=1.0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.datal = data
        self.dataset = dataset
        self.iterdatal = iter(data)

        self.val_data = val_data
        self.val_interval = val_interval

        self.max_train_hours = max_train_hours
        self._train_start_time = None
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.grad_clip = grad_clip
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0

        self.best_dpearson = -float("inf")
        self.best_step = -1
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            grad_clip=self.grad_clip,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            _dev = dist_util.dev()
            self.ddp_model = DDP(
                self.model,
                device_ids=[_dev.index],
                output_device=_dev.index,
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)

            if self.resume_step == 0:
                try:
                    import json
                    _d = bf.dirname(resume_checkpoint)
                    step_file = (bf.join(_d, "step_best.json")
                                 if "_best" in bf.basename(resume_checkpoint)
                                 and bf.exists(bf.join(_d, "step_best.json"))
                                 else bf.join(_d, "step_last.json"))
                    if bf.exists(step_file):
                        with bf.BlobFile(step_file, "r") as f:
                            self.resume_step = int(json.load(f)["step"])
                        logger.log(f"[resume] continuing from step {self.resume_step} "
                                   f"(read from step_last.json)")
                    else:
                        logger.log("[resume] no step_last.json; step counter and LR "
                                   "schedule restart from 0")
                except Exception as e:
                    logger.log(f"[resume] could not read step_last.json: {e}")

            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                try:
                    self.model.load_state_dict(
                        dist_util.load_state_dict(
                            resume_checkpoint, map_location=dist_util.dev()
                        )
                    )
                except:
                    state_dict = th.load(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        new_key = key.replace('module.', '')
                        new_state_dict[new_key] = value
                    self.model.load_state_dict(new_state_dict)
                self.model.to(dist_util.dev())

            if dist.is_initialized() and dist.get_world_size() > 1:
                # All ranks, outside the rank-0 check: DDP all-reduces gradients, not
                # parameters, so without this every rank but 0 trains from its random
                # initialisation for the entire run.
                dist_util.sync_params(self.model.parameters())
                dist_util.sync_params(self.model.buffers())
                dist.barrier()
                _pn = sum(float((p.detach() ** 2).sum()) for p in
                          self.model.parameters()) ** 0.5
                logger.log(f"[sync] rank {dist.get_rank()} param_norm "
                           f"after broadcast = {_pn:.2f} "
                           f"(all ranks must print the SAME value)")

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                try:
                    ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
                except:
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        new_key = key.replace('module.', '')
                        new_state_dict[new_key] = value
                    ema_params = self.mp_trainer.state_dict_to_master_params(
                        new_state_dict
                    )
        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ckpt_dir = bf.dirname(main_checkpoint)

        resuming_from_best = "_best" in bf.basename(main_checkpoint)
        candidates = ([bf.join(ckpt_dir, "opt_best.pt")] if resuming_from_best else []) + [
            bf.join(ckpt_dir, "opt_last.pt"),
            bf.join(ckpt_dir, f"opt{self.resume_step:06}.pt"),
        ]
        opt_checkpoint = next((p for p in candidates if bf.exists(p)), None)
        if opt_checkpoint is None:
            logger.log(f"[resume] no optimizer state found in {ckpt_dir} "
                       f"(looked for opt_last.pt); Adam will restart cold")
            return
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        state_dict = dist_util.load_state_dict(
            opt_checkpoint, map_location=dist_util.dev()
        )
        self.opt.load_state_dict(state_dict)
        logger.log("[resume] optimizer state restored")

    def run_loop(self):

        NAN_ABORT_SCALE = -20.0
        import os as _os
        NAN_ABORT_SCALE = float(_os.environ.get("NAN_ABORT_SCALE", NAN_ABORT_SCALE))
        import time
        th.cuda.empty_cache()
        print("Memory allocated ", th.cuda.memory_allocated())
        print("Memory cached ", th.cuda.memory_cached())

        self._train_start_time = time.time()
        if self.max_train_hours and self.max_train_hours > 0:
            logger.log(f"Wall-clock training budget: {self.max_train_hours:.1f} h "
                       f"(graceful save + exit when reached)")

        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):

            if self.dataset == 'brats' or self.dataset == 'gmd':
                try:
                    batch, cond, label, _, _ = next(self.iterdatal)
                except StopIteration:
                    self.iterdatal = iter(self.datal)
                    batch, cond, label, _, _ = next(self.iterdatal)
            elif self.dataset == 'chexpert':
                batch, cond = next(self.datal)
                cond.pop("path", None)

            self.run_step(batch, cond)

            lg = getattr(self.mp_trainer, "lg_loss_scale", None)
            if lg is not None and lg < NAN_ABORT_SCALE:
                if dist.get_rank() == 0:
                    logger.log(
                        f"[abort] lg_loss_scale = {lg:.1f} < {NAN_ABORT_SCALE} "
                        f"at step {self.step + self.resume_step}: gradients have "
                        f"underflowed and the weights are NaN. Training cannot "
                        f"recover - stopping so the job releases the GPU.")
                    if self.best_step >= 0:
                        logger.log(
                            f"[abort] best checkpoint is intact: "
                            f"val delta-Pearson {self.best_dpearson:.4f} at step "
                            f"{self.best_step} -> ema_{self.ema_rate[0]}_best.pt")
                    logger.log(
                        "[abort] This threshold needs ~40 NET NaN events "
                        "(-1 each, +fp16_scale_growth per clean step), so it "
                        "cannot fire on isolated NaN - the scaler genuinely "
                        "collapsed.")
                    logger.log(
                        "[abort] On this project every collapse so far had "
                        "lr=1.4e-4; the one run that trained 41106 steps clean "
                        "had lr=1e-4. corr_weight is NOT the variable (failures "
                        "span 0.1/0.3/1.0). Try lr=1e-4 first, then "
                        "use_fp16: false. Resume from *_best.pt.")
                break

            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            if (self.val_data is not None and self.val_interval > 0
                    and self.step % self.val_interval == 0 and self.step > 0):
                val_loss, val_dp = self.run_validation()
                if val_loss is not None and dist.get_rank() == 0:
                    gstep = self.step + self.resume_step
                    dp_txt = "n/a" if val_dp is None else f"{float(val_dp):.4f}"
                    logger.log(f"[val] step {gstep}  val loss = "
                               f"{float(val_loss):.6f}  delta-Pearson = {dp_txt}")
                    logger.logkv("val_loss", float(val_loss))
                    if val_dp is not None:
                        logger.logkv("val_dpearson", float(val_dp))
                    self._save_val_csv(
                        gstep, float(val_loss),
                        None if val_dp is None else float(val_dp),
                    )

                if val_dp is not None:
                    flag = th.zeros(1, device=dist_util.dev())
                    if dist.get_rank() == 0 and float(val_dp) > self.best_dpearson:
                        self.best_dpearson = float(val_dp)
                        self.best_step = self.step + self.resume_step
                        flag[0] = 1.0
                    dist.broadcast(flag, src=0)
                    if flag.item() > 0:
                        self.save_best()
            if self.step % self.save_interval == 0:
                self.save()
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            if self.max_train_hours and self.max_train_hours > 0:
                elapsed_h = (time.time() - self._train_start_time) / 3600.0
                if elapsed_h >= self.max_train_hours:
                    logger.log(f"Reached wall-clock budget "
                               f"({elapsed_h:.2f} h >= {self.max_train_hours:.1f} h) "
                               f"at step {self.step + self.resume_step}. "
                               f"Saving final checkpoint and exiting.")
                    self.save()

                    if self.val_data is not None and self.val_interval > 0:
                        vl, vdp = self.run_validation()
                        if vl is not None and dist.get_rank() == 0:
                            self._save_val_csv(
                                self.step + self.resume_step, float(vl),
                                None if vdp is None else float(vdp),
                            )
                    return
            self.step += 1

        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        lossmse, sample = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return lossmse, sample

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())

            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
                if k not in ("target_slice", "input_slice")
            }

            if "target_slice" in cond:
                micro_cond["target_slice"] = cond["target_slice"][
                    i: i + self.microbatch
                ].to(dist_util.dev())
            if "input_slice" in cond:
                micro_cond["input_slice"] = cond["input_slice"][
                    i: i + self.microbatch
                ].to(dist_util.dev())

            if "cond_image" in cond:
                micro_cond["cond_image"] = cond["cond_image"][
                    i: i + self.microbatch
                ].to(dist_util.dev())

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(
                micro.shape[0], dist_util.dev()
            )

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses1 = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses1 = compute_losses()

            losses = losses1[0]
            sample = losses1[1]

            loss = (losses["loss"] * weights).mean()
            lossmse = (losses["mse"] * weights).mean().detach()

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

            return lossmse.detach(), sample

    def run_validation(self):

        if self.val_data is None:
            return None, None
        self.ddp_model.eval()
        losses, corrs = [], []

        gen = th.Generator(device="cpu").manual_seed(1234)
        with th.no_grad():
            for batch, cond, label, _, _ in self.val_data:
                losses.append(self._validate_batch(batch, cond))
                c = self._validate_dpearson(batch, cond, gen)
                if c is not None:
                    corrs.append(c)
        self.ddp_model.train()
        if not losses:
            return None, None
        mean_loss = th.mean(th.stack(losses))
        mean_corr = th.mean(th.stack(corrs)) if corrs else None
        return mean_loss, mean_corr

    def _validate_dpearson(self, batch, cond, gen):

        from onestep_diffusion.gaussian_diffusion import ModelMeanType

        dev = dist_util.dev()
        x_start = batch.to(dev)
        T = self.diffusion.num_timesteps

        micro_cond = {
            k: v.to(dev) for k, v in cond.items()
            if k not in ("target_slice", "input_slice")
        }
        if "cond_image" in cond:
            micro_cond["cond_image"] = cond["cond_image"].to(dev)
        gm = micro_cond.get("gm_mask", None)

        model_kwargs = {k: v for k, v in micro_cond.items() if k != "gm_mask"}

        x_t = th.randn(x_start.shape, generator=gen).to(dev)
        t = th.full((x_start.shape[0],), T - 1, device=dev, dtype=th.long)

        out = self.ddp_model(x_t, self.diffusion._scale_timesteps(t), **model_kwargs)
        if self.diffusion.model_mean_type != ModelMeanType.START_X:
            out = self.diffusion._predict_xstart_from_eps(x_t=x_t, t=t, eps=out)

        p = out.float()
        q = x_start.float()
        m = gm.float() if gm is not None else th.ones_like(p)
        dims = list(range(1, len(p.shape)))
        n = m.sum(dim=dims).clamp(min=1.0)
        shape = [-1] + [1] * (len(p.shape) - 1)
        pc = (p - ((p * m).sum(dim=dims) / n).view(shape)) * m
        qc = (q - ((q * m).sum(dim=dims) / n).view(shape)) * m
        corr = (pc * qc).sum(dim=dims) / (
            (pc * pc).sum(dim=dims).sqrt() * (qc * qc).sum(dim=dims).sqrt() + 1e-8
        )
        return corr.clamp(-1.0, 1.0).mean().detach()

    def _validate_batch(self, batch, cond):

        validation_losses = []
        with th.no_grad():
            for i in range(0, batch.shape[0], self.microbatch):
                micro = batch[i: i + self.microbatch].to(dist_util.dev())

                micro_cond = {
                    k: v[i: i + self.microbatch].to(dist_util.dev())
                    for k, v in cond.items()
                    if k not in ("target_slice", "input_slice")
                }
                if "target_slice" in cond:
                    micro_cond["target_slice"] = cond["target_slice"][
                        i: i + self.microbatch].to(dist_util.dev())
                if "input_slice" in cond:
                    micro_cond["input_slice"] = cond["input_slice"][
                        i: i + self.microbatch].to(dist_util.dev())
                if "cond_image" in cond:
                    micro_cond["cond_image"] = cond["cond_image"][
                        i: i + self.microbatch].to(dist_util.dev())

                t, weights = self.schedule_sampler.sample(
                    micro.shape[0], dist_util.dev()
                )
                losses1 = self.diffusion.training_losses(
                    self.ddp_model, micro, t, model_kwargs=micro_cond
                )
                losses = losses1[0]

                validation_losses.append((losses["loss"] * weights).mean().detach())
        return th.mean(th.stack(validation_losses))

    def _save_val_csv(self, step, val_loss, val_dpearson=None):

        try:
            out_dir = logger.get_dir() or "."
            path = os.path.join(out_dir, "validation_ddpm.csv")
            new = not os.path.exists(path)
            dp = "" if val_dpearson is None else f"{val_dpearson:.6f}"
            with open(path, "a") as f:
                if new:
                    f.write("step,val_loss,val_dpearson\n")
                f.write(f"{step},{val_loss:.6f},{dp}\n")
        except Exception as e:
            logger.log(f"[val] could not write validation_ddpm.csv: {e}")

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv(
            "samples",
            (self.step + self.resume_step + 1) * self.global_batch
        )

    def save_best(self):

        def _save(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                filename = "model_best.pt" if not rate else f"ema_{rate}_best.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        _save(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            _save(rate, params)

        if dist.get_rank() == 0:
            try:
                with bf.BlobFile(bf.join(get_blob_logdir(), "opt_best.pt"), "wb") as f:
                    th.save(self.opt.state_dict(), f)
                import json
                with open(bf.join(get_blob_logdir(), "step_best.json"), "w") as f:
                    json.dump({"step": self.best_step,
                               "val_dpearson": self.best_dpearson}, f)
            except Exception as e:
                logger.log(f"[best] could not write opt_best.pt/step_best.json: {e}")

            logger.log(f"[best] new best delta-Pearson = {self.best_dpearson:.4f} "
                       f"at step {self.best_step} -> saved *_best.pt")
            try:
                import json
                meta = {"best_step": self.best_step,
                        "best_val_dpearson": self.best_dpearson,
                        "ema_rate": list(self.ema_rate)}
                with open(os.path.join(logger.get_dir() or ".",
                                       "best_checkpoint.json"), "w") as f:
                    json.dump(meta, f, indent=2)
            except Exception as e:
                logger.log(f"[best] could not write best_checkpoint.json: {e}")

        dist.barrier()

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")

                if not rate:
                    filename = "model_last.pt"
                else:
                    filename = f"ema_{rate}_last.pt"
                print('filename', filename)
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(
                    get_blob_logdir(),
                    "opt_last.pt"
                ),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

            try:
                import json
                with open(bf.join(get_blob_logdir(), "step_last.json"), "w") as f:
                    json.dump({"step": self.step + self.resume_step}, f)
            except Exception as e:
                logger.log(f"could not write step_last.json: {e}")

        dist.barrier()


def parse_resume_step_from_filename(filename):
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return logger.get_dir()


def find_resume_checkpoint():
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(
            ts.cpu().numpy(), values.detach().cpu().numpy()
        ):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
