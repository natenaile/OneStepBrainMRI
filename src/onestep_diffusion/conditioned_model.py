import torch as th
import torch.nn as nn


class ConditionedModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, timesteps, **kwargs):
        cond = kwargs.pop("cond_image", None)

        kwargs.pop("input_slice", None)
        kwargs.pop("target_slice", None)
        kwargs.pop("gm_mask", None)

        if cond is not None:

            x = th.cat([x, cond.to(x.dtype)], dim=1)
        return self.model(x, timesteps, **kwargs)

    def convert_to_fp16(self):
        if hasattr(self.model, "convert_to_fp16"):
            self.model.convert_to_fp16()

    def convert_to_fp32(self):
        if hasattr(self.model, "convert_to_fp32"):
            self.model.convert_to_fp32()

    @property
    def dtype(self):
        return getattr(self.model, "dtype", th.float32)
