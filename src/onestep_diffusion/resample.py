from abc import ABC, abstractmethod

import numpy as np
import torch as th


def create_named_schedule_sampler(name, diffusion, maxt):

    if name == "uniform":
        return UniformSampler(diffusion, maxt)
    raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):

    @abstractmethod
    def weights(self):
        pass

    def sample(self, batch_size, device):

        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion, maxt):
        self.diffusion = diffusion
        self._weights = np.ones([maxt])

    def weights(self):
        return self._weights
