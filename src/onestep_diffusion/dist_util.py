import io
import os
import socket

import blobfile as bf

import torch as th
import torch.distributed as dist

GPUS_PER_NODE = 2

SETUP_RETRY_COUNT = 3


def setup_dist():

    if dist.is_initialized():
        return
    backend = "nccl" if th.cuda.is_available() else "gloo"
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if th.cuda.is_available():
            th.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        if th.cuda.is_available():
            th.cuda.set_device(0)
        dist.init_process_group(backend=backend, init_method="env://")


def dev():

    if th.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return th.device(f"cuda:{local_rank}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):

    mpigetrank=0

    if mpigetrank==0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):

    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
