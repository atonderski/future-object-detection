import os
import signal
import threading

import torch
import torch.distributed as distrib


# Parts of code from
# https://github.com/erikwijmans/skynet-ddp-slurm-example/blob/master/ddp_example/train_cifar10.py .
# Namely, EXIT and exit_handler, get_ifname, and distributed part of init_distributed_and_device_
EXIT = threading.Event()
EXIT.clear()


def _clean_exit_handler(signum, frame):
    EXIT.set()
    print("Exiting cleanly", flush=True)


signal.signal(signal.SIGINT, _clean_exit_handler)
signal.signal(signal.SIGTERM, _clean_exit_handler)
signal.signal(signal.SIGUSR2, _clean_exit_handler)


def disable_prints_unless_master(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_and_device_(args):
    if args.distributed:
        args.world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
        args.world_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
        # distrib.init_process_group(
        #    backend='nccl', rank=args.world_rank, world_size=args.world_size)
        distrib.init_process_group(backend="nccl", init_method="env://")
        args.device = torch.device("cuda", args.local_rank)
        torch.cuda.set_device(args.device)
        disable_prints_unless_master(args.world_rank == 0)
    else:
        # Set some dummy values
        args.local_rank = 0
        args.world_rank = 0
        args.world_size = 1
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    return


def reduce_distrib_loss(input_dict, average=True):
    world_size = distrib.get_world_size()
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        distrib.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def gather_distrib_od_map_stuffs(inputs):
    world_size = distrib.get_world_size()
    with torch.no_grad():
        gathered_inputs = [
            [torch.zeros_like(tensor) for _ in range(world_size)] for tensor in inputs
        ]
        for idx, tensor in enumerate(inputs):
            distrib.all_gather(gathered_inputs[idx], tensor)

    return gathered_inputs
