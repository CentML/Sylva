import os
import torch


def init_process_group(args):
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group()
    args.global_rank = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size()
    print(
        f"local_rank={args.local_rank}, rank={args.global_rank}, world size={args.world_size}"
    )
    return torch.device("cuda", args.local_rank)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor
