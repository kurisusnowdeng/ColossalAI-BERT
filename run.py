import os

from colossalai.core import global_context as gpc

from helper import init_w_col
from train import train
from utils import get_args, get_config


def setup_mpi():
    from titan_plugins import mpi_discovery_plugin as mdp
    mdp.mpi_discovery_for_pytorch_ddp()

    MASTER_PORT = os.environ.get('MASTER_PORT')
    MASTER_ADDR = os.environ.get('MASTER_ADDR')
    WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
    RANK = os.environ.get('RANK')
    LOCAL_RANK = os.environ.get('LOCAL_RANK')
    print("MASTER_ADDR:{}, MASTER_PORT:{}, WORLD_SIZE:{}, RANK:{}, LOCAL_RANK:{}".format(
        MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, LOCAL_RANK))


def run_bert():
    train(*init_w_col())
    gpc.destroy()


if __name__ == '__main__':
    args = get_args()
    config = get_config()

    if args.use_mpi:
        setup_mpi()

    run_bert()
