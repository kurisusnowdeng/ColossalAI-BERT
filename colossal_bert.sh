#!/bin/bash
# NCCL_DEBUG=INFO
OMPI_MCA_btl_tcp_if_include=eth0
NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG
export OMPI_MCA_btl_tcp_if_include
export NCCL_SOCKET_IFNAME

export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:/usr/local/cuda/targets/x86_64-linux/lib/:/usr/lib/x86_64-linux-gnu/
export PATH=/usr/local/nvm/versions/node/v15.2.1/bin:/usr/oppo/bin/:/usr/local/hadoop/bin:/usr/local/hive/bin:/usr/local/spark/bin:/opt/conda/bin:/opt/conda/condabin:/opt/conda/bin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

CONFIG="configs/bert_config_tp1d.json"
DATA=/home/notebook/data/personal/V48952789/bert-test
VOCAB=$DATA/bert-large-uncased-vocab.txt

python run.py --config=$CONFIG --data-path=$DATA --vocab-file=$VOCAB --use-mpi
