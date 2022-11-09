# Bert Benchmark
Bert Benchmark with Colossal-AI.

## Setup
1. Install dependencies if you do not have them.
```
pip install -r requirement.txt
```

2. Install Colossal-AI
```
git clone https://github.com/hpcaitech/ColossalAI.git

cd ColossalAI

pip install --no-cache-dir .

```

2. Prepare dataset from JSON file.
```
# install nltk
pip install --upgrade --no-cache-dir nltk

python -c "import nltk; nltk.download('punkt')"

python process_data.py \
    --data-path /PATH/TO/JSON/FILE \
    --vocab-file /PATH/TO/VOCAB/FILE \
    --output-path /PATH/TO/OUTPUT/DATASET/ \
    --seq-len 512
```

## Colossal-AI Usage

1. Run benchmark
- Prepare your configuration file such as [configs/bert_config_tp1d.json](./configs/bert_config_tp1d.json)
- Locally:
```
torchrun --nproc_per_node=NUM_GPUS \
    run.py \
    --config configs/bert_config_tp1d.json \
    --data-path /PATH/TO/DATASET/ \
    --vocab-file /PATH/TO/VOCAB/FILE
```
- On platform: submit [colossal_bert.sh](./colossal_bert.sh).

## Baseline Usage

```
git clone https://github.com/NVIDIA/Megatron-LM.git

cd Megatron-LM

torchrun --nproc_per_node=NUM_GPUS \
    pretrain_bert.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --num-layers 36 \
    --hidden-size 2048 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 32 \
    --micro-batch-size 8 \
    --global-batch-size 5120 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --train-iters 20 \
    --iter-per-epoch 10 \
    --data-path /PATH/TO/DATASET/my-bert_text_sentence \
    --vocab-file /PATH/TO/VOCAB/FILE \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
    --lr 0.0001 \
    --lr-decay-style linear \
    --min-lr 1.0e-5 \
    --lr-decay-iters 16 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --lr-warmup-iters 4 \
    --log-interval 2 \
    --eval-iters 1 \
    --fp16
```