FROM nvcr.io/nvidia/pytorch:22.04-py3

SHELL ["/bin/bash", "-c"]
WORKDIR /workspace/apps

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install --upgrade --no-cache-dir pip

# install pytorch
# RUN pip install --upgrade --no-cache-dir torch torchvision torchaudio torchtext \
#     --extra-index-url https://download.pytorch.org/whl/cu116

# install huggingface
RUN pip install --upgrade --no-cache-dir transformers datasets==1.18.0

# install apex
RUN git clone https://github.com/NVIDIA/apex && cd apex && \
    pip install --disable-pip-version-check --upgrade --no-cache-dir --verbose \
    --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" .

# install flash attention
RUN git clone https://github.com/HazyResearch/flash-attention.git && \
    cd flash-attention && pip install --upgrade --no-cache-dir --verbose -e .

# install colossalai
RUN git clone https://github.com/hpcaitech/ColossalAI.git && cd ColossalAI && pip install --upgrade --no-cache-dir --verbose -e .

# clean up
WORKDIR /workspace
RUN echo 'alias gpu="nvidia-smi"' >> ~/.bashrc && \
    echo 'alias gpu-t="nvidia-smi topo -m"' >> ~/.bashrc && \
    echo 'alias gpu-m="watch -n 0.5 nvidia-smi"' >> ~/.bashrc
