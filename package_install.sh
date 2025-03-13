#!/bin/bash

# CUDA 12.4
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install numpy
pip install jax
pip install -U "jax[cuda12]"
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html