#!/bin/bash

# run.sh

# --
# Install

conda create -y -n ppnp_env python=3.7
conda activate ppnp_env

conda install -y -c pytorch pytorch=1.2.0
conda install -y scipy=1.3.1
conda install -y pandas

# --
# Run
# !! Mean scores below match paper.  std's may be slightly different.

python main.py --inpath ppnp/data/cora_ml.npz --n-runs 32
# valid_acc = 0.856 (std=0.009)

python main.py --inpath ppnp/data/citeseer.npz --n-runs 32
# valid_acc = 0.760 (std=0.012)

python main.py --inpath ppnp/data/pubmed.npz --n-runs 10
# valid_acc = 0.812 (std=0.007)

python main.py --inpath ppnp/data/ms_academic.npz --n-runs 10 --verbose | tee res.jl
# valid_acc = 0.932221 (std=0.003)

# --
# Test batch-main.py

python batch-main.py --inpath ppnp/data/cora_ml.npz --n-runs 1 --verbose --batch-size 32 --ppr-topk 128
python batch-main.py --inpath ppnp/data/ms_academic.npz --n-runs 1 --verbose --batch-size 1024 --ppr-topk 256


# --
# Node embedding experiments

python unsupervised-main.py

CUDA_VISIBLE_DEVICES=6 python batch-unsupervised-main.py --n-runs 32
CUDA_VISIBLE_DEVICES=6 python batch-unsupervised-main.py --n-runs 32 --sparse

# !! Truncating PPR helps (on cora_ml)