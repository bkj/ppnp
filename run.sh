#!/bin/bash

# run.sh

# --
# Install

conda create -y -n ppnp_env python=3.7
conda activate ppnp_env

conda install -y -c pytorch pytorch=1.3.1 cudatoolkit=10.0
conda install -y scipy=1.3.1
conda install -y pandas
conda install -y numba
conda install -y scikit-learn

pip install networkx


# >>

python main.py

# <<

# --
# Run
# !! Mean scores below match paper.  std's may be slightly different.

CUDA_VISIBLE_DEVICES=6 python main.py --inpath ppnp/data/cora_ml.npz --n-runs 32
# valid_acc = 0.856 (std=0.009)

CUDA_VISIBLE_DEVICES=6 python batch-main.py --inpath ppnp/data/cora_ml.npz --n-runs 32 --batch-size 512

python main.py --inpath ppnp/data/citeseer.npz --n-runs 32
# valid_acc = 0.760 (std=0.012)

python main.py --inpath ppnp/data/pubmed.npz --n-runs 10
# valid_acc = 0.812 (std=0.007)

python main.py --inpath ppnp/data/ms_academic.npz --n-runs 10 --verbose | tee res.jl
# valid_acc = 0.932221 (std=0.003)

# --
# Test batch-main.py

CUDA_VISIBLE_DEIVCES=6 python batch-main.py --inpath ppnp/data/cora_ml.npz --n-runs 32

python batch-main.py --inpath ppnp/data/ms_academic.npz --n-runs 1 --verbose --batch-size 32 --ppr-topk 128
python batch-main.py --inpath ppnp/data/ms_academic.npz --n-runs 1 --verbose --batch-size 1024 --ppr-topk 256

# --
# Test variants

python unsupervised.py --n-runs 5 --max-epochs 500
python unsupervised.py --n-runs 5 --ppr-topk 128 --max-epochs 500
python unsupervised.py --n-runs 5 --ppr-topk 128 --sparse --max-epochs 500
python unsupervised.py --n-runs 5 --ppr-topk 128 --sparse --ppr-mode nibble --max-epochs 500

python main.py --n-runs 5
python main.py --n-runs 5 --ppr-topk 128
python main.py --n-runs 5 --ppr-topk 128 --sparse
python main.py --n-runs 5 --ppr-topk 128 --ppr-mode nibble
python main.py --n-runs 5 --ppr-mode nibble --max-epochs 500
python main.py --n-runs 5 --ppr-topk 128 --ppr-mode nibble --sparse --max-epochs 500

# --
# Scaling experiments

mkdir -p data
python utils/dump.py
