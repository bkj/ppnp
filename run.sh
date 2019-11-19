#!/bin/bash

# run.sh

# --
# Install

conda create -y -n ppnp_env python=3.7
conda activate ppnp_env

conda install -y -c pytorch pytorch=1.2.0
conda install -y scipy=1.3.1

# --
# Run w/ standard datasets

python main.py

# {'epochs': 1986, 'best_epoch': 1258, 'train_acc': 1.0, 'stop_acc': 0.8460000157356262, 'valid_acc': 0.8511450290679932}
# {'epochs': 1953, 'best_epoch': 1750, 'train_acc': 1.0, 'stop_acc': 0.8480000495910645, 'valid_acc': 0.8541984558105469}
# {'epochs': 2316, 'best_epoch': 2092, 'train_acc': 1.0, 'stop_acc': 0.8460000157356262, 'valid_acc': 0.8442748188972473}
# {'epochs': 1664, 'best_epoch': 1382, 'train_acc': 1.0, 'stop_acc': 0.8460000157356262, 'valid_acc': 0.8587786555290222}
# {'epochs': 1563, 'best_epoch': 1463, 'train_acc': 1.0, 'stop_acc': 0.8440000414848328, 'valid_acc': 0.8534350991249084}

# --
# Run w/ custom datasets

wget http://www.cis.jhu.edu/~parky/TT/Data/DS72784/Graphs/DS72784-graphml-raw.zip 
    -O data/DS72784-graphml-raw.zip