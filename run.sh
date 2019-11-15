#!/bin/bash

# run.sh

# --
# Install

conda create -y -n ppnp_env python=3.7
conda activate ppnp_env

conda install -y -c pytorch pytorch=1.2.0
conda install -y scipy=1.3.1

# --
# Run

python main.py
# ^ Runs 5x w/ different seeds