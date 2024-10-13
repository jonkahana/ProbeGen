#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate neural-graphs

python main.py \
  --exp_name=Vanilla_Probing_64__seed_1 \
  --seed=1 \
  --dataset=nfn_cnn_zoo \
  \
  --n_tokens=64 \
  --d_hid=256 \
  --n_layers=6 \
  \
  --gen_type=deep_linear_0 \
  \
  --batch_size=32 \
  --lr=0.0003 \
  --epochs=150 \
  --eval_every=500 \
  --n_workers=4 \
  --device=cuda
