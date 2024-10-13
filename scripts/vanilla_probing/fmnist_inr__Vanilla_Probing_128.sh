#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate neural-graphs


python main.py \
  --exp_name=Vanilla_Probing_128__seed_1 \
  --seed=1 \
  --dataset=fmnist_inr \
  \
  --n_tokens=128 \
  --d_hid=256 \
  --n_layers=6 \
  \
  --gen_type=linear_0_no_acts \
  \
  --batch_size=32 \
  --lr=0.0003 \
  --epochs=30 \
  --eval_every=500 \
  --n_workers=0 \
  --device=cuda

