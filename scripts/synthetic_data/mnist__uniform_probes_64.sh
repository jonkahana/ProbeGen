#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate neural-graphs

python main__real_data.py \
  --exp_name=Uniform_Probes_64__seed_1 \
  --seed=1 \
  --dataset=mnist_inr \
  \
  --n_tokens=64 \
  --d_hid=256 \
  --n_layers=6 \
  \
  --gen_type=uniform_coords__no_opt \
  \
  --batch_size=32 \
  --lr=0.0003 \
  --epochs=30 \
  --eval_every=500 \
  --n_workers=4 \
  --device=cuda
