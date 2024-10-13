#!/bin/bash

datasets=('dead_leaves-textures')

for DATASET in ${datasets[@]}
do
    echo "Downloading $DATASET"
    wget -O $DATASET.zip http://data.csail.mit.edu/noiselearning/zipped_data/small_scale/$DATASET.zip
    yes | unzip $DATASET.zip -d $DATASET
    rm $DATASET.zip
done
