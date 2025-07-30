#!/bin/bash

MODEL_PATH=$1
MODEL_TYPE=$2
IMAGE_MODEL={$3:-$MODEL_PATH}

# Supported MODEL_TYPE values:
# git, flamingo, llava, flava, clip, blip, siglip, bridgetower, vilt, cvcl

# If you need a different MODEL_TYPE, implement it in the `evaluation_pipeline/devbench/model_classes` folder.
# (See other files in that folder for examples.)
# Then add a wrapper to `evaluation_pipeline/devbench/eval.py`.
# Be sure to submit a pull request so others can benefit from your implementation!

python -m evaluation_pipeline.devbench.eval --model $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --image_model $IMAGE_MODEL
