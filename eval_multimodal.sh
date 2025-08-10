#!/bin/bash

MODEL_PATH=$1
BACKEND=$2
MODEL_TYPE=$3
IMAGE_MODEL=$4
EVAL_DIR=${5:-"evaluation_data/full_eval"}

if [[ "$BACKEND" == *"enc_dec"* ]]; then
    BACKEND_READ="enc_dec"
else
    BACKEND_READ=$BACKEND
fi

echo $BACKEND_READ

# Winoground and VQA
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task vqa --data_path "evaluation_data/full_eval/vqa_filtered" --save_predictions --images_path=HuggingFaceM4/VQAv2 --image_split=validation --batch_size=1
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task winoground --data_path "evaluation_data/full_eval/winoground_filtered" --save_predictions --images_path=facebook/winoground --image_split=test --batch_size=1

# Devbench
# Supported MODEL_TYPE values:
# git, flamingo, llava, flava, clip, blip, siglip, bridgetower, vilt, cvcl

# If you need a different MODEL_TYPE, implement it in the `evaluation_pipeline/devbench/model_classes` folder.
# (See other files in that folder for examples.)
# Then add a wrapper to `evaluation_pipeline/devbench/eval.py`.
# Be sure to submit a pull request so others can benefit from your implementation!

python -m evaluation_pipeline.devbench.eval --model $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --image_model $IMAGE_MODEL



