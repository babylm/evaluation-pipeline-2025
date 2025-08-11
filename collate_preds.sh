#!/bin/bash

MODEL_NAME=$1
BACKEND=$2 # "mlm", "causal", "mntp", "enc_dec_mask", "enc_dec_prefix
TRACK=$3 # "strict-small", "multimodal", "strict", "interaction"

python -m evaluation_pipeline.collate_preds --model_path_or_name=$MODEL_NAME --backend=$BACKEND --fast --track=$TRACK
