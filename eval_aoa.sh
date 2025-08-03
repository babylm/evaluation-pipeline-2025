#!/bin/bash
MODEL_NAME=$1
WORD_PATH=${2:-"context/stas/c4-en-10k/5/merged.json"}
OUTPUT_DIR=${3:-"results"}
# Set default parameters
INTERVAL=${INTERVAL:-10}
MIN_CONTEXT=${MIN_CONTEXT:-20}
START=${START:-14}
END=${END:-142}
echo "Running AoA evaluation for model: $MODEL_NAME"
echo "Word path: $WORD_PATH"
echo "Output directory: $OUTPUT_DIR"
# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
# Run the main evaluation
python run.py \
--model_name $MODEL_NAME \
--word_path $WORD_PATH \
--interval $INTERVAL \
--min_context $MIN_CONTEXT \
--start $START \
--end $END