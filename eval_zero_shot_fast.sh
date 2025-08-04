#!/bin/bash

MODEL_PATH=$1
REVISION_NAME=$2
BACKEND=$3
EVAL_DIR=${4:-"evaluation_data/fast_eval"}

if [[ "$BACKEND" == *"enc_dec"* ]]; then
    BACKEND_READ="enc_dec"
else
    BACKEND_READ=$BACKEND
fi

python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task blimp --data_path "${EVAL_DIR}/blimp_fast" --save_predictions --revision_name $REVISION_NAME
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task blimp --data_path "${EVAL_DIR}/supplement_fast" --save_predictions --revision_name $REVISION_NAME
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task ewok --data_path "${EVAL_DIR}/ewok_fast" --save_predictions --revision_name $REVISION_NAME
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task wug_adj --data_path "${EVAL_DIR}/wug_adj_nominalization" --save_predictions --revision_name $REVISION_NAME
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task wug_past --data_path "${EVAL_DIR}/wug_past_tense" --save_predictions --revision_name $REVISION_NAME
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task entity_tracking --data_path "${EVAL_DIR}/entity_tracking_fast" --save_predictions --revision_name $REVISION_NAME
python -m evaluation_pipeline.reading.run --model_path_or_name $MODEL_PATH --backend $BACKEND_READ --data_path "${EVAL_DIR}/reading/reading_data.csv" --revision_name $REVISION_NAME