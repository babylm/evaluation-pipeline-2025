#!/bin/bash

MODEL_PATH=$1
BACKEND=$2
TRACK=$3
EVAL_DIR=${4:-"evaluation_data/fast_eval"}

for i in {1..9}; do
    checkpoint="chck_${i}M"
    echo "Evaluating checkpoint ${checkpoint}"
    bash eval_zero_shot_fast.sh $MODEL_PATH $checkpoint $BACKEND $EVAL_DIR
done

for ((i=10; i<=100; i+=10)); do
    checkpoint="chck_${i}M"
    echo "Evaluating checkpoint ${checkpoint}"
    bash eval_zero_shot_fast.sh $MODEL_PATH $checkpoint $BACKEND $EVAL_DIR
done

# Conditional on whether the track is strict-small
if [[ "$3" != "strict-small" ]]; then
    for ((i=200; i<=1000; i+=100)); do
	checkpoint="chck_${i}M"
	echo "Evaluating checkpoint ${checkpoint}"
	bash eval_zero_shot_fast.sh $MODEL_PATH $checkpoint $BACKEND $EVAL_DIR
    done
fi
