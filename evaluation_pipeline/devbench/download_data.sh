#!/bin/bash

DB_PREFIX="evaluation_data/full_eval/devbench"

mkdir $DB_PREFIX
mkdir $DB_PREFIX/assets
mkdir $DB_PREFIX/evals

## THINGS
mkdir $DB_PREFIX/assets/sem-things
mkdir $DB_PREFIX/evals/sem-things
wget "https://osf.io/download/wb36u/" -O $DB_PREFIX/assets/sem-things/things_assets.zip
unzip $DB_PREFIX/assets/sem-things/things_assets.zip -d $DB_PREFIX/assets/sem-things/
rm $DB_PREFIX/assets/sem-things/things_assets.zip
wget "https://osf.io/download/w75eu/" -O $DB_PREFIX/evals/sem-things/spose_similarity.mat
cp evaluation_pipeline/devbench/manifests/sem-things.csv $DB_PREFIX/assets/sem-things/manifest.csv

## TROG
mkdir $DB_PREFIX/assets/gram-trog
mkdir $DB_PREFIX/evals/gram-trog
# The following code is adapted from the DevBench repository: https://github.com/alvinwmtan/dev-bench/blob/master/assets/gram-trog/trog_dl.sh
URL="https://api.github.com/repos/levante-framework/core-tasks/contents/assets/TROG/original?ref=1dba11a50621186daad2ccc4dc2943f1536fb6db"

# Create images directory if it doesn't exist
mkdir -p $DB_PREFIX/assets/gram-trog/images

# Download the JSON file
curl -s "$URL" -o data.json

# Extract the download URLs and download the images
grep -o '"download_url": *"[^"]*"' data.json | sed 's/"download_url": *"\([^"]*\)"/\1/' | while read -r download_url; do
    wget -P $DB_PREFIX/assets/gram-trog/images "$download_url"
done

# Clean up
rm data.json
# End DevBench code

cp evaluation_pipeline/devbench/manifests/gram-trog.csv $DB_PREFIX/assets/gram-trog/manifest.csv
cp evaluation_pipeline/devbench/data_human/gram-trog_human.csv $DB_PREFIX/evals/gram-trog/human.csv

## Visual Vocabulary
mkdir $DB_PREFIX/assets/lex-viz_vocab
mkdir $DB_PREFIX/evals/lex-viz_vocab
ln -s `realpath $DB_PREFIX/assets/sem-things/object_images_CC0/` $DB_PREFIX/assets/lex-viz_vocab/images
cp evaluation_pipeline/devbench/manifests/lex-viz_vocab.csv $DB_PREFIX/assets/lex-viz_vocab/manifest.csv
cp evaluation_pipeline/devbench/data_human/lex-viz_vocab_human.csv $DB_PREFIX/evals/lex-viz_vocab/human.csv
