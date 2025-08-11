# Age of Acquisition (AoA) Evaluation Benchmark

This repository provides a comprehensive benchmark for evaluating Age of Acquisition (AoA) in neural language models, following the methodology from [Chang & Bergen (2022). Word Acquisition in Neural Language Models](https://doi.org/10.1162/tacl_a_00444). The benchmark computes word surprisal across different training steps to analyze how language models acquire words during training, enabling comparison with child language development patterns.



## Quick Start

Please download the `cdi_childes.json` from https://osf.io/ryjfm/, the target file is in `evaluation_data/full_eval/cdi_childes/cdi_childes.json `

Run the evaluation pipeline using `run.py`:
```bash
python run.py \
    --word_path evaluation_data/full_eval/cdi_childes/cdi_childes.json \
    --model_name models/gpt2 \
    --backend causal \
    --track_name non-strict-small \
    --output_dir results
```

Available arguments:
- `--word_path`: Path to target words and their contexts (.json file downloaded from osf link above)
- `--model_name`: model directory with different trainign step checkpoints
- `--backend`: differnt model architectures, options are: ["mlm", "causal", "mntp", "enc_dec_mask", "enc_dec_prefix"]
- `--track_name`: which track to go
- `--output_dir`: the output directory to save the results; final results are stores in output_dir/surprisal.json


The current scripts use the util functions in the follwoing scripts: 

The `eval_util.py` file contains core utilities for model evaluation:

- **StepConfig**: Manages model checkpoint configurations 
- **JsonProcessor**: Handles JSON file loading and saving 
- **StepPathProcessor**: Manages resumable processing across training steps


The `evaluation_functions.py` file defines the main evaluation framework:

- **StepSurprisalExtractor**: Main class for extracting word surprisal across training steps



## Output Format

The pipeline generates structured JSON outputs:


```json
{
  "metadata": {
    "model_name": "gpt2",
    "use_bos_only": false,
    "total_steps": 12,
    "completed_steps": 12
  },
  "results": [
    {
      "step": 0,
      "target_word": "dog",
      "context_id": 0,
      "context": "I raised a ",
      "surprisal": 5.234
    }
  ]
}
```


## Dataset 

We provide the extracted eval dataset, so you do NOT need to re-run the script and extract the dataset from C4-EN. The script is provided here for your reference.  

The `dataset_util.py` file handles n-gram context collection and preprocessing for your reference; we provide the extracted eval dataset, so you do not need to re-run the script and extract the dataset:

- **NGramContextCollector**: Collects and processes n-gram statistics from datasets
- **TextPreprocessor**: Handles sentence segmentation and tokenization using spaCy
- **Context Filtering**: Selects contexts based on frequency or random sampling
- **Word Validation**: Filters correctly spelled words using spell checking


The `prepare_dataset.py` script handles dataset preparation and context extraction.:

```bash
python prepare_dataset.py \
    --words_file matched/oxford-understand.csv \
    --dataset stas/c4-en-10k \
    --window_size 5 \
    --n_contexts 20 \
    --mode frequent
```

Available arguments:
- `--words_file`: Path to target words file (CSV or TXT)
- `--output_path`: Output directory for extracted contexts
- `--dataset`: HuggingFace dataset name for context extraction
- `--split`: Dataset split to use (default: "train")
- `--window_size`: Minimum context window size
- `--n_contexts`: Number of contexts per word
- `--mode`: Selection mode ("frequent" or "random")




## Metric computation

The evaluation computes the cruve fitness between model and child "age-of-acquisition" for a certain 

compared with child  CDI (MacArthur-Bates Communicative Development Inventory) reference data (link:https://wordbank.stanford.edu)


## Citation

If you use this benchmark, please cite:

```bibtex
@article{chang2022word,
  title={Word Acquisition in Neural Language Models},
  author={Chang, Tyler A. and Bergen, Benjamin K.},
  journal={Transactions of the Association for Computational Linguistics},
  volume={10},
  pages={1--16},
  year={2022},
  publisher={MIT Press}
}

@article{frank2017wordbank,
    title={Wordbank: An open repository for developmental vocabulary data},
    author={Frank, Michael and Braginsky, Mika and Yurovsky, Daniel and Marchman, Virginia},
    journal={Journal of Child Language},
    volume={44},
    number={3},
    pages={677--694},
    year={2017},
    publisher={Cambridge University Press}
}

```
