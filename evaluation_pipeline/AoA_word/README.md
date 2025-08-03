# Age of Acquisition (AoA) Evaluation Benchmark

This repository provides a comprehensive benchmark for evaluating Age of Acquisition (AoA) in neural language models, following the methodology from [Chang & Bergen (2022). Word Acquisition in Neural Language Models](https://doi.org/10.1162/tacl_a_00444). The benchmark computes word surprisal across different training steps to analyze how language models acquire words during training, enabling comparison with child language development patterns.

## Dataset 

The `dataset_util.py` file handles n-gram context collection and preprocessing for your reference; we provide the extracted eval dataset:

- **NGramContextCollector**: Collects and processes n-gram statistics from datasets
- **TextPreprocessor**: Handles sentence segmentation and tokenization using spaCy
- **Context Filtering**: Selects contexts based on frequency or random sampling
- **Word Validation**: Filters correctly spelled words using spell checking


The `prepare_dataset.py` script handles dataset preparation and context extraction:

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


## Run

The `eval_util.py` file contains core utilities for model evaluation:

- **StepConfig**: Manages Pythia checkpoint configurations with interval sampling
- **JsonProcessor**: Handles JSON serialization with NumPy type conversion
- **StepPathProcessor**: Manages resumable processing across training steps
- **Memory Management**: GPU memory cleanup and optimization functions


The `evaluation_functions.py` file defines the main evaluation framework:

- **StepSurprisalExtractor**: Main class for extracting word surprisal across training steps
- **Model Loading**: Handles checkpoint-specific model and tokenizer loading
- **Surprisal Computation**: Computes surprisal for target words given contexts
- **Learning Curves**: Tracks surprisal changes across training progression

The module outputs JSON-structured data compatible with downstream analysis tools.

The `run.py` file orchestrates the complete evaluation pipeline:

```bash
python run.py \
    --word_path context/stas/c4-en-10k/5/merged.json \
    --model_name EleutherAI/pythia-70m-deduped \
    --use_bos_only \
    --interval 10 \
    --start 14 \
    --end 142
```

Available arguments:
```python
parser.add_argument("-w", "--word_path", type=Path, 
                   default="context/stas/c4-en-10k/5/merged.json",
                   help="Path to target words and contexts JSON file")
parser.add_argument("-m", "--model_name", type=str, 
                   default="gp2",
                   help="HuggingFace model name for evaluation")
parser.add_argument("--eval_lst", type=list, 
                   help="List of evaluation files for subset analysis")
parser.add_argument("--interval", type=int, default=1, 
                   help="Checkpoint sampling interval")
parser.add_argument("--min_context", type=int, default=20, 
                   help="Minimum number of contexts per word")
parser.add_argument("--use_bos_only", action="store_true", 
                   help="Use only beginning-of-sentence context")
parser.add_argument("--start", type=int, default=12, 
                   help="Start index of checkpoint range")
parser.add_argument("--end", type=int, default=12, 
                   help="End index of checkpoint range")
parser.add_argument("--debug", action="store_true", 
                   help="Debug mode (process only first 5 words)")
parser.add_argument("--resume", action="store_true", 
                   help="Resume from existing checkpoint")
```

## Output Format

The pipeline generates structured JSON outputs:


```json
{
  "metadata": {
    "model_name": "gpt2",
    "use_bos_only": true,
    "total_steps": 12,
    "completed_steps": 12
  },
  "results": [
    {
      "step": 0,
      "target_word": "dog",
      "context_id": 0,
      "context": "BOS_ONLY",
      "surprisal": 5.234
    }
  ]
}
```


## Reference Data Requirements

The evaluation requires CDI (MacArthur-Bates Communicative Development Inventory) reference data


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
```
