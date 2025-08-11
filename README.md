# 2025 BabyLM Challenge Evaluation Pipeline

![BabyLM Challenge](assets/babylm.png)

## Overview

This code provides the backend for the BabyLM Challenge's evaluation pipeline. This year we decided to implement it from scratch. It currently supports 3 different evaluation types: fine-tuning (sequence), sentence-level zero-shot logit calculations, and word level logit calculations (although the last one is implemented for a specific task).

A new addition this year is that we have two evaluation types: **fast** evaluation uses a smaller set of evaluation samples, allows for quick testing of your models, and is what you will report performance on for the intermediate model checkpoints. The **full** evaluation should be run on your final model.

If you have questions about or suggestions for this code, please open an issue and consider [joining our Slack](https://join.slack.com/t/babylmchallenge/shared_invite/zt-2gqgqaumu-5ebxxADuT561aT_ooKbT1Q). Join the `#evaluation` channel, which is dedicated to support for use of this repository.

We also welcome pull requests!

## New Tasks

We have added a couple of new tasks to the text-only evaluation suite:
- **Derivational Morphology Reveals Analogical Generalization in Large Language Models** [(Hofmann et al., 2024)](https://arxiv.org/abs/2411.07990) - *Tests morphological generalization in LMs through an adjective nominalization task.*
- **Entity Tracking in Language Models** [(Kim & Schuster, 2023)](https://aclanthology.org/2023.acl-long.213/) - *Tests entity state tracking in LMs. Note: We have changed the evaluation of this task to evaluate LMs' ability to assign the highest probability to the correct continuation (akin to BLiMP and EWoK) rather than generate the correct completion itself as was originally done, to allow for simpler, zero-shot evaluation.*
- **Cloze probability, predictability ratings, and computational estimates for 205 English sentences, aligned with existing EEG and reading time data** [(De Varda et al., 2023)](https://link.springer.com/article/10.3758/s13428-023-02261-8) - *Connects LM predictions to human reading times, allowing us to assess to what extent LM processing is aligned with human language processing.*

From last years iteration we are still including BLiMP, EWoK, and GLUE for evaluation.

## Hidden Tasks

- **Counting the Bugs in ChatGPT's Wugs: A Multilingual Investigation into the Morphological Capabilities of a Large Language Model** [(Weissweiler et al., 2023)](https://arxiv.org/abs/2310.15113/)  (WUG_PAST) introduce a task for testing morphological generalization, based on a past tense formation task of nonce ("wug") words. We evaluate this task against human predictions: from the participant responses we derive a distribution over possible inflections. This way we can test the "humanlikeness" of a model's generalization behavior, by correlating the model's probability for each inflection against the relative amount of responses we can measure whether they follow similar morphological patterns. Prior to the release of the hidden tasks, we had already included the task of [Hofmann et al. (2024)](https://arxiv.org/abs/2411.07990) (WUG_ADJ), who build on the work of Weissweiler et al. and introduce a similar task for adjective nominalization, in which a nonce ("wug") adjective is nominalized as either an -ity noun or a -ness noun. Performance is measured in the same way, by correlating model probabilities to human judgments. Note that we have changed the evaluation of this task from an "accuracy" based metric to the correlation metric of the WUG_PAST task!
- **COMPS: Conceptual Minimal Pair Sentences for testing Robust Property Knowledge and its Inheritance in Pre-trained Language Models** [(Misra et al., 2023)](https://aclanthology.org/2023.eacl-main.213/) (COMPS) [Misra et al. (2023)](https://aclanthology.org/2023.eacl-main.213/) introduce a task for testing the property knowledge of language models and whether they can infer that properties of superordinate concepts are inherited by subordinate concepts, each represented by nonce words. The dataset is composed of minimal pair sentences and models are evaluated by whether they assign higher probability to the correct sentence.
- **Age of Acquisition (AoA) Evaluation Benchmark** [(Chang & Bergen, 2022)](https://doi.org/10.1162/tacl_a_00444) This benchmark tracks word surprisal across training checkpoints to extract learning curves and compute ages of acquisition for vocabulary items. We compute surprisal scores as the negative log probability of target words given their contexts in a test set across training steps [c4-en-10k](https://huggingface.co/datasets/stas/c4-en-10k), and then fit sigmoid functions to the learning trajectory of each word. In the end, the benchmark enables direct comparison between language model and child language development by computing correlation scores between model-derived AoA and human AoA data from the MacArthur-Bates Communicative Development Inventory (CDI) [(Frank et al., 2017)](https://wordbank.stanford.edu).



## Install

> [!Note]
> The package is currently not installable given that it is a first version. Instead we recommend installing the packages needed and using it as a python module, i.e. to run a part of the pipeline (for example finetuning) you would to do: `python -m evaluation_pipeline.finetune.run ...` from the root folder (the folder that contains the evaluation_pipeline folder).

To be able to use the pipeline you need to install the `requirements.txt` packages.

> [!Warning]
> These packages were installed using Python 3.13, in case some of the packages are not compatible with your Python version (either because the version is too recent or is not supported). In that case, you could either update your Python version or pip/conda install the following packages: `transformers`, `torch`, `scikit-learn`, `numpy`, `pandas`, `statsmodels`, `datasets`, `wandb`, and `nltk`.

## File Structure

```
evaluation_pipeline
├── __init__.py
├── ewok
│   ├── dl_and_filter.py
│   └── vocab.txt
├── finetune
│   ├── README.md
│   ├── __init__.py
│   ├── classifier_model.py
│   ├── dataset.py
│   ├── run.py
│   ├── trainer.py
│   └── utils.py
├── reading
│   ├── README.md
│   ├── __init__.py
│   ├── evaluation_functions.py
│   └── run.py
└── sentence_zero_shot
    ├── README.md
    ├── __init__.py
    ├── compute_results.py
    ├── dataset.py
    ├── read_files.py
    └── run.py
```

## Results File Structure

```
results
├── model_name1
│   ├── main
│   │   ├── finetune
│   │   │   ├── boolq
│   │   │   │   ├── predictions.jsonl
│   │   │   │   └── results.txt
│   │   │   ├── mnli
│   │   │   └── ...
│   │   └── zero_shot
│   │       ├── causal
│   │       │       ├── blimp
│   │       │       │   ├── blimp_filtered
│   │       │       │   │   ├── best_temperature_report.txt
│   │       │       │   │   └── predictions.jsonl
│   │       │       │   ├── supplement_filtered
│   │       │       │   ├── blimp_fast
│   │       │       │   └── ...
│   │       │       ├── ewok
│   │       │       └── ...
│   │       └── ...
│   ├── revision_name1
│   └── revision_name2
├── model_name2
│   ├── ...
└── ...
```

## Data

Download the `evaluation_data` folder in [this OSF directory](https://osf.io/ryjfm/). Place it in the root directory of this repository.

Due to large file sizes and license restrictions, we do not provide images in the OSF directory of the evaluation tasks for the multimodal track. Instead, we link to HuggingFace datasets, two of which require approval (which is immediate). Go to this URL to download this dataset:
- [Winoground](https://huggingface.co/datasets/facebook/winoground)

Furthermore, the EWoK data requires agreeing to the terms & conditions on the HuggingFace Hub, which can be agreed to here:
- [EWoK](https://huggingface.co/datasets/ewok-core/ewok-core-1.0)

For the EWoK fast dataset found in the [OSF](https://osf.io/ryjfm), the password to unzip the file is: BabyLM2025

On both pages, make sure you're logged in to your HuggingFace account, and request approval. Then, in your terminal, log in to your account using `huggingface-cli login`, and enter your HuggingFace login token.

For EWoK data, run `python -m evaluation_pipeline.ewok.dl_and_filter` from the root directory of this repository.

For the fast EWoK data, we provide a password-protected ZIP file called `ewok_fast.zip`.

For the DevBench data make sure to run:
```bash
./evaluation_pipeline/devbench/download_data.sh
```

## Evaluation 
This year, we provide different sets of evaluation tasks for different tracks.

### Text-only evaluation
If you are participating in one of the text-only tracks (Strict or Strict-small) or interaction track, use these instructions.
#### Zero-shot evaluation

Use the following shell script to evaluate on the full zero-shot evaluations:
```bash
./eval_zero_shot.sh <path_to_model> <architecture (causal/mntp/mlm/enc_dec_mask/enc_dec_prefix)> <eval_dir (optional, default:evaluation_data/full_eval)>
```

Use the following shell script to evaluate on the fast zero-shot evaluations:
```bash
./eval_zero_shot_fast.sh <path_to_model> <revision_name> <architecture (causal/mntp/mlm/enc_dec_mask/enc_dec_prefix)> <eval_dir (optional, default:evaluation_data/fast_eval)>
```

> [!Note]
> The revision name indicates the checkpoint to use (for example in the gpt-bert baselines `chck_1M` is the model trained for about 1M words).

These will work out of the box if you use a HuggingFace-based model. In the case you are not, you can either go to the `hf_conversion_tutorial` folder to create a HF repository or adapt the code to work with a pure PyTorch implementation (it should not be too complicated). The implementation currently only supports three types of trained langauge modeling tasks: causal, mlm, and mntp (mlm shifted similarly to causal). If another objective (like diffusion for example) was used to train the models, you will need to edit the files.

In addition we have added a script called `eval_zero_shot_fast_all_revisions.sh` to evaluate all the checkpoints in a single call. To make sure this work with your naming scheme, make sure to edit the for-loops:
```bash
for i in {1..9}; do
    checkpoint="chck_${i}M"
```
To make sure they fit your checkpoint naming scheme. To run the script type the following:
```bash
./eval_zero_shot_fast_all_revisions.sh <path_to_model> <architecture (causal/mntp/mlm/enc_dec_mask/enc_dec_prefix)> <track> <eval_dir (optional, default:evaluation_data/fast_eval)>
```
> [!NOTE]
> The code assumes that you trained on the entire budget (100M words for strict-small and 1B words for strict). Please change this if this is not the case.

> [!Important]
> For the Encoder-Decoder backend, there exists two different evaluation styles for the sentence zero_shot evaluations. Those are either prefix `enc_dec_prefix` or bi-direction/fill-in-the-gap `enc_dec_mask`. The style of input to the encoder and decoder are the following:
> - **Encoder**: `CLS` text with a mask / prefix ending with a mask `EOS` (the code checks whether a `CLS` and `EOS` are defined, if not, they are not added).
> - **Decoder**: `BOS` / `MASK` text to predict in case of prefix (The choice of `BOS` or `MASK` depends on the architecture).

#### AoA evaluation
Use the following shell script to evaluate on the fast zero-shot evaluations:
```bash
./eval_aoa.sh <path_to_model> <architecture (causal/mntp/mlm/enc_dec_mask/enc_dec_prefix)> <track> <eval_dir (optional, default:evaluation_data/full_eval/cdi_childes/cdi_childes.json)> <output_dir (optional, default:results)>
```

#### Fine-tuning or low-rank adapter training

Like last year, we provide a script to support fine-tuning on all tasks:
```bash
./eval_finetune.sh <path_to_model> <learning_rate (optional, default: 3e-5)> <batch_size (optional, default: 32)> <max_epochs (optional, default: 10)> <seed (optional, default: 42)>
```
This will fine-tune your model on all (Super)GLUE tasks.

> [!Note]
> To make finetuning evaluations more efficient, this year we randomly subsampled MNLI and QQP to 10k training samples. We found that 10k training samples for these datasets is sufficient for minimizing variance due to randomness. We also removed CoLA, SST2, MNLI-mm, and QNLI because they are highly correlated with other datasets.

> [!Note]
> The hyperparameters are shared through all tasks, if you want to have different ones for every task, you will either need to edit the file or run the python command found in the file from the terminal.

> [!Note]
> There are more hyperparameters you can play with! Checkout the README in the finetune folder of the evaluation_pipeline for more information. In addition, you can edit also edit the classifier head.

> [!Important]
> By default the code only expects encoders or decoders. Add the flag `--enc_dec` to the python calls to use an encoder-decoder. For the fine-tuning, the default classification models does the following:
> - Passes the inputs to the Encoder
> - Initializes a 1 token sequence containing the start token (defined in the `config.json` of the model as `decoder_start_token_id`) to pass to the decoder.
> - Does the Decoder forward pass
> - Takes the encoding of the start token and passes it to the classification head.
> This is the way we decided to implement the finetuning, however, this can be adapted by yourselves.

> [!Note]
> The classification head and the way the model does classification can be adapted by you. Change the code found in the `evaluation_pipeline/finetune/classifier_model.py` file to do this.

<!---
Here are the hyperparameters used for fine-tuning for all tasks. Feel free to modify these, or to set task-specific hyperparameters:
| Hyperparameter | Value |
| -------------- | ----- |
| Initial learning rate | 5e-5 |
| Batch size | 32 |
| Maximum epochs | 10 |
| Seed | 42 |
--->

### Multimodal evaluation

If you are participating in the multimodal track, use these instructions.

First, run your models on the text-only evaluations, including BLiMP, the BLiMP supplement, EWoK, and (Super)GLUE. As long as your model is compatible with the AutoModelForCausalLM and AutoModelForSequenceClassification classes, you can use the same instructions as above to evaluate on the text-only tasks.

In addition, use the following command to evaluate on Winoground (where we use an unpaired text score), VQA (accuracy with 7 distractors) and DevBench:
```bash
./eval_multimodal.sh <path_to_model> <architecture (causal/mntp/mlm/enc_dec_mask/enc_dec_prefix)> <model_type (git/flamingo/llava/flava/clip/blip/siglip/bridgetower/vilt/cvcl)> <image_model>
```
The model types are used for DevBench, if you need a different model_type, implement it in the `evaluation_pipeline/devbench/model_classes` folder. (See other files in that folder for examples.) Then add a wrapper to `evaluation_pipeline/devbench/eval.py`. Be sure to submit a pull request so others can benefit from your implementation!

## Baselines
The baseline models are available from the BabyLM Community huggingface page here: https://huggingface.co/BabyLM-community .

For the strict and strict-small tracks, we release the following baselines: [GPT-BERT](https://arxiv.org/pdf/2410.24159), the winning submission from the 2024 iteration, and GPT-2 Small as a purely autoregressive baseline. Models containing `-100m` are for the strict track; those containing `-10m` are for strict-small.

For the multimodal tracks, we release [Flamingo](https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf) and [GIT](https://openreview.net/pdf?id=b4tMhpN0JC) baselines.

For the interaction track, we release two baselines: An "RLHF" baseline, where a model pre-trained on the BabyLM corpus is further finetuned via [PPO](https://arxiv.org/pdf/1707.06347) to maximize a scalar reward mimicking caregiver responses, and a "Preference Optimization" baseline, where a model is optimized via [SimPO](https://arxiv.org/pdf/2405.14734) to prefer teacher corrections over its own generated outputs. More details are available in Section 4.5 of the [call for papers](https://arxiv.org/pdf/2502.10645?).

<!---
Here are scores for each model on each evaluation task. Each task score is an unweighted mean of each subtask score within that task. We also show macroaverages, which are simply means of each task score (i.e., means across a row of the table). NOTE: for GLUE, we average *accuracies* for all tasks except QQP and MRPC (where we use F1 scores). See end of README for more detailed score breakdowns.

> [!Note]
> The evaluations below are run on the final model (the one trained for 10 epochs (100M words in Strict-small and 1B words in Strict and Interaction)).

**Strict-small Track (10M)**

*Causal*

| Model | BLiMP | BLiMP Supplement | EWoK | Reading (Eye Tracking) | Reading (Self-Paced Reading Time) | Entity Tracking | WUGs | GLUE | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | **71.66** | **63.21** | 49.49 | **9.89** | **3.45** | **33.96** | 43.00 |
|GPT-BERT (Mixed) | 69.62 | 61.56 | **50.23** | 9.50 | 3.37 | 22.27 | 45.00 |
|GPT-BERT (Masked-focus) | 65.22 | 59.49 | 49.47 | 9.52 | 3.44 | 30.60 | **68.00** |
|GPT-2 Small | 67.29 | 59.09 | 49.80 | 0.13 | 0.04 | 18.92 | 39.00 |

*MNTP/MLM*

| Model | BLiMP | BLiMP Supplement | EWoK | Reading (Eye Tracking) | Reading (Self-Paced Reading Time) | Entity Tracking | WUGs | GLUE | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 69.07 | **64.33** | 49.62 | 9.47 | **3.48** | 39.17 | 43.00 |
|GPT-BERT (Mixed) | **71.29** | 63.30 | 49.93 | **9.78** | 3.33 | 39.95 | 16.00 |
|GPT-BERT (Masked-focus) | 70.36 | 63.71 | **49.95** | 9.40 | 3.37 | **40.02** | **57.5** |


**Strict Track (100M)**

*Causal*

| Model | BLiMP | BLiMP Supplement | EWoK | Reading (Eye Tracking) | Reading (Self-Paced Reading Time) | Entity Tracking | WUGs | GLUE | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | **79.29** | **70.42** | **52.32** | 8.36 | 3.02 | | 43.5 |
|GPT-BERT (Mixed) | 78.37 | 69.23 | 51.79 | 8.74 | **3.59** | | 39.5 |
|GPT-BERT (Masked-focus) | 74.56 | 63.63 | 51.57 | **8.80** | 3.30 | | **59.00** |
|GPT-2 Small | 75.07 | 64.96 | 51.22 | 0.47 | 0.06 | 30.11 | 42.5 |

*MNTP/MLM*

| Model | BLiMP | BLiMP Supplement | EWoK | Reading (Eye Tracking) | Reading (Self-Paced Reading Time) | Entity Tracking | WUGs | GLUE | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | | | | 9.08 | 3.12 | | 37.5 |
|GPT-BERT (Mixed) | | | | 9.15 | **3.43** | | 37.00 |
|GPT-BERT (Masked-focus) | | | | **9.34** | 3.34 | | **55.00** |


**Interaction Track**

| Model | BLiMP | BLiMP Supplement | EWoK | Reading (Eye Tracking) | Reading (Self-Paced Reading Time) | Entity Tracking | WUGs | GLUE | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|Preference Optimization | 71.91 | 64.85 | 52.44 | 0.5 | 0.01 | 27.71 | 38.5 |

**Multimodal Track**

Here, we show the performance of the Flamingo and GIT baselines on all text-only *and* multimodal tasks. We also show how performance changes on the multimodal tasks when images are not provided to the model during evaluation (i.e., we use the same trained text-and-image model, but modify the evaluation setup to remove any visual information).

| Model | BLiMP | BLiMP Supplement | EWoK | Reading (Eye Tracking) | Reading (Self-Paced Reading Time) | Entity Tracking | WUGs | GLUE | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| Model | Winoground | VQA | DevBench | *Vision Macroaverage* |
| --- | --- | --- | --- | --- |
| Flamingo | 51.6 | 52.3 | 59.5 | *54.5* |
| Flamingo (no vision) | 50.0 | 45.0 | - | *47.5(\*)* |
| GIT | 55.5 | 54.1 | 51.1 | *53.6* |
| GIT (no vision) | 50.0 | 48.4 | - | *49.2(\*)* |

(*) Not directly comparable to other macroaverages, since DevBench scores without vision are not well-defined. These rows are more useful as comparison points for Winoground and VQA with and without visual signals.
--->

## Submission Requirements

### Checkpoints

We require to provide the following checkpoints for all tracks:
- The model checkpoint every 1M words for the first 10M words (total count not unique) trained on (1M, 2M, ..., 10M)
- The model checkpoint every 10M words for the first 100M words (total count not unique) trained on (10M, 20M, ..., 100M)

For all the tracks except strict-small:
- The model checkpoint every 100M words for the first 1000M words (total count not unique) trained on (100M, 200M, ..., 1000M)

For the submiting the checkpoints we encourage creating multiple branches in a HuggingFace repository containing each checkpoint (a brach for the 7M checkpoint could be called chck_7M). Checkout [this repository](https://huggingface.co/BabyLM-community/babylm-baseline-10m-gpt-bert-mixed) for an example.

### Submission Evaluation

This year we require both the evaluation of the final model, on a set of full evaluation (which include the finetuning). And the evaluation of all the checkpoints mentioned above (or up until the one you trained, if for example you only train for 20M words then we require: 1M, 2M, ..., 10M, 20M) on a set of fast tasks, that do not include finetuning and are a subsampled set of the full evaluations.

### Submission Format
To create a submission file to the leaderboard or challenge use the following command:
```bash
bash collate_preds.sh NAME_OF_YOUR_MODEL BACKEND SUBMISSION_TRACK
```
The `--fast` flag in the script is to add the fast evaluation results of each checkpoint to the collation and make the submission valid for the BabyLM challenge.
> [!NOTE]
> Currently the code assumes that the checkpoint naming scheme is `chck_*M`. If you use a different naming scheme make sure to edit lines 16 and 17 of the `collate_preds.py` file. In addition the code assumes that the training is done on the maximum number of words possible, please edit this if it is not the case for you.

Make sure that all the evaluations have been run before collating them.


The submission is a JSON file where the first key represents the benchmark, the next the task of the benchmark, and the value either the label as a number (for GLUE tasks) or the predicted sentences for the zero-shot tasks. An example is:

```
{"glue": {"boolq": {"predictions": [{"id": "boolq_0", "pred": 0}, {"id": "boolq_1", "pred": 1}, ...]}}}
```

### Leaderboard
You can find the leaderboard for the non-hidden tasks [here](https://huggingface.co/spaces/BabyLM-community/babylm-leaderboard-2025-all-tasks).


### Steps to submission

For all challenge tracks, you should run the 'eval_zero_shot.sh' and 'eval_finetuning.sh' scripts on the final checkpoint. The 'eval_aoa.sh' and 'eval_zero_shot_fast_all_revisions.sh' scripts should likewise be run for all challenge tracks, but these conduct evaluations for all model checkpoints. If you are in the multimodal track, you should run 'eval_multimodal.sh' as well. Once you have run all scripts for evaluation, you should run 'collate_preds.sh' and submit to the [leaderboard](https://huggingface.co/spaces/BabyLM-community/babylm-leaderboard-2025-all-tasks).


----
## Visualizing Results

You can seamlessly visualize and analyze the results of your evaluation harness runs using Weights & Biases (W&B).

### Weights and Biases

To run your finetuning code with Weights and Biases, you need to pass the `--wandb` flag and set at minimum the `--wandb_entity` to your W&B user/project. You can also set the `--wandb_project` to specify which project the run should be logged to. By default this is *BabyLM Finetuning*. You can also set the name of the run with the `exp_name` flag, by default this is *model_name_task_seed*.

### Support

The best way to get support is to open an issue on this repo or join the [BabyLM slack](https://join.slack.com/t/babylmchallenge/shared_invite/zt-2gqgqaumu-5ebxxADuT561aT_ooKbT1Q). Join the `#evaluation-pipeline` channel, which is dedicated to support for use of this repository.

## Optional Extras
Extras dependencies can be installed via `pip install -e ".[NAME]"`

| Name          | Use                                   |
|---------------|---------------------------------------|
| anthropic     | For using Anthropic's models          |
| deepsparse     | For running NM's DeepSparse models    |
| dev           | For linting PRs and contributions     |
| gptq          | For loading models with GPTQ          |
| hf_transfer   | For speeding up HF Hub file downloads |
| ifeval        | For running the IFEval task           |
| neuronx       | For running on AWS inf2 instances     |
| mamba         | For loading Mamba SSM models          |
| math          | For running math task answer checking |
| multilingual  | For multilingual tokenizers           |
| openai        | For using OpenAI's models             |
| optimum       | For running Intel OpenVINO models     |
| promptsource  | For using PromptSource prompts        |
| sentencepiece | For using the sentencepiece tokenizer |
| sparseml      | For using NM's SparseML models        |
| testing       | For running library test suite        |
| vllm          | For loading models with vLLM          |
| zeno          | For visualizing results with Zeno     |
|---------------|---------------------------------------|
| all           | Loads all extras (not recommended)    |


## Cite as
Please cite both of the following papers if you use this repository in your work:
```
@misc{charpentier2025babylmturns3papers,
      title={BabyLM Turns 3: Call for papers for the 2025 BabyLM workshop}, 
      author={Lucas Charpentier and Leshem Choshen and Ryan Cotterell and Mustafa Omer Gul and Michael Hu and Jaap Jumelet and Tal Linzen and Jing Liu and Aaron Mueller and Candace Ross and Raj Sanjay Shah and Alex Warstadt and Ethan Wilcox and Adina Williams},
      year={2025},
      eprint={2502.10645},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.10645}, 
}
```

<!---
## Detailed Score Breakdown

**Strict-small Track (10M)**

*Causal*

*GLUE (Default: Acc.)*
| Model | BoolQ | MNLI | MRPC (F1) | MultiRC | QQP (F1) | RTE | WSC | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*BLiMP (Acc.)*
| Model | Ellipsis | Binding | Quantifiers | Argument Structure | Subject Verb Agreement | Anaphor Agreement | Filler Gap Dependency |
| --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*BLiMP Contd. (Acc.)*
| Model | S-Selection | Determiner Noun Agreement | NPI Licensing | Island Effects | Control Raising | Irregular Forms | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*BLiMP Supplement (Acc.)*
| Model | Hypernym | QA Congruence (easy) | QA Congruence (tricky) | Subj.-Aux. Inversion | Turn Taking | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*EWoK (Acc.)*
| Model | Agent Properties | Material Dynamics | Material Properties | Physical Dynamics | Physical Interactions | Physical Relations | Quantitative Properties | Social Interactions | Social Properties | Social Relations | Spatial Relations | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*Entity Tracking (Acc.)*
| Model | Regular | Ambiguous Reference | Move Contents | *Macroaverage* |
| --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |


---

*MNTP/MLM*

*GLUE (Default: Acc.)*
| Model | BoolQ | MNLI | MRPC (F1) | MultiRC | QQP (F1) | RTE | WSC | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*BLiMP (Acc.)*
| Model | Ellipsis | Binding | Quantifiers | Argument Structure | Subject Verb Agreement | Anaphor Agreement | Filler Gap Dependency |
| --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*BLiMP Contd. (Acc.)*
| Model | S-Selection | Determiner Noun Agreement | NPI Licensing | Island Effects | Control Raising | Irregular Forms | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*BLiMP Supplement (Acc.)*
| Model | Hypernym | QA Congruence (easy) | QA Congruence (tricky) | Subj.-Aux. Inversion | Turn Taking | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*EWoK (Acc.)*
| Model | Agent Properties | Material Dynamics | Material Properties | Physical Dynamics | Physical Interactions | Physical Relations | Quantitative Properties | Social Interactions | Social Properties | Social Relations | Spatial Relations | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*Entity Tracking (Acc.)*
| Model | Regular | Ambiguous Reference | Move Contents | *Macroaverage* |
| --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

---
---
**Strict Track (100M)**

*Causal*

*GLUE (Default: Acc.)*
| Model | BoolQ | MNLI | MRPC (F1) | MultiRC | QQP (F1) | RTE | WSC | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*BLiMP (Acc.)*
| Model | Ellipsis | Binding | Quantifiers | Argument Structure | Subject Verb Agreement | Anaphor Agreement | Filler Gap Dependency |
| --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*BLiMP Contd. (Acc.)*
| Model | S-Selection | Determiner Noun Agreement | NPI Licensing | Island Effects | Control Raising | Irregular Forms | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*BLiMP Supplement (Acc.)*
| Model | Hypernym | QA Congruence (easy) | QA Congruence (tricky) | Subj.-Aux. Inversion | Turn Taking | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*EWoK (Acc.)*
| Model | Agent Properties | Material Dynamics | Material Properties | Physical Dynamics | Physical Interactions | Physical Relations | Quantitative Properties | Social Interactions | Social Properties | Social Relations | Spatial Relations | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*Entity Tracking (Acc.)*
| Model | Regular | Ambiguous Reference | Move Contents | *Macroaverage* |
| --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

---

*MNTP/MLM*

*GLUE (Default: Acc.)*
| Model | BoolQ | MNLI | MRPC (F1) | MultiRC | QQP (F1) | RTE | WSC | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*BLiMP (Acc.)*
| Model | Ellipsis | Binding | Quantifiers | Argument Structure | Subject Verb Agreement | Anaphor Agreement | Filler Gap Dependency |
| --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*BLiMP Contd. (Acc.)*
| Model | S-Selection | Determiner Noun Agreement | NPI Licensing | Island Effects | Control Raising | Irregular Forms | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*BLiMP Supplement (Acc.)*
| Model | Hypernym | QA Congruence (easy) | QA Congruence (tricky) | Subj.-Aux. Inversion | Turn Taking | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*EWoK (Acc.)*
| Model | Agent Properties | Material Dynamics | Material Properties | Physical Dynamics | Physical Interactions | Physical Relations | Quantitative Properties | Social Interactions | Social Properties | Social Relations | Spatial Relations | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

*Entity Tracking (Acc.)*
| Model | Regular | Ambiguous Reference | Move Contents | *Macroaverage* |
| --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 
|GPT-BERT (Mixed) | 
|GPT-BERT (Masked-focus) |

---
---
**Interaction Track**

*GLUE (Default: Acc.)*
| Model | BoolQ | MNLI | MRPC (F1) | MultiRC | QQP (F1) | RTE | WSC | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |


*BLiMP (Acc.)*
| Model | Ellipsis | Binding | Quantifiers | Argument Structure | Subject Verb Agreement | Anaphor Agreement | Filler Gap Dependency |
| --- | --- | --- | --- | --- | --- | --- | --- |


*BLiMP Contd. (Acc.)*
| Model | S-Selection | Determiner Noun Agreement | NPI Licensing | Island Effects | Control Raising | Irregular Forms | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- |


*BLiMP Supplement (Acc.)*
| Model | Hypernym | QA Congruence (easy) | QA Congruence (tricky) | Subj.-Aux. Inversion | Turn Taking | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- |


*EWoK (Acc.)*
| Model | Agent Properties | Material Dynamics | Material Properties | Physical Dynamics | Physical Interactions | Physical Relations | Quantitative Properties | Social Interactions | Social Properties | Social Relations | Spatial Relations | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |


*Entity Tracking (Acc.)*
| Model | Regular | Ambiguous Reference | Move Contents | *Macroaverage* |
| --- | --- | --- | --- | --- |


---
---
**Multimodal Track**


*GLUE (Default: Acc.)*
| Model | BoolQ | CoLA (MCC) | MNLI | MNLI-mm | MRPC (F1) | MultiRC | QNLI | QQP (F1) | RTE | SST-2 | WSC | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Flamingo | 69.1 | 36.7 | 75.8 | 76.4 | 84.2 | 60.5 | 83.8 | 85.1 | 60.4 | 90.4 | 42.3 | *69.5* |
| GIT | 67.0 | 0.0 | 75.2 | 74.5 | 82.2 | 58.6 | 81.9 | 84.7 | 62.6 | 88.8 | 45.3 | *65.5* |


*GLUE (Default: Acc.)*
| Model    | BoolQ | MNLI | MRPC (F1) | MultiRC | QQP (F1) | RTE  | WSC  | *Macroaverage* |
| :------- | :---- | :--- | :-------- | :------ | :------- | :--- | :--- | :--- |
| Flamingo | 69.1  | 75.8 | 84.2      | 60.5    | 85.1     | 60.4 | 42.3 | 68.2 |
| GIT      | 67.0  | 75.2 | 82.2      | 58.6    | 84.7     | 62.6 | 45.3 | 67.9 |


*BLiMP Supplement (Acc.)*
| Model | Hypernym | QA Congruence (easy) | QA Congruence (tricky) | Subj.-Aux. Inversion | Turn Taking | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- |
| Flamingo | 48.8 | 75.0 | 43.6 | 86.2 | 71.4 | *65.0* |
| GIT | 48.9 | 67.2 | 49.7 | 86.6 | 61.1 | *62.7* |

*EWoK (Acc.)*
| Model | Agent Properties | Material Dynamics | Material Properties | Physical Dynamics | Physical Interactions | Physical Relations | Quantitative Properties | Social Interactions | Social Properties | Social Relations | Spatial Relations | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Flamingo | 50.8 | 61.0 | 55.3 | 53.3 | 50.9 | 50.1 | 52.5 | 54.4 | 49.4 | 49.6 | 52.2 | *52.7* |
| GIT | 51.0 | 61.9 | 51.2 | 54.2 | 50.2 | 49.9 | 52.5 | 51.4 | 52.4 | 50.2 | 51.6 | *52.4* |


*DevBench*
| Model | THINGS (RSA) | TROG (Acc.) | Visual Vocab (Acc.) | *Macroaverage* |
| --- | --- | --- | --- | --- |
| Flamingo | 46.5 | 51.3 | 80.7 | *59.5* |
| GIT | 32.6 | 38.2 | 82.4 | *51.1* |

| Model | THINGS (RSA) | TROG (Human sim.) | Visual Vocab (Human sim.) | *Macroaverage* |
| --- | --- | --- | --- | --- |
| Flamingo | 46.5 | 47.7 | 75.2 | *56.4* |
| GIT | 32.6 | 44.7 | 75.3 | *50.8* |

The human similarity scores are computed as `exp(-D)`, where `D` is the KL divergence from human response probability distributions to model logits. We exponentiate the negative value to normalize the divergence into a metric within the range [0,1], and to ensure that higher values are better. Note that the macroaverages reported in **Baselines** are from the first table containing accuracies and the THINGS RSA.

Winoground and VQA do not contain subtasks, so scores for these can be found above in the **Baselines** section.

## Bibliography

--->
