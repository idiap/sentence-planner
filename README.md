# sentence-planner

This is the code for the paper [Sentence-level Planning for Especially Abstractive Summarization](https://aclanthology.org/2021.newsum-1.1.pdf) presented at the *New Frontiers in Summarization* workshop at EMNLP 2021.

The repository is a fork of the [PreSumm](https://github.com/nlpyang/PreSumm) repository. Changed and added source files are marked in their header comment, others are left untouched. 

## Contents
1. [Data](#data)
2. [Installation](#installation)
3. [Training](#training)
4. [Evaluation](#evaluation)

## Data
For CNN/DM, download the preprocessed data or follow the instructions in the [PreSumm](https://github.com/nlpyang/PreSumm) repository.

For Curation Corpus, follow the instructions over at the [Curation Corpus](https://github.com/CurationCorp/curation-corpus) repository to download the articles. Then follow the instructions in Appendix B of our paper for the preprocessing. If you have trouble reconstructing the dataset, do not hesitate to contact us.

## Installation
First, get a working installation of conda, e.g. [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
Then, recreate the environment from inside this directory by running:

```shell
conda env create -f environment.yml
conda activate presumm
```

## Training
The model can be trained with:

```shell
python src/train.py \
--model sentsumm \
--sentplan_loss_weight 1 \
--mode train \
--bert_data_path <path_to_data_dir>/curation \
--result_path <path_to_result_dir>/curation \
--model_path <path_to_model_dir> \
--log_file train.log
```

A few things to note:
* Data paths consist of the path to the data directory and the prefix of the preprocessed data files (in our case, this is "cnndm" or "curation").
* The same holds for result paths.

## Evaluation
### Validation
Similar to training, the model is validated as follows:

```shell
python src/train.py \
--model sentsumm \
--sentplan_loss_weight 1 \
--mode validate \
--bert_data_path <path_to_data_dir>/curation \
--result_path <path_to_result_dir>/curation \
--model_path <path_to_model_dir> \
--log_file validate.log \
--test_all
```

By looking at the results in the log file, you can select the best checkpoint to keep and discard the rest.

Validation also generates candidate-reference summary pairs, each in their respective files in the results directory. These are used in the evaluations below.

### ROUGE
The official Perl implementation can be called with a helper script like:
```shell
python src/perl_rouge.py \
--candidates_path <path_to_result_dir>/<name_of_candidate_summaries_file> \
--references_path <path_to_result_dir>/<name_of_reference_summaries_file>
```

Installation instructions for the Perl ROUGE script can be found at the [PreSumm](https://github.com/nlpyang/PreSumm) repository.

### Novel bigrams, sentences, words
There is a short evaluation script for each of these, and they all work the same:
```shell
python src/eval/novel_bigrams.py --eval_dir <path_to_result_dir>
```

### Attribution
To compute the attribution to the sentence representation (Section 4.2), use the `src/gxa.py` for the Integrated Gradients algorithm, or `src/conductance.py` for the Conductance algorithm.
Example usage:
```shell
python src/gxa.py \
--model_path <path_to_model_dir> \
--result_path <path_to_result_dir> \
--bert_data_path <path_to_data_dir>/curation \
--share_word_enc \
--use_dec_src_attn \
--use_sent_cond \
--see_past_sents \
--num_examples 100 \
--num_ig_steps 50 \
--baseline zero
```

### Corefs
The Corefs evaluation tests the number of coreference links across sentence boundaries. The evaluation comes with a separate conda environment. Additionally, you have to download a spacy model, in our case `en_core_web_lg`.
Run it with:
```shell
conda env create -f coref.yml
conda activate coref
python -m spacy download en_core_web_lg
python src/eval/coref.py --eval_dir <path_to_result_dir>
```
