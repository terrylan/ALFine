# ALBERT-FineTune-Arch

A project for fine-tuning ALBERT (`albert-base-v2`) on the IMDB sentiment classification dataset using Hugging Face's `transformers` and `datasets`.

## Installation

Ensure you have Python 3.8+ installed. Then, install dependencies:

```sh
pip install torch transformers datasets
```

## Usage

Run the fine-tuning script:

```sh
python finetune_albert.py
```

The fine-tuned model and tokenizer will be saved in `./fine_tuned_albert`.
