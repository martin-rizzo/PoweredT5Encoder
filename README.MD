<h1 align="center">Powered T5 Encoder</h1>
<p align="center">An enhanced T5 encoder with integrated weighted prompt support.</p>
<p align="center">
    <img alt="Platform" src="https://img.shields.io/badge/platform-pytorch-33F">
    <img alt="Language" src="https://img.shields.io/badge/language-python-3776AB">
    <img alt="License"  src="https://img.shields.io/github/license/martin-rizzo/PoweredT5Encoder?color=11D">
    <img alt="Last"     src="https://img.shields.io/github/last-commit/martin-rizzo/PoweredT5Encoder">
</p>

<!--
# Powered T5 Encoder
-->

**Powered T5 Encoder** is a Python wrapper for T5Tokenizer and T5EncoderModel from the transformers library. This wrapper adds support for tokenizing and encoding prompts with weighted text areas, allowing you to assign different levels of importance to specific words. Additionally, it provides extra functionality for loading standalone .safetensors files without the need for model directories and configurations.

## Features

- **Weighted Text Prompts**: Tokenize and encode prompts with user-defined weights for specific words, e.g., "An (important:1.3) word".
- **Standalone .safetensors Loading**: Load .safetensors files directly, simplifying model handling.
- **Testing and Evaluation**: Tools for testing and evaluating how user-defined weights are parsed within prompts.

## Installation

The provided bash scripts automate the creation of a Python virtual environment and installation of necessary dependencies seamlessly.

For manual installation without a virtual environment, you can use the classic:

    pip install -r requirements.txt

## Testing and Evaluation

### Using t5weights.py

`t5weights.py` provides a way to test and evaluate how the weights in prompts are parsed.

    python t5weights.py "This is an (important:1.3) example."

### Using t5weights.sh

`t5weights.sh` is a bash script to run the Python code in a virtual environment.

    ./t5weights.sh "This is an (important:1.3) example."


## Lincense

Copyright (c) 2024 Martin Rizzo  
This project is licensed under the MIT license.  
See the ["LICENSE"](LICENSE) file for details.

## Acknowledgements

This project utilizes the [transformers](https://github.com/huggingface/transformers) library by Hugging Face.

