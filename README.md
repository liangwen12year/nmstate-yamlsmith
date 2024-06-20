# Translating Natural Language into Nmstate States

This repository contains a project that utilizes the `transformers` library to train a
model for generating Nmstate YAML states.

## Project Structure

* `./dataset/`: Contains the dataset for traning and evaluation.
* `./evaluation_results/`: Contains the evaluation result of the most recent
    YAMLsmith model.
* `./metrics/`: Contains the customized evaluation metrics (`nmstate_correct`,
  `yaml_correct` , `levenshtein_distance` ) for evaluating the model.
* `./src/`: Contains the pipeline for pretraing, training, and evaluating the model,
    as well as the UI for hosting the model.

## Workflows

### Pretraining Process Overview

* Model Configuration: Define the model configuration, using the same settings as for
  codegen-350M.
* Load and Tokenize Dataset: Load and tokenize the YAML dataset
  substratusai/the-stack-yaml-k8s.
* Pretraining: Pretrain the model to enhance its understanding of YAML syntax and
  semantics until the training loss converges.
* Save Model: Save the pretrained model.

### Training Process Overview

* Prepare Training Dataset: Prepare the dataset for training, ensuring each training
  sample consists of a natural language description and a YAML state.
* Load, Preprocess, and Tokenize Dataset: Load the dataset, preprocess the data, and
  tokenize it for training.
* Load Pretrained Model: Initialize the model using pretrained weights.
* Train the Model: Train the model using the prepared dataset until the training loss
  converges.
* Save Model and Tokenizer: Save the trained model and tokenizer for future use.

### Evaluation Process Overview

* Prepare Evaluation Dataset: Set up the dataset for evaluation, ensuring each
  sample includes a natural language description and the corresponding expected
  YAML state.
* Load, Preprocess, and Tokenize Dataset: Load the dataset and preprocess the
  data for evalution.
* Load Pretrained Model: Initialize the model with pretrained weights.
* Define Evaluation Metrics: Specify the metrics (nmstate_correct, exact_match,
  yaml_correct, levenshtein_distance) to be used for evaluation.
* Evaluate the Model: Use the model to perform inference on the natural
  language descriptions. Compare the generated YAML with the expected YAML
  based on the defined evaluation metrics to calculate the metric scores. Use
  cached data if available, or generate new predictions for evaluation.

## Installation

Before running the script, ensure you have Python 3.8+ installed. You can then install
the necessary dependencies by running:

```bash
pip install -r requirements.txt
```
