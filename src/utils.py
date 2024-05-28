#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
This module provides utility functions for processing data contained in YAML files.
It includes functions to load YAML data from a file and preprocess it for further
analysis or processing.

Usage:
- This module can be imported and its functions invoked in other scripts where YAML
  data handling is required.
"""

# pylint: disable=import-error
import yaml
from transformers import AutoModelForCausalLM, GPT2Tokenizer


def read_and_preprocess_data(data):
    processed_data = []
    for item in data:
        question = item["question"]
        answer = item["answer"].strip()  # Ensuring no extra whitespace
        processed_data.append({"question": question, "answer": answer})

    return processed_data


def load_yaml_dataset(file_path):
    """
    Load a YAML file and return the data.

    Parameters:
    file_path (str): The path to the YAML file to be loaded.

    Returns:
    dict: The data loaded from the YAML file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data


def load_model(model_path, model_type="YAMLsmith"):
    if model_type == "YAMLsmith":
        evaluation_tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        trained_model = AutoModelForCausalLM.from_pretrained(model_path)
        evaluation_tokenizer.pad_token = evaluation_tokenizer.eos_token
        return evaluation_tokenizer, trained_model
    # TO DO: load other models like Ansible Lightspeed
    return None, None
