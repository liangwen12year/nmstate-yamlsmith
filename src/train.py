#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
This module orchestrates the training of a causal language model using the Hugging Face
Transformers and Datasets libraries. It includes functionalities for loading and
preprocessing YAML-based question and answer data, tokenizing this data for NLP tasks,
and setting up a training environment using predefined training arguments.

Key components:
- Tokenization: Adapt the GPT2Tokenizer to process and tokenize question-answer pairs,
  ensuring appropriate padding and truncation.
- Data Preparation: Load YAML data, preprocess it, and prepare it for training by
  creating a suitable dataset split into training and validation sets.
- Training Setup: Configure and initiate model training using the Trainer class with
  specified training arguments, without direct evaluation on a test set.
- Model Management: Save the trained model and tokenizer for future use or deployment.


Usage:
- The script is designed to be run as a standalone module to facilitate end-to-end model
  training, starting from raw data loading to saving the trained model.
"""

# pylint: disable=import-error
import argparse
from datasets import Dataset, DatasetDict
from transformers import GPT2Tokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from utils import read_and_preprocess_data, load_yaml_dataset


def tokenize_function(examples, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )  # Or use tokenizer.add_special_tokens({'pad_token': '[PAD]'}) if eos_token
        # is not suitable

    # Tokenize both questions and answers together as pairs
    tokenized_inputs = tokenizer(
        examples["question"],
        examples["answer"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
        add_special_tokens=True,
    )

    # Set up labels for training
    tokenized_inputs["labels"] = tokenized_inputs.input_ids.detach().clone()

    return tokenized_inputs


def main():
    parser = argparse.ArgumentParser(description="Train a model with given dataset")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the pretrained model"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset YAML file"
    )
    parser.add_argument(
        "--save_path", type=str, required=False, help="Path to save the finetuned model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory where the training outputs will be saved",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
        help="Directory where training logs will be saved",
    )
    args = parser.parse_args()

    # Load tokenizer and pretrained model
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load the training data
    raw_data = load_yaml_dataset(args.data_path)
    preprocessed_data = read_and_preprocess_data(raw_data)

    dataset = Dataset.from_dict(
        {
            "question": [d["question"] for d in preprocessed_data],
            "answer": [d["answer"] for d in preprocessed_data],
        }
    )

    # Example: Splitting the data into 90% train and 10% validation
    train_dataset, validation_dataset = dataset.train_test_split(test_size=0.1).values()

    # If the original dataset was not a DatasetDict but a Dataset, encapsulate it into
    # a DatasetDict
    tokenized_datasets = DatasetDict(
        {"train": train_dataset, "validation": validation_dataset}
    )

    # Now correctly map the tokenization function to the splits
    tokenized_datasets = tokenized_datasets.map(
        tokenize_function,
        fn_kwargs={
            "tokenizer": tokenizer
        },  # Passing tokenizer as an additional keyword argument
        batched=True,
    )

    # Training arguments
    # FP16 halves the amount of memory required to store numbers compared to FP32.
    # This is particularly beneficial when training large models or working with
    # large datasets that would otherwise require extensive memory. But lower numerical
    # precison might affect model precision and learning dynamics.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=6,
        num_train_epochs=1,  # change to 1 training epoch for integration test
        weight_decay=0.01,
        use_cpu=True,  # Use CPU only for integration test
        logging_dir=args.logging_dir,
        logging_steps=100,  # Adjust according to dataset size
        # Evaluation settings are removed or adjusted if needed
        # fp16=True,
    )

    # Initialize the Trainer without evaluation on test set
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation", None),
        # No evaluation dataset is provided here
    )

    # Train the model
    trainer.train()

    # Optionally save the model
    if args.save_path:
        model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main()
