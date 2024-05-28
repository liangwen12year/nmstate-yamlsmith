#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
This module provides functionalities for evaluating the YAMLsmith model designed to
generate YAML state configurations based on naturural language input. It includes
functions to load models and testing data, preprocess data, compute evaluation
metrics (Exact Match, YAML Correctness, and Levenshtein Distance), and record the
evaluation results into a CSV file.

The evaluation process involves loading a trained model and a dataset, generating
answers for the provided questions, comparing these answers against expected outputs,
computing specified metrics, and finally writing these metrics to a CSV file for
further analysis.

Usage:
- This module is intended to be run as a script. The main function orchestrates
  loading of data and model, evaluation of the model, and writing out results.
"""

# pylint: disable=import-error, line-too-long, too-many-locals
import argparse
import csv
import json
import os
import shutil
from evaluate import load  # pylint: disable=import-self
from utils import read_and_preprocess_data, load_yaml_dataset, load_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_CORRECT_METRIC_PATH = os.path.join(BASE_DIR, "../metrics/yaml_correct")
LEVENSHTEIN_DISTANCE_METRIC_PATH = os.path.join(
    BASE_DIR, "../metrics/levenshtein_distance"
)
NMSTATE_CORRECT_METRIC_PATH = os.path.join(BASE_DIR, "../metrics/nmstate_correct")


def evaluate_model(
    evaluation_tokenizer, trained_model, processed_data, metric_types, cache_file
):
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        expected_answer_list = cache_data["expected"]
        generated_answer_list = cache_data["generated"]
    else:
        expected_answer_list = []
        generated_answer_list = []
        for data in processed_data:
            question = data["question"]
            expected_answer = data["answer"]
            expected_answer_list.append(expected_answer)
            answer = generate_answer(question, evaluation_tokenizer, trained_model)
            generated_answer_list.append(answer)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(
                {"expected": expected_answer_list, "generated": generated_answer_list},
                f,
            )

    results = []
    for metric_type in metric_types:
        if metric_type == "exact_match":
            exact_match_metric = load("exact_match")
            results.append(
                exact_match_metric.compute(
                    predictions=generated_answer_list, references=expected_answer_list
                )
            )
        elif metric_type == "yaml_correct":
            yaml_correct_metric = load(YAML_CORRECT_METRIC_PATH)
            results.append(
                yaml_correct_metric.compute(predictions=generated_answer_list)
            )
        elif metric_type == "levenshtein_distance":
            levenshtein_distance_metric = load(LEVENSHTEIN_DISTANCE_METRIC_PATH)
            results.append(
                levenshtein_distance_metric.compute(
                    predictions=generated_answer_list, references=expected_answer_list
                )
            )
        elif metric_type == "nmstate_correct":
            try:
                check_nmstatectl()
                print("nmstatectl is available.")
            except FileNotFoundError as e:
                print(f"nmstatectl is not available: {e}")
                raise
            nmstate_correct_metric = load(NMSTATE_CORRECT_METRIC_PATH)
            nmstate_correct_score_prediction, generated_answer_list = (
                nmstate_correct_metric.compute(predictions=generated_answer_list)
            )
            nmstate_correct_score_prediction["nmstate_correct_predictions"] = (
                nmstate_correct_score_prediction.pop("nmstate_correct")
            )
            results.append(nmstate_correct_score_prediction)
            nmstate_correct_score_reference, expected_answer_list = (
                nmstate_correct_metric.compute(predictions=expected_answer_list)
            )
            nmstate_correct_score_reference["nmstate_correct_references"] = (
                nmstate_correct_score_reference.pop("nmstate_correct")
            )
            results.append(nmstate_correct_score_reference)

    return results


def check_nmstatectl():
    # Check if 'nmstatectl' is in the system path
    path = shutil.which("nmstatectl")
    if not path:
        raise FileNotFoundError("'nmstatectl' not found in the system path.")


def generate_answer(question, evaluation_tokenizer, trained_model):
    # Tokenize the question
    inputs = evaluation_tokenizer.encode(question, return_tensors="pt")

    # Generate the answer
    outputs = trained_model.generate(
        inputs, max_length=500, pad_token_id=evaluation_tokenizer.eos_token_id
    )
    answer = evaluation_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Removing the repeated question if it appears in the answer
    cleaned_answer = answer.replace(question, "").strip()

    # Find the position of '---'
    index = cleaned_answer.find("---")

    # If '---' exists and there are characters before it, remove them
    if index not in [0, -1]:
        cleaned_answer = cleaned_answer[index:].strip()

    return cleaned_answer


def write_results_to_csv(results, filename):
    # Open the file in write mode
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Metric", "Score"])
        # Iterate over the dictionary and write rows
        for result in results:
            for key, value in result.items():
                if key != "levenshtein_distance":
                    value = round(value * 100, 2)  # convert to percentage value
                writer.writerow([key, value])

        # Add a blank row to separate the results from the instructions
        writer.writerow([])

        # Write the instructions as a paragraph
        instructions_paragraph = (
            "\nBelow is a description of the metrics used to evaluate the model:\n"
            "   - Nmstate Correct: Assesses whether the generated output correctly follows the Nmstate schema without any structural or syntactic errors.\n"
            "   - EM (Exact Match): Measures the percentage of predictions that exactly match any one of the ground truth answers.\n"
            "   - YAML Correct: Assesses whether the generated output correctly follows a predefined YAML schema without any structural or syntactic errors.\n"
            "   - Levenshtein Distance: Quantifies the minimum number of single-character edits (insertions, deletions, or substitutions) required to change the prediction into the ground truth answer.\n\n\n"
            "To reproduce the evaluation results, please follow these steps:\n"
            "1. Pretraining Process Overview:\n"
            "   - Model Configuration: Define the model configuration, using the same settings as for codegen-350M.\n"
            "   - Load and Tokenize Dataset: Load and tokenize the YAML dataset substratusai/the-stack-yaml-k8s.\n"
            "   - Pretraining: Pretrain the model to enhance its understanding of YAML syntax and semantics until the training loss converges.\n"
            "   - Save Model: Save the pretrained model.\n\n"
            "2. Training Process Overview:\n"
            "   - Prepare Training Dataset: Prepare the dataset for training, ensuring each training sample consists of a natural language description and a YAML state.\n"
            "   - Load, Preprocess, and Tokenize Dataset: Load the dataset, preprocess the data, and tokenize it for training.\n"
            "   - Load Pretrained Model: Initialize the model using pretrained weights.\n"
            "   - Train the Model: Train the model using the prepared dataset until the training loss converges.\n"
            "   - Save Model and Tokenizer: Save the trained model and tokenizer for future use.\n\n"
            "3. Evaluation Process Overview:\n"
            "   - Prepare Evaluation Dataset: Set up the dataset for evaluation, ensuring each sample includes a natural language description and the corresponding expected YAML state.\n"
            "   - Load, Preprocess, and Tokenize Dataset: Load the dataset and preprocess the data for evalution.\n"
            "   - Load Pretrained Model: Initialize the model with pretrained weights.\n"
            "   - Define Evaluation Metrics: Specify the metrics (nmstate_correct, exact_match, yaml_correct, levenshtein_distance) to be used for evaluation.\n"
            "   - Evaluate the Model: Use the model to perform inference on the natural language descriptions. Compare the generated YAML with the expected YAML based on the defined evaluation metrics to calculate the metric scores. Use cached data if available, or generate new predictions for evaluation."
        )

        # Write the instructions paragraph
        writer.writerow([instructions_paragraph])
    # Reading back and printing the file contents
    print("\n*****Reading the content of the evaluation result file*****:\n")
    with open(filename, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            print(",".join(row))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model with given paths.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model directory"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset YAML file"
    )
    parser.add_argument(
        "--cache_file",
        type=str,
        default="generated_answers_cache.json",
        help="Path to the cache file for generated answers",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="Path to save the evaluation results CSV file",
    )

    args = parser.parse_args()
    # Load tokenizer and model
    tokenizer, model = load_model(args.model_path)

    # Load the testing data
    raw_data = load_yaml_dataset(args.data_path)
    testing_data = read_and_preprocess_data(raw_data)

    # Define metrics to use
    evaluation_metrics = [
        "nmstate_correct",
        "exact_match",
        "yaml_correct",
        "levenshtein_distance",
    ]  # Example of multiple metrics

    # Evaluate the model
    results = evaluate_model(
        tokenizer, model, testing_data, evaluation_metrics, args.cache_file
    )

    # Write results to CSV
    write_results_to_csv(results, args.result_path)


if __name__ == "__main__":
    main()
