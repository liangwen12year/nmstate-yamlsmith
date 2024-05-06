# YAMLsmith Evaluation Results

## Introduction

This README provides an overview of the evaluation results for the YAMLsmith model.
The evaluation is based on several key metrics which are crucial for assessing the
model's performance in translating the natural language into Nmstate states.

## Metrics Explained

Below is a description of the metrics used to evaluate the model:

- **Nmstate Correct**: Assesses whether the generated output correctly follows the
  Nmstate schema without any structural or syntactic errors.
- **EM (Exact Match)**: Measures the percentage of predictions that exactly match any
  one of the ground truth answers.
- **YAML Correct**: Assesses whether the generated output correctly follows a
  predefined YAML schema without any structural or syntactic errors.
- **Levenshtein Distance**: Quantifies the minimum number of single-character edits
  (insertions, deletions, or substitutions) required to change the prediction into the
  ground truth answer.

## Evaluation Results

Here are the results from our latest model evaluation:

| Metric                   | Score  |
|--------------------------|--------|
| Nmstate Correct Predictions | 91.18   |
| Nmstate Correct References | 94.12   |
| Exact Match (EM)          | 85.29   |
| YAML Correct              | 94.12  |
| Levenshtein Distance      | 14.12    |

## Interpretation of Results

- **Nmstate Correct Predictions**: A score of 91.18 indicates that the model's
  predictions matched the correct Nmstate schema 91.18% of the time.
- **Nmstate Correct References**: A score of 94.12 indicates that the model's
  references matched the correct Nmstate schema 94.12% of the time.
- **Exact Match (EM)**: A score of 85.29 indicates that the model's predictions exactly
  matched the ground truth answers 85.29% of the time.
- **YAML Correct**: A score of 94.12 signifies that all generated outputs correctly
  adhere to the required YAML schema.
- **Levenshtein Distance**: The average minimal edit distance of 14.12 indicates that,
  on average, a relatively higher number of edits are required to align the model's
  predictions with the ground truth.

## Conclusion

The high scores in Nmstate Correct Predictions, Nmstate Correct References, and YAML
Correctness demonstrate the model's effectiveness in generating accurate and
structurally correct outputs. The Exact Match score and Levenshtein Distance indicate
there is room for improvements in the model's precision. The result is as expected
since we are using the same dataset for both training and testing. The data is split
into a 9:1 ratio for training and validation. We plan to make changes to this testing
setup once we gather more data.
