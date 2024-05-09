import yaml
import evaluate
import datasets

class YamlCorrect(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="Custom metric to check if the provided answer is a correctly structured YAML.",
            citation="",
            inputs_description="Takes a string input and checks if it is valid YAML.",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                }
            ),
        )

    def _compute(self, predictions):
        """
        Validate if each prediction is correctly structured YAML.

        Parameters:
            predictions (list of str): List of YAML strings to validate.

        Returns:
            dict: Dictionary containing the metric result.
        """
        results = []
        for answer in predictions:
            try:
                # Attempt to parse the YAML
                loaded_yaml = yaml.safe_load(answer)
                if loaded_yaml is not None:
                    results.append(1)  # Valid YAML
                else:
                    results.append(0)  # Invalid YAML
            except yaml.YAMLError:
                results.append(0)  # Invalid YAML

        average_score = sum(results) / len(results)
        return {"yaml_correct": average_score}
