import evaluate
import datasets
import Levenshtein


class LevenshteinDistance(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="Custom metric to calculate the Levenshtein distance between two text representations.",
            citation="",
            inputs_description="Takes two string inputs and calculates the Levenshtein distance between them.",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            reference_urls=[],
        )

    def _compute(self, predictions, references):
        """
        Calculate the Levenshtein distance between two YAML configurations.

        Parameters:
            predictions (list of str): List of model generated YAML strings.
            references  (list of str): List of ground truth YAML strings.

        Returns:
            float: The average Levenshtein distance between the list of predictions
            and the list of references.
        """
        distances = []
        for pred, ref in zip(predictions, references):
            if not isinstance(pred, str) or not isinstance(ref, str):
                raise ValueError("Both prediction and reference must be of type str.")
            distances.append(Levenshtein.distance(pred, ref))

        average_distance = sum(distances) / len(distances)
        return {"levenshtein_distance": average_distance}
