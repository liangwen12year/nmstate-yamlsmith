import os
import subprocess
import tempfile
import evaluate
import datasets


class NmstateCorrect(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="Custom metric to check if the provided answer conforms to Nmstate schema.",
            citation="",
            inputs_description="Takes a string input and checks if it is a valid Nmstate YAML state.",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                }
            ),
        )

    def _compute(self, predictions):
        results = []
        normalized_predictions = []
        for answer in predictions:
            try:
                # Write the YAML content to a temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False, mode="w", suffix=".yml"
                ) as temp_file:
                    temp_file.write(answer)
                    temp_file_path = temp_file.name

                # Run the nmstatectl format command
                command = ["nmstatectl", "format", temp_file_path]
                process = subprocess.run(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
                )

                # Check if the command was successful
                if process.returncode == 0:
                    # Capture the command output and remove the line contains 'Nmstate version'
                    normalized_prediction = process.stdout.decode("utf-8")
                    normalized_prediction = "\n".join(
                        line
                        for line in normalized_prediction.splitlines()
                        if "Nmstate version" not in line
                    )
                    normalized_predictions.append(normalized_prediction)
                    results.append(1)  # Valid Nmstate YAML state
                else:
                    normalized_predictions.append(answer)
                    results.append(0)  # Invalid Nmstate YAML state
            except subprocess.CalledProcessError:
                # If the command failed, append 0 (invalid Nmstate YAML state)
                normalized_predictions.append(answer)
                results.append(0)
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        average_score = sum(results) / len(results)
        return {"nmstate_correct": average_score}, normalized_predictions
