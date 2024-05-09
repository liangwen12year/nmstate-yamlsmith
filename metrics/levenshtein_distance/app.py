import evaluate
from evaluate.utils import launch_gradio_widget

module = evaluate.load("levenshtein_distance")
launch_gradio_widget(module)
