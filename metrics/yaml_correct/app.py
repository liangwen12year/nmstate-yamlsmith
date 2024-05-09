import evaluate
from evaluate.utils import launch_gradio_widget

module = evaluate.load("yaml_correct")
launch_gradio_widget(module)
