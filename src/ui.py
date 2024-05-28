"""
This script provides a Gradio-based web interface for interacting with a language model
and allowing users to rate the generated responses. It includes functionalities to load
a model, generate responses based on user input, and save user ratings for the
responses. Additionally, it maintains the history of the conversation on the page.
"""

# pylint: disable=import-error, line-too-long, too-many-locals, no-member
import argparse
import csv
import os
import re
import gradio as gr
from transformers import pipeline
import torch
from utils import load_model


# Generates a response from the language model based on the user input.
def chat_with_model(user_input, model, tokenizer, device):
    # Create a pipeline for text generation
    chat_pipeline = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=device
    )
    response = chat_pipeline(
        user_input,
        max_length=500,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Removing the repeated question if it appears in the answer
    cleaned_answer = response[0]["generated_text"].replace(user_input, "").strip()

    # Find the position of '---'
    index = cleaned_answer.find("---")

    # If '---' exists and there are characters before it, remove them
    if index not in [0, -1]:
        cleaned_answer = cleaned_answer[index:].strip()

    return cleaned_answer


# Saves the user input, model response, and rating to a CSV file.
def save_rating(user_input, model_response, rating, rating_result_file):
    file_exists = os.path.isfile(rating_result_file)
    with open(rating_result_file, mode="a", newline="\n", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["user_input", "model_response", "rating"])
        writer.writerow([user_input, model_response, rating])
        file.write("\n")  # Adding an extra newline after each entry
    return "Thank you for your feedback!"


def preprocess_input(user_input):
    # Replace multiple spaces with a single space and remove newlines
    user_input = re.sub(r"\s+", " ", user_input).strip()
    return user_input


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model with ratings")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model directory"
    )
    parser.add_argument(
        "--rating_result_path",
        type=str,
        required=True,
        help="Path to save the rating file",
    )
    args = parser.parse_args()
    # Load tokenizer and model
    tokenizer, model = load_model(args.model_path)

    # Check if a GPU is available and set the device accordingly
    device = 0 if torch.cuda.is_available() else -1
    # Create the Gradio interface
    with gr.Blocks() as interface:
        gr.Markdown("# YAMLsmith with Ratings")
        user_input = gr.Textbox(
            label="User Input", placeholder="Type your question here..."
        )
        response_output = gr.Textbox(label="Model Response", interactive=False)
        history = gr.Textbox(label="Conversation History", interactive=False)
        generate_button = gr.Button("Generate Response")
        rating = gr.Slider(1, 5, step=1, label="Rate the response")
        feedback_output = gr.Textbox(label="Feedback", interactive=False)
        submit_button = gr.Button("Submit Rating")

        conversation_history = []

        def generate_response(user_input):
            user_input = preprocess_input(user_input)
            response = chat_with_model(user_input, model, tokenizer, device)
            conversation_history.append(f"User: {user_input}\nModel: {response}\n")
            history_text = "\n\n".join(conversation_history)
            return response, history_text

        def submit_rating(user_input, response_output, rating):
            feedback = save_rating(
                user_input, response_output, rating, args.rating_result_path
            )
            return feedback

        # Bind Enter key to generate response
        user_input.submit(
            fn=generate_response, inputs=user_input, outputs=[response_output, history]
        )
        generate_button.click(
            fn=generate_response, inputs=user_input, outputs=[response_output, history]
        )
        submit_button.click(
            fn=submit_rating,
            inputs=[user_input, response_output, rating],
            outputs=feedback_output,
        )

    # Launch the interface with the share parameter set to True
    interface.launch(share=True)


if __name__ == "__main__":
    main()
