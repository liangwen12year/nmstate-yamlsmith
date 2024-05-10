#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
This module is intended to train the Polyglot Model for translating natural language
into nmstate states.

Dependencies:
- transformers: Provides the models and tokenizers used for sequence generation.
- datasets: Used for loading and preprocessing the dataset.
- torch: Utilized for model operations, especially for managing device placement
  (CPU/GPU).

Note:
- Ensure that the CUDA-capable device is available for GPU acceleration when training
  and generating files, as the operations are compute-intensive.
"""

# pylint: disable=import-error, broad-exception-caught
from transformers import (
    CodeGenForCausalLM,
    CodeGenConfig,
    Trainer,
    TrainingArguments,
    GPT2Tokenizer,
)
from datasets import load_dataset
import torch


def tokenize_function(examples, tokenizer):
    # Adjusted to tokenize the 'content' field
    tokenized_inputs = tokenizer(
        examples["content"], truncation=True, padding="max_length", max_length=2048
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs


def generate_yaml_files(model, tokenizer, device, num_files=3):
    yaml_files = []
    prompt = (
        "apiVersion: v1\nkind: "  # Adjust prompt to fit the expected YAML structure
    )

    for _ in range(num_files):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        # Generate sequences
        # top_k selects the top k number of next words at each step of the generation
        # top_p selects the smallest set of words whose cumulative probability exceeds
        # the threshold p
        outputs = model.generate(
            input_ids,
            max_length=2000,  # adjust the max length according to needs
            num_return_sequences=1,
            temperature=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,  # Set pad token ID to eos_token_id
        )

        # Convert the tensor to a list
        generated_sequence = outputs[0].tolist()

        # Ensure all tokens are valid
        filtered_sequence = [token for token in generated_sequence if token is not None]

        # Decode the filtered sequence
        try:
            generated_text = tokenizer.decode(
                filtered_sequence, skip_special_tokens=True
            )
            yaml_files.append(generated_text)
        except TypeError as e:
            print(f"TypeError decoding sequence with output {outputs[0]}: {e}")
            continue
        except Exception as e:  # Catch any other unforeseen exceptions
            print(f"Unexpected error decoding sequence with output {outputs[0]}: {e}")
            raise

    return yaml_files


def save_yaml_files(yaml_files):
    for idx, content in enumerate(yaml_files, 1):
        with open(f"generated_yaml_{idx}.yaml", "w", encoding="utf-8") as file:
            file.write(content)


def main():
    # the actual dataset used for pretraining is substratusai/the-stack-yaml-k8s, use
    # the following smaller dataset for integration test
    dataset = load_dataset("rs0x29a/the-stack-yaml-camel-k-k8s")
    # Tokenize the dataset
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    # use 0.01 size of the dataset to train for integration test
    data_split = tokenized_datasets["train"].train_test_split(test_size=0.99)
    small_train_dataset = data_split["train"]

    # Define the model configuration, I get the configuration by printing model config
    # for codegen-350M
    config = CodeGenConfig(
        _name_or_path="Salesforce/codegen-350M-multi",
        activation_function="gelu_new",
        architectures=["CodeGenForCausalLM"],
        attn_pdrop=0.0,
        bos_token_id=1,
        embd_pdrop=0.0,
        eos_token_id=50256,
        gradient_checkpointing=False,
        initializer_range=0.02,
        layer_norm_epsilon=1e-05,
        model_type="codegen",
        n_ctx=2048,
        n_embd=1024,
        n_head=16,
        n_layer=20,
        n_positions=2048,
        resid_pdrop=0.0,
        rotary_dim=32,
        scale_attn_weights=True,
        summary_activation=None,
        summary_first_dropout=0.1,
        summary_proj_to_labels=True,
        summary_type="cls_index",
        summary_use_proj=True,
        tie_word_embeddings=False,
        tokenizer_class="GPT2Tokenizer",
        torch_dtype="float16",
        transformers_version="4.33.2",
        use_cache=True,
        vocab_size=51200,
    )

    # Initialize the model from the configuration
    model = CodeGenForCausalLM(config=config)

    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir="./model_output",
        per_device_train_batch_size=6,  # adjust according to GPU capacity
        logging_steps=500,
        num_train_epochs=1,
        save_steps=5000,
        save_total_limit=3,  # limits the total amount of checkpoints
        prediction_loss_only=True,
        gradient_accumulation_steps=2,
        evaluation_strategy="no",
        use_cpu=True,  # Use CPU only for integration test
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=small_train_dataset
    )

    # Start training
    trainer.train()
    trainer.save_model("./pretrained_yaml_model")

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # It is possible that generated YAML examples are not complete. To make
    # the generated YAML examples complete, we can provide a more detailed
    # prompt to guide the model or adjust the `max_length` parameter to
    # allow the model to generate longer sequences or adjust the generation
    # parameters like `temperature`, `top_k`, `top_p` to control the randomness
    # and creativity of the output.
    generated_yaml_files = generate_yaml_files(model, tokenizer, device)
    for i, yaml_file in enumerate(generated_yaml_files, 1):
        print(f"Generated YAML #{i}:\n{yaml_file}\n")

    save_yaml_files(generated_yaml_files)


if __name__ == "__main__":
    main()
