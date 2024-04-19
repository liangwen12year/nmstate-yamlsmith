# SPDX-License-Identifier: LGPL-2.1-or-later

from transformers import CodeGenForCausalLM, CodeGenConfig, Trainer, TrainingArguments
from transformers import GPT2Tokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torch

def tokenize_function(examples, tokenizer):
    # Adjusted to tokenize the 'content' field
    tokenized_inputs= tokenizer(examples['content'], truncation=True, padding="max_length", max_length=2048)
    tokenized_inputs['labels'] = tokenized_inputs['input_ids'].copy()
    return tokenized_inputs

def generate_yaml_files(model, tokenizer, device, num_files=10):
    yaml_files = []
    prompt = "apiVersion: v1\nkind: "  # Adjust prompt to fit the expected YAML structure

    for _ in range(num_files):
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        # Generate sequences
        outputs = model.generate(
            input_ids,
            max_length=2000,  # adjust the max length according to needs
            num_return_sequences=3,
            temperature=1,  
            no_repeat_ngram_size=2,
            top_k=50, # select the top k number of next words at each step of the generation
            top_p=0.9, # select the smallest set of words whose cumulative probability exceeds the threshold p
            do_sample=True
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        yaml_files.append(generated_text)
    
    return yaml_files

def save_yaml_files(yaml_files):
    for idx, content in enumerate(yaml_files, 1):
        with open(f"generated_yaml_{idx}.yaml", 'w') as file:
            file.write(content)

def main():
    dataset = load_dataset("substratusai/the-stack-yaml-k8s")
    # Tokenize the dataset
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    train_test_split = tokenized_datasets['train'].train_test_split(test_size=0.95)
    small_train_dataset = train_test_split['train']

    # Define the model configuration, I get the configuration by printing model config for codegen-350M
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
        vocab_size=51200
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
        gradient_accumulation_steps = 2,
        evaluation_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset
    )

    # Start training
    trainer.train()
    trainer.save_model("./pretrained_yaml_model")

    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    generated_yaml_files = generate_yaml_files(model, tokenizer, device)
    for i, yaml_file in enumerate(generated_yaml_files, 1):
      print(f"Generated YAML #{i}:\n{yaml_file}\n")


    save_yaml_files(generated_yaml_files)



if __name__ == "__main__":
    main()
