import os
import json
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

# Model Configuration
MODEL_NAME = "Qwen/Qwen2.5-14B"  # Change this to any Qwen 14B model
MAX_SEQ_LENGTH = 2048  # Maximum sequence length
LOAD_IN_4BIT = True  # Use 4-bit quantization

# LoRA Configuration
LORA_R = 16  # LoRA rank
LORA_ALPHA = 16  # LoRA alpha
LORA_DROPOUT = 0  # LoRA dropout
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]  # Target modules for LoRA

# Training Configuration
OUTPUT_DIR = "./qwen14b_finetuned"
DATASET_PATH = "dataset.json"
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_STEPS = 5
LOGGING_STEPS = 10
SAVE_STEPS = 100

# GGUF Export Configuration
GGUF_QUANTIZATION_METHODS = ["q4_k_m", "q5_k_m", "q8_0"]  # Multiple quantization methods

def format_prompt(sample):
    """
    Format the dataset sample into Qwen chat template format.
    Supports Alpaca format with instruction, input, output, and optional system.
    """
    system_message = sample.get("system", "You are a helpful assistant.")
    instruction = sample["instruction"]
    input_text = sample.get("input", "")
    output_text = sample["output"]
    
    # Construct the prompt with Qwen chat template
    if input_text:
        user_message = f"{instruction}\n\n{input_text}"
    else:
        user_message = instruction
    
    # Qwen chat template format
    prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{output_text}<|im_end|>"""
    
    return {"text": prompt}

def load_and_prepare_dataset(dataset_path):
    """Load and prepare the dataset for training."""
    print(f"Loading dataset from {dataset_path}...")
    
    # Load JSON dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Sample keys: {dataset[0].keys()}")
    
    # Format dataset
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
    
    # Split into train and validation (90/10 split)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['test'])}")
    print(f"\nSample formatted prompt:\n{dataset['train'][0]['text'][:500]}...\n")
    
    return dataset

def load_model_and_tokenizer():
    """Load the model and tokenizer with Unsloth optimizations."""
    print(f"Loading model: {MODEL_NAME}")
    print(f"4-bit quantization: {LOAD_IN_4BIT}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect dtype
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    # Configure LoRA
    print("\nConfiguring LoRA...")
    print(f"LoRA rank: {LORA_R}")
    print(f"LoRA alpha: {LORA_ALPHA}")
    print(f"Target modules: {TARGET_MODULES}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized gradient checkpointing
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Print model info
    print("\nModel loaded successfully!")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer

def train_model(model, tokenizer, dataset):
    """Train the model using SFTTrainer."""
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=LOGGING_STEPS,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,  # Can set to True for efficiency
        args=training_args,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    print("\nTraining completed!")
    return trainer

def save_model(model, tokenizer):
    """Save the fine-tuned model in multiple formats."""
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80 + "\n")
    
    # Save LoRA adapters
    lora_output_dir = f"{OUTPUT_DIR}/lora_adapters"
    print(f"Saving LoRA adapters to {lora_output_dir}...")
    model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)
    print("LoRA adapters saved!")
    
    # Save merged model (16-bit)
    merged_16bit_dir = f"{OUTPUT_DIR}/merged_16bit"
    print(f"\nSaving merged 16-bit model to {merged_16bit_dir}...")
    model.save_pretrained_merged(merged_16bit_dir, tokenizer, save_method="merged_16bit")
    print("16-bit merged model saved!")
    
    # Save merged model (4-bit for inference)
    merged_4bit_dir = f"{OUTPUT_DIR}/merged_4bit"
    print(f"\nSaving merged 4-bit model to {merged_4bit_dir}...")
    model.save_pretrained_merged(merged_4bit_dir, tokenizer, save_method="merged_4bit")
    print("4-bit merged model saved!")
    
    return merged_16bit_dir

def export_to_gguf(model, tokenizer, merged_model_dir):
    """Export the model to GGUF format for Ollama."""
    print("\n" + "="*80)
    print("EXPORTING TO GGUF FORMAT")
    print("="*80 + "\n")
    
    gguf_output_dir = f"{OUTPUT_DIR}/gguf"
    os.makedirs(gguf_output_dir, exist_ok=True)
    
    for quant_method in GGUF_QUANTIZATION_METHODS:
        print(f"\nExporting with quantization method: {quant_method}")
        output_file = f"{gguf_output_dir}/qwen14b_finetuned_{quant_method}.gguf"
        
        try:
            model.save_pretrained_gguf(
                gguf_output_dir,
                tokenizer,
                quantization_method=quant_method
            )
            print(f"✓ Successfully exported: {output_file}")
        except Exception as e:
            print(f"✗ Error exporting {quant_method}: {e}")
    
    print(f"\nGGUF models saved to: {gguf_output_dir}")
    print("\nTo use with Ollama:")
    print("1. Create a Modelfile:")
    print(f"   FROM {gguf_output_dir}/qwen14b_finetuned_q4_k_m.gguf")
    print("2. Create the model:")
    print("   ollama create qwen14b-finetuned -f Modelfile")
    print("3. Run the model:")
    print("   ollama run qwen14b-finetuned")

def test_inference(model, tokenizer):
    """Test the fine-tuned model with sample inference."""
    print("\n" + "="*80)
    print("TESTING INFERENCE")
    print("="*80 + "\n")
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    # Test prompts
    test_prompts = [
        {
            "system": "Bạn là công tác viên tư vấn lịch sử của Đại học Cần Thơ nhân dịp kỷ niệm 60 năm thành lập trường.",
            "instruction": "Can Tho University được thành lập vào năm nào?",
            "input": ""
        },
        {
            "system": "You are a helpful assistant.",
            "instruction": "What is the capital of France?",
            "input": ""
        }
    ]
    
    for i, prompt_data in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        
        # Format prompt
        system_msg = prompt_data.get("system", "You are a helpful assistant.")
        instruction = prompt_data["instruction"]
        input_text = prompt_data.get("input", "")
        
        if input_text:
            user_message = f"{instruction}\n\n{input_text}"
        else:
            user_message = instruction
        
        prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
        
        print(f"Prompt: {instruction}")
        
        # Tokenize and generate
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            use_cache=True
        )
        
        # Decode and print
        response = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
        # Extract only the assistant's response
        assistant_response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
        
        print(f"Response: {assistant_response.strip()}")
        print()

def main():
    """Main execution function."""
    print("="*80)
    print("QWEN 14B FINE-TUNING WITH UNSLOTH AND QLORA")
    print("="*80)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    # Step 1: Load and prepare dataset
    dataset = load_and_prepare_dataset(DATASET_PATH)
    
    # Step 2: Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Step 3: Train the model
    trainer = train_model(model, tokenizer, dataset)
    
    # Step 4: Save the model
    merged_model_dir = save_model(model, tokenizer)
    
    # Step 5: Export to GGUF
    export_to_gguf(model, tokenizer, merged_model_dir)
    
    # Step 6: Test inference
    test_inference(model, tokenizer)
    
    print("\n" + "="*80)
    print("FINE-TUNING COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nDirectory structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  ├── lora_adapters/       # LoRA adapter weights")
    print(f"  ├── merged_16bit/        # Full merged model (16-bit)")
    print(f"  ├── merged_4bit/         # Full merged model (4-bit)")
    print(f"  └── gguf/                # GGUF models for Ollama")
    print()

if __name__ == "__main__":
    main()
