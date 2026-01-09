import json
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Model configuration
MODEL_NAME = "unsloth/Qwen2.5-14B-bnb-4bit"
MAX_SEQ_LENGTH = 4096
DTYPE = None
LOAD_IN_4BIT = True

# LoRA configuration
LORA_R = 16 
LORA_ALPHA = 16
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Training configuration
OUTPUT_DIR = "./qwen14b_finetuned"
NUM_TRAIN_EPOCHS = 3 
PER_DEVICE_TRAIN_BATCH_SIZE = 2 
GRADIENT_ACCUMULATION_STEPS = 4 
LEARNING_RATE = 2e-4 
WARMUP_STEPS = 5 
LOGGING_STEPS = 10 
SAVE_STEPS = 100 
MAX_GRAD_NORM = 0.3 
WEIGHT_DECAY = 0.01 
OPTIM = "adamw_8bit"

DATASET_PATH = "dataset.json"

print("Loading model and tokenizer...")

print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

print("Model loaded successfully!")

print("Configuring LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Enable gradient checkpointing
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

print("LoRA configured successfully!")

print(f"Loading dataset from {DATASET_PATH}...")
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Dataset loaded: {len(data)} samples")

def format_prompt(sample):
    instruction = sample.get('instruction', '')
    input_text = sample.get('input', '')
    output = sample.get('output', '')
    system = sample.get('system', 'Bạn là một trợ lý AI hữu ích.')
    
    # Combine instruction and input if input exists
    if input_text:
        user_message = f"{instruction}\n\n{input_text}"
    else:
        user_message = instruction
    
    # Format using Qwen chat template
    prompt = f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
    
    return {"text": prompt}

# Convert to HuggingFace Dataset and apply formatting
dataset = Dataset.from_list(data)
dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

print(f"Dataset formatted: {len(dataset)} samples")
print("\nSample formatted prompt:")
print(dataset[0]['text'][:500] + "...")

# ============================================================================
# Training Arguments
# ============================================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=LOGGING_STEPS,
    optim=OPTIM,
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type="linear",
    warmup_steps=WARMUP_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    max_grad_norm=MAX_GRAD_NORM,
    report_to="none",  # Disable wandb/tensorboard
    seed=3407,
)

# Initialize Trainer
print("Initializing trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
    packing=False,  # Can set to True for efficiency with short sequences
)

# Train Model
print("\n" + "="*80)
print("Starting training...")
print("="*80 + "\n")

trainer_stats = trainer.train()

print("\n" + "="*80)
print("Training completed!")
print("="*80 + "\n")

# Save Model
print("Saving model...")

# Save LoRA adapters
model.save_pretrained(OUTPUT_DIR + "/lora_adapters")
tokenizer.save_pretrained(OUTPUT_DIR + "/lora_adapters")

# Save merged model in 4-bit
print("Saving merged model (4-bit)...")
model.save_pretrained_merged(
    OUTPUT_DIR + "/merged_4bit",
    tokenizer,
    save_method="merged_4bit"
)

# Save to GGUF format for llama.cpp
print("Saving to GGUF format...")
model.save_pretrained_gguf(
    OUTPUT_DIR + "/gguf",
    tokenizer,
    quantization_method="q4_k_m"
)

print("\n" + "="*80)
print("All models saved successfully!")
print(f"LoRA adapters: {OUTPUT_DIR}/lora_adapters")
print(f"Merged 16-bit: {OUTPUT_DIR}/merged_16bit")
print(f"Merged 4-bit: {OUTPUT_DIR}/merged_4bit")
print(f"GGUF format: {OUTPUT_DIR}/gguf")
print("="*80)

# Test Inference

print("\n" + "="*80)
print("Testing inference...")
print("="*80 + "\n")

# Enable inference mode
FastLanguageModel.for_inference(model)

# Test prompt
test_instruction = "Đại học Cần Thơ được thành lập vào năm nào?"
test_system = "Bạn là công tác viên tư vấn lịch sử của Đại học Cần Thơ nhân dịp kỷ niệm 60 năm thành lập trường."

test_prompt = f"""<|im_start|>system
{test_system}<|im_end|>
<|im_start|>user
{test_instruction}<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

print("Test Question:", test_instruction)
print("\nGenerating answer...")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=False)
# Extract only the assistant's response
assistant_response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]

print("\nModel Response:")
print(assistant_response)

print("\n" + "="*80)
print("Script completed successfully!")
print("="*80)
