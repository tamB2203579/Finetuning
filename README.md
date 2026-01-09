# Qwen 14B Fine-tuning with Unsloth

This project fine-tunes Qwen 14B model using Unsloth on a Vietnamese Q&A dataset about Can Tho University.

## Features

- **4-bit Quantization**: Uses 4-bit quantization for memory efficiency
- **LoRA Adapters**: Efficient fine-tuning with Low-Rank Adaptation
- **Multiple Save Formats**: Saves models in LoRA, merged 16-bit, merged 4-bit, and GGUF formats
- **Optimized for GPU**: Configured for efficient training on consumer GPUs
- **Vietnamese Language Support**: Properly handles Vietnamese text encoding

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- PyTorch with CUDA support

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Verify your dataset is in the correct location:
```
dataset.json
```

## Dataset Format

The dataset should be a JSON array with objects containing:
- `instruction`: The question or instruction
- `input`: Additional context (can be empty)
- `output`: The expected response
- `system`: System prompt (optional)

Example:
```json
[
  {
    "instruction": "Đại học Cần Thơ được thành lập vào năm nào?",
    "input": "",
    "output": "Trường Đại học Cần Thơ (CTU) được thành lập vào năm 1966.",
    "system": "Bạn là công tác viên tư vấn lịch sử của Đại học Cần Thơ."
  }
]
```

## Usage

Run the fine-tuning script:
```bash
python finetune_qwen14b_unsloth.py
```

## Configuration

You can modify the following parameters in `finetune_qwen14b_unsloth.py`:

### Model Configuration
- `MODEL_NAME`: Base model to use (default: "unsloth/Qwen2.5-14B-bnb-4bit")
- `MAX_SEQ_LENGTH`: Maximum sequence length (default: 2048)
- `LOAD_IN_4BIT`: Use 4-bit quantization (default: True)

### LoRA Configuration
- `LORA_R`: LoRA rank (default: 16)
- `LORA_ALPHA`: LoRA alpha (default: 16)
- `LORA_DROPOUT`: LoRA dropout (default: 0)

### Training Configuration
- `NUM_TRAIN_EPOCHS`: Number of epochs (default: 3)
- `PER_DEVICE_TRAIN_BATCH_SIZE`: Batch size (default: 2)
- `GRADIENT_ACCUMULATION_STEPS`: Gradient accumulation (default: 4)
- `LEARNING_RATE`: Learning rate (default: 2e-4)

## Output

The script will create the following directories:

- `./qwen14b_finetuned/lora_adapters/`: LoRA adapter weights only
- `./qwen14b_finetuned/merged_16bit/`: Full merged model in 16-bit
- `./qwen14b_finetuned/merged_4bit/`: Full merged model in 4-bit
- `./qwen14b_finetuned/gguf/`: GGUF format for llama.cpp

## Memory Requirements

- **Training**: ~16GB VRAM (with 4-bit quantization and batch size 2)
- **Inference**: ~8GB VRAM (with 4-bit model)

To reduce memory usage:
- Decrease `PER_DEVICE_TRAIN_BATCH_SIZE`
- Increase `GRADIENT_ACCUMULATION_STEPS`
- Reduce `MAX_SEQ_LENGTH`

## Inference Example

```python
from unsloth import FastLanguageModel

# Load the fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./qwen14b_finetuned/lora_adapters",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# Create prompt
prompt = """<|im_start|>system
Bạn là công tác viên tư vấn lịch sử của Đại học Cần Thơ.<|im_end|>
<|im_start|>user
Đại học Cần Thơ được thành lập vào năm nào?<|im_end|>
<|im_start|>assistant
"""

# Generate response
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response)
```

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce batch size: `PER_DEVICE_TRAIN_BATCH_SIZE = 1`
- Increase gradient accumulation: `GRADIENT_ACCUMULATION_STEPS = 8`
- Reduce sequence length: `MAX_SEQ_LENGTH = 1024`

### Slow Training
- Enable packing: Set `packing=True` in SFTTrainer (for short sequences)
- Increase batch size if you have more VRAM
- Use mixed precision training (automatically enabled)

### Model Not Learning
- Increase learning rate: `LEARNING_RATE = 3e-4`
- Increase number of epochs: `NUM_TRAIN_EPOCHS = 5`
- Adjust LoRA rank: `LORA_R = 32`

## License

This project uses the Qwen model which is subject to its own license terms.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [Qwen Team](https://github.com/QwenLM/Qwen) for the base model
- Can Tho University dataset
