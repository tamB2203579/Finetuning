# Qwen 14B Fine-tuning with Unsloth and QLoRA

This project fine-tunes any Qwen 14B base model using Unsloth and QLoRA with 4-bit quantization, then exports to GGUF format for Ollama deployment while preserving the original model's tool capabilities.

## Features

- **Unsloth Optimization**: 2x faster training with less memory usage
- **QLoRA 4-bit Quantization**: Efficient training on consumer GPUs
- **Alpaca Format Support**: Works with instruction/input/output/system format
- **GGUF Export**: Multiple quantization methods for Ollama
- **Tool Preservation**: Maintains original model's tool/function calling abilities
- **Multiple Output Formats**: LoRA adapters, merged 16-bit, merged 4-bit, and GGUF

## Requirements

- NVIDIA GPU with CUDA support (minimum 16GB VRAM recommended)
- Python 3.10+
- CUDA 11.8 or higher

## Installation

1. **Clone or download this repository**

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

Note: If you encounter issues with `unsloth`, you can install it directly:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## Dataset Format

The script expects a JSON file with Alpaca format:

```json
[
  {
    "instruction": "Your question or task here",
    "input": "Optional additional context",
    "output": "Expected response",
    "system": "Optional system prompt"
  }
]
```

Your `dataset.json` is already in the correct format!

## Usage

### 1. Basic Training

Simply run the script:

```bash
python finetune_qwen14b_unsloth.py
```

### 2. Monitor Training

The script will display:
- Dataset loading progress
- Model configuration
- Training progress with loss metrics
- Validation results
- Export progress

### 3. Output Structure

After training, you'll have:

```
qwen14b_finetuned/
├── lora_adapters/       # LoRA adapter weights (smallest, ~100MB)
├── merged_16bit/        # Full merged model in 16-bit
├── merged_4bit/         # Full merged model in 4-bit
└── gguf/                # GGUF models for Ollama
    ├── qwen14b_finetuned_q4_k_m.gguf
    ├── qwen14b_finetuned_q5_k_m.gguf
    └── qwen14b_finetuned_q8_0.gguf
```

## Using with Ollama

### Method 1: Using GGUF (Recommended)

1. **Create a Modelfile:**

```bash
cat > Modelfile << EOF
FROM ./qwen14b_finetuned/gguf/qwen14b_finetuned_q4_k_m.gguf

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF
```

2. **Create the Ollama model:**

```bash
ollama create qwen14b-finetuned -f Modelfile
```

3. **Run the model:**

```bash
ollama run qwen14b-finetuned
```

### Method 2: Using llama.cpp Directly

If you want to use llama.cpp for conversion:

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build
make

# Convert the 16-bit model to GGUF
python convert.py ../qwen14b_finetuned/merged_16bit --outtype f16 --outfile qwen14b_finetuned_f16.gguf

# Quantize to different formats
./quantize qwen14b_finetuned_f16.gguf qwen14b_finetuned_q4_k_m.gguf q4_k_m
./quantize qwen14b_finetuned_f16.gguf qwen14b_finetuned_q5_k_m.gguf q5_k_m
./quantize qwen14b_finetuned_f16.gguf qwen14b_finetuned_q8_0.gguf q8_0
```

## Use Different Qwen Models

The script works with any Qwen 14B model:

```python
# Qwen 3
MODEL_NAME = "Qwen/Qwen3-14B"
MODEL_NAME = "Qwen/Qwen3-14B-Instruct"

# Qwen 2.5
MODEL_NAME = "Qwen/Qwen2.5-14B"
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

# Qwen 2
MODEL_NAME = "Qwen/Qwen2-14B"
MODEL_NAME = "Qwen/Qwen2-14B-Instruct"

```

## Preserving Tool Capabilities

The script preserves the original model's tool/function calling abilities by:

1. **Using the correct chat template** (Qwen's `<|im_start|>` format)
2. **Training on all target modules** (attention and MLP layers)
3. **Not modifying the tokenizer** or special tokens
4. **Maintaining the original model architecture**

If your base model supports tools, the fine-tuned version will too!

## Performance Tips

1. **Use bf16 if available:**
   - Automatically detected and used on Ampere GPUs (RTX 30xx, A100, etc.)

2. **Enable Flash Attention:**
   - Automatically used by Unsloth when available

3. **Optimize batch size:**
   - Find the largest batch size that fits in your GPU memory
   - Use gradient accumulation to maintain effective batch size

4. **Use packing for efficiency:**
   ```python
   trainer = SFTTrainer(
       # ... other arguments ...
       packing=True  # Packs multiple samples into one sequence
   )
   ```

## Troubleshooting

### Out of Memory (OOM)

- Reduce `BATCH_SIZE` to 1
- Reduce `MAX_SEQ_LENGTH` to 1024 or 512
- Increase `GRADIENT_ACCUMULATION_STEPS`

### Slow Training

- Ensure you're using a GPU (check with `nvidia-smi`)
- Verify CUDA is properly installed
- Check if bf16/fp16 is being used

### Import Errors

```bash
# Reinstall unsloth
pip uninstall unsloth -y
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Update transformers
pip install --upgrade transformers
```

### GGUF Export Fails

The script includes fallback methods. If one fails, try:

```bash
# Use llama.cpp directly (see Method 2 above)
```

## Example Inference

After training, test your model:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./qwen14b_finetuned/merged_4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

prompt = """<|im_start|>system
Bạn là công tác viên tư vấn lịch sử của Đại học Cần Thơ.<|im_end|>
<|im_start|>user
Can Tho University được thành lập vào năm nào?<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

## Citation

If you use this project, please cite:

```bibtex
@software{unsloth,
  author = {Unsloth AI},
  title = {Unsloth: Fast Language Model Fine-tuning},
  year = {2024},
  url = {https://github.com/unslothai/unsloth}
}

@article{qwen3,
  title={Qwen3 Technical Report},
  author={Qwen Team},
  year={2025}
}
```

## License

This project is provided as-is for educational and research purposes. Please check the licenses of:
- Unsloth: Apache 2.0
- Qwen models: Check model card on HuggingFace
- Your dataset: Ensure you have rights to use it

## Support

For issues:
- Unsloth: https://github.com/unslothai/unsloth/issues
- Qwen: https://github.com/QwenLM/Qwen/issues
- This project: Open an issue in this repository

## Acknowledgments

- **Unsloth AI** for the amazing optimization library
- **Qwen Team** for the excellent base models
- **HuggingFace** for the transformers library
- **llama.cpp** for GGUF format and quantization tools
