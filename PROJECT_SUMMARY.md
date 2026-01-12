# Qwen 14B Fine-tuning Project - Complete Setup

## 📁 Project Structure

```
Finetuning/
├── dataset.json                    # Your training dataset (Alpaca format)
├── finetune_qwen14b_unsloth.py    # Main training script
├── inference.py                    # Interactive inference script
├── requirements.txt                # Python dependencies
├── config.ini                      # Configuration file
├── Modelfile                       # Ollama model configuration
├── README.md                       # Main documentation
├── LLAMA_CPP_GUIDE.md             # llama.cpp conversion guide
├── quick_start.bat                 # Windows quick start script
├── quick_start.sh                  # Linux/Mac quick start script
├── .gitignore                      # Git ignore rules
└── PROJECT_SUMMARY.md             # This file
```

## 🎯 What This Project Does

This project provides a complete pipeline to:

1. **Fine-tune** any Qwen 14B model on your custom dataset
2. **Optimize** training with Unsloth (2x faster, less memory)
3. **Use QLoRA** with 4-bit quantization (fits on consumer GPUs)
4. **Export** to multiple formats (LoRA, merged, GGUF)
5. **Deploy** to Ollama for easy inference
6. **Preserve** original model capabilities (including tools)

## 🚀 Quick Start

### Option 1: Automated (Recommended)

**Windows:**
```bash
quick_start.bat
```

**Linux/Mac:**
```bash
chmod +x quick_start.sh
./quick_start.sh
```

### Option 2: Manual

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python finetune_qwen14b_unsloth.py

# Test inference
python inference.py
```

## 📊 Dataset Format

Your `dataset.json` is already in the correct Alpaca format:

```json
{
  "instruction": "Question or task",
  "input": "Optional context",
  "output": "Expected answer",
  "system": "Optional system prompt"
}
```

**Current dataset:** 17,066 samples about Can Tho University history

## ⚙️ Configuration

Edit `config.ini` or modify variables in `finetune_qwen14b_unsloth.py`:

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen2.5-14B` | Base model to fine-tune |
| `MAX_SEQ_LENGTH` | `2048` | Maximum sequence length |
| `LORA_R` | `16` | LoRA rank (8-32 recommended) |
| `BATCH_SIZE` | `2` | Batch size per GPU |
| `NUM_TRAIN_EPOCHS` | `3` | Number of training epochs |
| `LEARNING_RATE` | `2e-4` | Learning rate |

### Memory Optimization

If you run out of memory:

1. Reduce `BATCH_SIZE` to `1`
2. Reduce `MAX_SEQ_LENGTH` to `1024`
3. Increase `GRADIENT_ACCUMULATION_STEPS` to `8`

## 📦 Output Files

After training, you'll get:

```
qwen14b_finetuned/
├── lora_adapters/       # LoRA weights only (~100MB)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
│
├── merged_16bit/        # Full model in FP16 (~28GB)
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
│
├── merged_4bit/         # Full model in 4-bit (~8GB)
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
│
└── gguf/                # GGUF files for Ollama
    ├── qwen14b_finetuned_q4_k_m.gguf  (~8GB)
    ├── qwen14b_finetuned_q5_k_m.gguf  (~10GB)
    └── qwen14b_finetuned_q8_0.gguf    (~15GB)
```

## 🔧 Using the Fine-tuned Model

### 1. Python Inference

```bash
python inference.py
```

Features:
- Interactive chat mode
- Batch testing
- Adjustable temperature
- Custom system messages

### 2. Ollama Deployment

```bash
# Create Ollama model
ollama create qwen14b-finetuned -f Modelfile

# Run
ollama run qwen14b-finetuned

# API usage
curl http://localhost:11434/api/generate -d '{
  "model": "qwen14b-finetuned",
  "prompt": "Can Tho University được thành lập vào năm nào?"
}'
```

### 3. llama.cpp

See `LLAMA_CPP_GUIDE.md` for detailed instructions on:
- Converting to GGUF
- Different quantization methods
- Benchmarking
- Server mode

## 🎓 Training Process

The training script automatically:

1. ✅ Loads and validates your dataset
2. ✅ Splits into train/validation (90/10)
3. ✅ Loads model with 4-bit quantization
4. ✅ Applies LoRA adapters
5. ✅ Trains with progress monitoring
6. ✅ Saves checkpoints every 100 steps
7. ✅ Evaluates on validation set
8. ✅ Exports to multiple formats
9. ✅ Tests inference

**Expected time:** 4-8 hours on RTX 3090/4090 (depends on dataset size)

## 📈 Monitoring Training

The script displays:

```
Training samples: 15,359
Validation samples: 1,707

Epoch 1/3
Step 100/2000 | Loss: 1.234 | LR: 0.0002
Step 200/2000 | Loss: 0.987 | LR: 0.00018
...

Validation Loss: 0.856
```

## 🔍 Quality Checks

### Automatic Tests

The script runs inference tests after training:

```python
test_prompts = [
    "Can Tho University được thành lập vào năm nào?",
    "CTU có vai trò gì trong việc phát triển cộng đồng MD?"
]
```

### Manual Testing

```bash
python inference.py
```

Then ask questions from your dataset to verify quality.

## 🛠️ Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch size, sequence length |
| CUDA not found | Install CUDA toolkit, PyTorch with CUDA |
| Slow training | Check GPU usage with `nvidia-smi` |
| Import errors | Reinstall unsloth: `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"` |
| GGUF export fails | Use llama.cpp manually (see guide) |

### Getting Help

1. Check `README.md` for detailed documentation
2. Check `LLAMA_CPP_GUIDE.md` for GGUF conversion
3. Review error messages carefully
4. Check GPU memory with `nvidia-smi`

## 📋 Requirements

### Hardware

- **Minimum:** NVIDIA GPU with 16GB VRAM (RTX 4060 Ti 16GB, RTX 3090)
- **Recommended:** NVIDIA GPU with 24GB+ VRAM (RTX 3090, RTX 4090, A5000)
- **RAM:** 32GB+ system RAM
- **Storage:** 100GB+ free space

### Software

- **OS:** Windows 10/11, Linux (Ubuntu 20.04+), macOS (limited support)
- **Python:** 3.10 or higher
- **CUDA:** 11.8 or higher
- **PyTorch:** 2.1.0 or higher

## 🎯 Use Cases

This fine-tuned model is perfect for:

- ✅ Question answering about Can Tho University
- ✅ Historical information retrieval
- ✅ Vietnamese language understanding
- ✅ Educational chatbots
- ✅ Knowledge base systems

## 🔄 Updating the Model

To fine-tune on new data:

1. Add new samples to `dataset.json`
2. Run training again
3. Or use the previous checkpoint:

```python
# In finetune_qwen14b_unsloth.py
trainer = SFTTrainer(
    # ... other args ...
    resume_from_checkpoint="./qwen14b_finetuned/checkpoint-XXX"
)
```

## 📚 Additional Resources

- **Unsloth:** https://github.com/unslothai/unsloth
- **Qwen:** https://github.com/QwenLM/Qwen
- **llama.cpp:** https://github.com/ggerganov/llama.cpp
- **Ollama:** https://ollama.ai/
- **LoRA Paper:** https://arxiv.org/abs/2106.09685
- **QLoRA Paper:** https://arxiv.org/abs/2305.14314

## 🎉 Success Checklist

After running the project, you should have:

- ✅ Fine-tuned model in multiple formats
- ✅ GGUF files for Ollama deployment
- ✅ Inference script for testing
- ✅ Ollama model ready to use
- ✅ Training logs and checkpoints

## 🚀 Next Steps

1. **Test the model** with `inference.py`
2. **Deploy to Ollama** using the Modelfile
3. **Share your model** (optional)
4. **Integrate into your application**
5. **Fine-tune further** if needed

## 📝 Notes

- The model preserves Qwen's original capabilities (including tools)
- Training is optimized with Unsloth (2x faster)
- Multiple export formats for flexibility
- Comprehensive documentation included
- Ready for production deployment

## 🙏 Acknowledgments

- **Unsloth AI** for the optimization library
- **Qwen Team** for the excellent base models
- **HuggingFace** for the transformers ecosystem
- **llama.cpp** for GGUF format and quantization

---

**Happy Fine-tuning! 🎓**

For questions or issues, please refer to the documentation or create an issue in the repository.
