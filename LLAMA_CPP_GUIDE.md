# Using llama.cpp for GGUF Conversion and Quantization

This guide explains how to use llama.cpp to convert your fine-tuned model to GGUF format and create different quantization variants.

## Why Use llama.cpp?

- More control over quantization methods
- Better compatibility with various inference engines
- Ability to create custom quantization mixes
- Benchmark different quantization methods

## Installation

### Windows

```powershell
# Install dependencies
# You need Visual Studio with C++ support or MinGW

# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with CMake
mkdir build
cd build
cmake ..
cmake --build . --config Release

# Or use make (with MinGW)
cd llama.cpp
make
```

### Linux/Mac

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build
make

# For GPU support (CUDA)
make LLAMA_CUBLAS=1

# For GPU support (ROCm/AMD)
make LLAMA_HIPBLAS=1

# For Metal (Mac M1/M2)
make LLAMA_METAL=1
```

## Step-by-Step Conversion

### Step 1: Convert to GGUF (FP16)

First, convert your fine-tuned model to GGUF format in FP16:

```bash
# Navigate to llama.cpp directory
cd llama.cpp

# Convert the model
python convert.py \
    ../qwen14b_finetuned/merged_16bit \
    --outtype f16 \
    --outfile qwen14b_finetuned_f16.gguf
```

**Note:** This creates a large file (~28GB for 14B model). This is your base GGUF file.

### Step 2: Quantize to Different Formats

Now create quantized versions:

#### Q4_K_M (Recommended - Best Balance)

```bash
./quantize qwen14b_finetuned_f16.gguf qwen14b_finetuned_q4_k_m.gguf q4_k_m
```

- **Size:** ~8GB
- **Quality:** Very good
- **Speed:** Fast
- **Use case:** General purpose, best for most users

#### Q5_K_M (Higher Quality)

```bash
./quantize qwen14b_finetuned_f16.gguf qwen14b_finetuned_q5_k_m.gguf q5_k_m
```

- **Size:** ~10GB
- **Quality:** Excellent
- **Speed:** Medium-fast
- **Use case:** When you need better quality and have more VRAM

#### Q8_0 (Highest Quality)

```bash
./quantize qwen14b_finetuned_f16.gguf qwen14b_finetuned_q8_0.gguf q8_0
```

- **Size:** ~15GB
- **Quality:** Near-original
- **Speed:** Slower
- **Use case:** Maximum quality, research purposes

#### Q4_0 (Smallest)

```bash
./quantize qwen14b_finetuned_f16.gguf qwen14b_finetuned_q4_0.gguf q4_0
```

- **Size:** ~7.5GB
- **Quality:** Good
- **Speed:** Fastest
- **Use case:** Limited VRAM, maximum speed

#### Q6_K (Balanced High Quality)

```bash
./quantize qwen14b_finetuned_f16.gguf qwen14b_finetuned_q6_k.gguf q6_k
```

- **Size:** ~12GB
- **Quality:** Very high
- **Speed:** Medium
- **Use case:** High quality with reasonable size

## All Quantization Methods

| Method | Size | Quality | Speed | Recommended For |
|--------|------|---------|-------|-----------------|
| q2_k   | ~5GB | Low | Fastest | Testing only |
| q3_k_m | ~6GB | Fair | Very fast | Limited resources |
| q4_0   | ~7.5GB | Good | Fast | Speed priority |
| q4_k_m | ~8GB | Very good | Fast | **General use** ⭐ |
| q5_0   | ~9GB | Very good | Medium | Balanced |
| q5_k_m | ~10GB | Excellent | Medium | **Quality priority** ⭐ |
| q6_k   | ~12GB | Very high | Medium-slow | High quality |
| q8_0   | ~15GB | Near-original | Slow | **Maximum quality** ⭐ |
| f16    | ~28GB | Original | Slowest | Reference/benchmarking |

## Batch Conversion Script

### Windows (PowerShell)

Create `convert_all.ps1`:

```powershell
# Convert to FP16 first
python convert.py ../qwen14b_finetuned/merged_16bit --outtype f16 --outfile qwen14b_finetuned_f16.gguf

# Quantize to different formats
$methods = @("q4_0", "q4_k_m", "q5_0", "q5_k_m", "q6_k", "q8_0")

foreach ($method in $methods) {
    Write-Host "Quantizing to $method..."
    .\quantize.exe qwen14b_finetuned_f16.gguf "qwen14b_finetuned_$method.gguf" $method
}

Write-Host "All conversions complete!"
```

Run with: `powershell -ExecutionPolicy Bypass -File convert_all.ps1`

### Linux/Mac (Bash)

Create `convert_all.sh`:

```bash
#!/bin/bash

# Convert to FP16 first
python convert.py ../qwen14b_finetuned/merged_16bit --outtype f16 --outfile qwen14b_finetuned_f16.gguf

# Quantize to different formats
methods=("q4_0" "q4_k_m" "q5_0" "q5_k_m" "q6_k" "q8_0")

for method in "${methods[@]}"; do
    echo "Quantizing to $method..."
    ./quantize qwen14b_finetuned_f16.gguf "qwen14b_finetuned_$method.gguf" $method
done

echo "All conversions complete!"
```

Run with: `chmod +x convert_all.sh && ./convert_all.sh`

## Testing with llama.cpp

### Command Line Inference

```bash
# Basic inference
./main -m qwen14b_finetuned_q4_k_m.gguf -p "Can Tho University được thành lập vào năm nào?"

# With parameters
./main -m qwen14b_finetuned_q4_k_m.gguf \
    -p "Can Tho University được thành lập vào năm nào?" \
    --temp 0.7 \
    --top-p 0.9 \
    --top-k 40 \
    --repeat-penalty 1.1 \
    -n 256

# Interactive mode
./main -m qwen14b_finetuned_q4_k_m.gguf -i
```

### Server Mode

```bash
# Start server
./server -m qwen14b_finetuned_q4_k_m.gguf --host 0.0.0.0 --port 8080

# Test with curl
curl http://localhost:8080/completion \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Can Tho University được thành lập vào năm nào?",
        "n_predict": 256,
        "temperature": 0.7
    }'
```

## Benchmarking

Compare different quantization methods:

```bash
# Perplexity test (quality metric)
./perplexity -m qwen14b_finetuned_q4_k_m.gguf -f test_data.txt

# Speed test
time ./main -m qwen14b_finetuned_q4_k_m.gguf -p "Test prompt" -n 100
```

## Using with Ollama

After creating GGUF files, use them with Ollama:

```bash
# Create Modelfile
cat > Modelfile << EOF
FROM ./qwen14b_finetuned_q4_k_m.gguf

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

# Create model
ollama create qwen14b-finetuned -f Modelfile

# Run
ollama run qwen14b-finetuned
```

## Troubleshooting

### "Unknown model architecture"

Make sure you're using the latest llama.cpp:

```bash
cd llama.cpp
git pull
make clean
make
```

### Out of Memory

Use a smaller quantization:

```bash
# Try q4_0 instead of q4_k_m
./quantize model_f16.gguf model_q4_0.gguf q4_0
```

### Slow Conversion

This is normal for large models. The FP16 conversion can take 10-30 minutes.

### Python Dependencies

If `convert.py` fails:

```bash
pip install torch numpy sentencepiece protobuf
```

## Advanced: Custom Quantization Mix

You can create custom quantization mixes for different layers:

```bash
# Example: Use Q6_K for attention, Q4_K for FFN
./quantize model_f16.gguf model_custom.gguf \
    --allow-requantize \
    --quantize-output-tensor \
    --token-embedding-type q8_0 \
    --attention-wv-type q6_k \
    --attention-wo-type q6_k \
    --feed-forward-type q4_k_m
```

## Resources

- llama.cpp GitHub: https://github.com/ggerganov/llama.cpp
- Quantization guide: https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md
- GGUF format spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

## Recommended Workflow

1. **Convert to FP16** (once)
   ```bash
   python convert.py ../qwen14b_finetuned/merged_16bit --outtype f16 --outfile base.gguf
   ```

2. **Create Q4_K_M** (general use)
   ```bash
   ./quantize base.gguf qwen14b_q4_k_m.gguf q4_k_m
   ```

3. **Test quality** with your dataset

4. **If quality is insufficient**, try Q5_K_M or Q6_K

5. **If quality is good**, optionally create Q4_0 for faster inference

6. **Deploy** the best version to Ollama
