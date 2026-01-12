"""
Inference Script for Fine-tuned Qwen 14B Model
==============================================
This script loads the fine-tuned model and allows interactive testing.
"""

import torch
from unsloth import FastLanguageModel

# Configuration
MODEL_PATH = "./qwen14b_finetuned/merged_4bit"  # Change to your model path
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

def load_model():
    """Load the fine-tuned model."""
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    print("Model loaded successfully!")
    return model, tokenizer

def generate_response(model, tokenizer, instruction, input_text="", system_message=None, 
                     max_new_tokens=512, temperature=0.7, top_p=0.9):
    """Generate a response from the model."""
    
    # Default system message
    if system_message is None:
        system_message = "Bạn là công tác viên tư vấn lịch sử của Đại học Cần Thơ nhân dịp kỷ niệm 60 năm thành lập trường."
    
    # Format the prompt
    if input_text:
        user_message = f"{instruction}\n\n{input_text}"
    else:
        user_message = instruction
    
    prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
    
    # Tokenize
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
    
    # Extract assistant's response
    try:
        assistant_response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
    except:
        assistant_response = response
    
    return assistant_response.strip()

def interactive_mode(model, tokenizer):
    """Run interactive chat mode."""
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("\nCommands:")
    print("  /quit or /exit - Exit the program")
    print("  /system <message> - Change system message")
    print("  /temp <value> - Change temperature (0.0-2.0)")
    print("  /clear - Clear conversation history")
    print("\nType your question and press Enter.\n")
    
    system_message = "Bạn là công tác viên tư vấn lịch sử của Đại học Cần Thơ nhân dịp kỷ niệm 60 năm thành lập trường."
    temperature = 0.7
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['/quit', '/exit']:
                print("\nGoodbye!")
                break
            
            if user_input.startswith('/system '):
                system_message = user_input[8:].strip()
                print(f"✓ System message updated: {system_message}")
                continue
            
            if user_input.startswith('/temp '):
                try:
                    temperature = float(user_input[6:].strip())
                    temperature = max(0.0, min(2.0, temperature))
                    print(f"✓ Temperature set to: {temperature}")
                except:
                    print("✗ Invalid temperature value. Use a number between 0.0 and 2.0")
                continue
            
            if user_input.lower() == '/clear':
                print("✓ Conversation cleared")
                continue
            
            # Generate response
            print("\n🤖 Assistant: ", end="", flush=True)
            response = generate_response(
                model, tokenizer, 
                instruction=user_input,
                system_message=system_message,
                temperature=temperature
            )
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")

def batch_test(model, tokenizer):
    """Run batch testing with predefined questions."""
    print("\n" + "="*80)
    print("BATCH TESTING MODE")
    print("="*80 + "\n")
    
    test_cases = [
        {
            "instruction": "Can Tho University được thành lập vào năm nào?",
            "system": "Bạn là công tác viên tư vấn lịch sử của Đại học Cần Thơ nhân dịp kỷ niệm 60 năm thành lập trường."
        },
        {
            "instruction": "Ai đã trình bày tại diễn đàn Mekong Delta Forum 2019?",
            "system": "Bạn là công tác viên tư vấn lịch sử của Đại học Cần Thơ nhân dịp kỷ niệm 60 năm thành lập trường."
        },
        {
            "instruction": "CTU có vai trò gì trong việc phát triển cộng đồng MD?",
            "system": "Bạn là công tác viên tư vấn lịch sử của Đại học Cần Thơ nhân dịp kỷ niệm 60 năm thành lập trường."
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Question: {test['instruction']}")
        print(f"\nResponse: ", end="", flush=True)
        
        response = generate_response(
            model, tokenizer,
            instruction=test['instruction'],
            system_message=test.get('system')
        )
        print(response)
        print()

def main():
    """Main function."""
    print("="*80)
    print("QWEN 14B FINE-TUNED MODEL - INFERENCE")
    print("="*80)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("\n⚠ WARNING: CUDA not available. Running on CPU will be very slow.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    else:
        print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model, tokenizer = load_model()
    
    # Choose mode
    print("\nSelect mode:")
    print("1. Interactive chat")
    print("2. Batch testing")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        interactive_mode(model, tokenizer)
    elif choice == "2":
        batch_test(model, tokenizer)
    elif choice == "3":
        batch_test(model, tokenizer)
        interactive_mode(model, tokenizer)
    else:
        print("Invalid choice. Running interactive mode...")
        interactive_mode(model, tokenizer)

if __name__ == "__main__":
    main()
