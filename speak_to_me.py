
import openvino as ov
import numpy as np
import os
import sys

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def generate_text():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("📦 Installing transformers...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers>=4.38.0"])
        from transformers import AutoTokenizer

    core = ov.Core()

    # Load Extension
    ext_path = os.path.abspath("src/custom_ops/build/Release/openvino_tssn_extension.dll")
    if os.path.exists(ext_path):
        core.add_extension(ext_path)
        print(f"✅ Loaded Extension")

    model_path = "gemma_ir_tssn/openvino_model.xml"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    print(f"📖 Reading Model...")
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, "CPU")
    infer_request = compiled_model.create_infer_request()

    print("🧩 Loading Tokenizer...")
    # Try local first, then HF
    tokenizer_path = "gemma_ir"
    try:
        if os.path.exists(tokenizer_path) and (os.path.exists(os.path.join(tokenizer_path, "tokenizer.model")) or os.path.exists(os.path.join(tokenizer_path, "tokenizer.json"))):
             tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
             print(f"   Using local tokenizer: {tokenizer_path}")
        else:
             print("   Local tokenizer not found, trying 'google/gemma-2b' (requires auth)...")
             tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    except Exception as e:
        print(f"⚠️ Failed to load tokenizer: {e}")
        print("   Trying 'philschmid/gemma-tokenizer' as fallback...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("philschmid/gemma-tokenizer")
        except:
            print("❌ Could not load any tokenizer.")
            return

    prompt = "The future of AI is"
    print(f"📝 Prompt: '{prompt}'")

    input_ids = tokenizer(prompt, return_tensors="np").input_ids

    # Generation Loop (Simple Greedy)
    print("🤖 Generating...", end="", flush=True)

    generated_ids = input_ids.tolist()[0]

    for i in range(20): # Generate 20 tokens
        # Prepare inputs
        current_len = len(generated_ids)
        input_tensor = np.array([generated_ids], dtype=np.int64)
        attention_mask = np.ones((1, current_len), dtype=np.int64)
        position_ids = np.arange(current_len, dtype=np.int64).reshape(1, current_len)
        beam_idx = np.zeros((1,), dtype=np.int32)

        inputs = {
            "input_ids": input_tensor,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "beam_idx": beam_idx
        }

        # Reset state because we are passing the full sequence history every time
        if hasattr(infer_request, 'reset_state'):
            infer_request.reset_state()

        res = infer_request.infer(inputs)

        # Get logits (output 0)
        logits = res[compiled_model.output(0)]
        next_token_logits = logits[0, -1, :]
        next_token = np.argmax(next_token_logits)

        generated_ids.append(next_token)
        print(".", end="", flush=True)

        if next_token == tokenizer.eos_token_id:
            break

    print() # Newline
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"🗣️ Result: {output_text}")

if __name__ == "__main__":
    generate_text()
