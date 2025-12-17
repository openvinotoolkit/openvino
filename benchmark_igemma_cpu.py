
import openvino as ov
import numpy as np
import time
import os
import sys

# Force UTF-8 for stdout to handle emojis correctly in Windows subprocess
sys.stdout.reconfigure(encoding='utf-8')

def benchmark():
    core = ov.Core()

    # Load Extension
    ext_path = os.path.abspath("src/custom_ops/build/Release/openvino_tssn_extension.dll")
    if os.path.exists(ext_path):
        core.add_extension(ext_path)
        print(f"✅ Loaded Extension: {ext_path}")
    else:
        print(f"❌ Extension not found: {ext_path}")
        return

    model_path = "gemma_ir_tssn/openvino_model.xml"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    print(f"📖 Reading Model: {model_path}")
    model = core.read_model(model_path)

    device = "CPU" # Focusing on our AVX2 win
    print(f"⚙️ Compiling for {device}...")
    compiled_model = core.compile_model(model, device)

    print("🔍 Model Inputs:")
    for input in model.inputs:
        print(f"  - {input.any_name}: {input.get_partial_shape()}")

    # Create dummy inputs
    # Gemma inputs: input_ids, attention_mask, position_ids, beam_idx
    seq_len = 128
    batch_size = 1
    inputs = {
        "input_ids": np.random.randint(0, 256000, (batch_size, seq_len), dtype=np.int64),
        "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
        "beam_idx": np.zeros((batch_size,), dtype=np.int32) # Required for stateful models
    }

    # Filter inputs
    model_inputs = {input.any_name: inputs[input.any_name] for input in model.inputs if input.any_name in inputs}
    print(f"👉 Provided Inputs: {list(model_inputs.keys())}")

    print("🔥 Warmup...")
    request = compiled_model.create_infer_request()
    # Reset state is crucial for stateful models
    if hasattr(request, 'reset_state'):
        request.reset_state()

    for _ in range(5):
        request.infer(model_inputs)

    print("⏱️ Benchmarking (20 iters)...")
    start = time.time()
    for _ in range(20):
        request.infer(model_inputs)
    end = time.time()

    avg_time = (end - start) / 20
    print(f"✅ Average Inference Time: {avg_time*1000:.2f} ms")
    print(f"🚀 FPS: {1/avg_time:.2f}")

if __name__ == "__main__":
    benchmark()
