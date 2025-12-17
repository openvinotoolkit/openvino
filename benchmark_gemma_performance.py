import openvino as ov
import numpy as np
import time
import os
from transformers import AutoTokenizer

def benchmark_model(model_path, device="GPU", use_extension=False, model_name="Model"):
    print(f"--- Benchmarking {model_name} on {device} ---")
    core = ov.Core()
    
    if use_extension:
        ext_path = os.path.abspath("src/custom_ops/build/Release/openvino_tssn_extension.dll")
        gpu_xml_path = os.path.abspath("src/custom_ops/composite_tssn_gpu.xml")
        print(f"Loading extension: {ext_path}")
        core.add_extension(ext_path)
        core.set_property("GPU", {"CONFIG_FILE": gpu_xml_path})
    
    print(f"Reading model: {model_path}")
    model = core.read_model(model_path)
    
    print("Compiling model...")
    start_compile = time.time()
    compiled_model = core.compile_model(model, device)
    print(f"Compilation time: {time.time() - start_compile:.2f}s")
    
    # Prepare inputs (Sequence Length 128)
    seq_len = 128
    input_ids = np.random.randint(0, 256000, (1, seq_len), dtype=np.int64)
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
    beam_idx = np.zeros((1,), dtype=np.int32)
    
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "beam_idx": beam_idx
    }
    
    # Filter inputs to match model
    model_inputs = {input.any_name: inputs[input.any_name] for input in model.inputs if input.any_name in inputs}
    
    # Warmup
    print("Warmup...")
    request = compiled_model.create_infer_request()
    for _ in range(5):
        request.infer(model_inputs)
        
    # Benchmark
    print("Running benchmark loop (50 iterations)...")
    latencies = []
    for _ in range(50):
        start = time.time()
        request.infer(model_inputs)
        latencies.append(time.time() - start)
        
    avg_latency = np.mean(latencies)
    fps = 1.0 / avg_latency
    print(f"Average Latency: {avg_latency*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    return fps

def main():
    # Benchmark Dense Model
    dense_path = "gemma_ir/openvino_model.xml"
    dense_fps = benchmark_model(dense_path, device="GPU", use_extension=False, model_name="Dense Gemma")
    
    # Benchmark TSSN Model
    tssn_path = "gemma_ir_tssn/openvino_model.xml"
    tssn_fps = benchmark_model(tssn_path, device="GPU", use_extension=True, model_name="TSSN Gemma")
    
    print("\n--- Results Summary ---")
    print(f"Dense Gemma FPS: {dense_fps:.2f}")
    print(f"TSSN Gemma FPS:  {tssn_fps:.2f}")
    print(f"Speedup: {tssn_fps / dense_fps:.2f}x")

if __name__ == "__main__":
    main()
