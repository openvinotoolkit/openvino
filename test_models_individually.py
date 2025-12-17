"""
Memory-efficient testing script - tests models one at a time.
"""
import openvino as ov
import numpy as np
import time
import os
import json

# Configuration
DENSE_MODEL_PATH = "gemma_ir/openvino_model.xml"
TSSN_ORIGINAL_PATH = "gemma_ir_tssn/openvino_model.xml"
TSSN_EVOLVED_PATH = "gemma_ir_tssn/evolved_checkpoint.xml"
EXT_PATH = os.path.abspath("src/custom_ops/build/Release/openvino_tssn_extension.dll")
GPU_CONFIG = os.path.abspath("src/custom_ops/composite_tssn_gpu.xml")
DEVICE = "CPU"  # Changed to CPU to avoid GPU memory issues

NUM_WARMUP = 5
NUM_ITERATIONS = 20
SEQ_LEN = 32

def get_inputs(seq_len=32):
    """Generate random test inputs."""
    np.random.seed(42)  # Fixed seed for reproducibility
    input_ids = np.random.randint(0, 256000, (1, seq_len), dtype=np.int64)
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
    beam_idx = np.zeros((1,), dtype=np.int32)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "beam_idx": beam_idx
    }

def benchmark_model(compiled_model, inputs, warmup=5, iterations=20):
    """Benchmark a compiled model."""
    print(f"  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        compiled_model(inputs)
    
    print(f"  Benchmarking ({iterations} iterations)...")
    latencies = []
    for i in range(iterations):
        start = time.perf_counter()
        result = compiled_model(inputs)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    return {
        "mean_latency": np.mean(latencies),
        "std_latency": np.std(latencies),
        "min_latency": np.min(latencies),
        "max_latency": np.max(latencies),
        "fps": 1000.0 / np.mean(latencies),
        "output": result[0]
    }

def compute_accuracy(reference, target):
    """Compute accuracy metrics."""
    mse = np.mean((reference - target) ** 2)
    mae = np.mean(np.abs(reference - target))
    max_diff = np.max(np.abs(reference - target))
    
    # Cosine similarity
    ref_flat = reference.flatten()
    tgt_flat = target.flatten()
    cos_sim = np.dot(ref_flat, tgt_flat) / (
        np.linalg.norm(ref_flat) * np.linalg.norm(tgt_flat) + 1e-8
    )
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "max_diff": float(max_diff),
        "cosine_similarity": float(cos_sim)
    }

def main():
    print("="*80)
    print("INDIVIDUAL MODEL TESTING (Memory Efficient)")
    print("="*80)
    
    # Initialize
    core = ov.Core()
    print(f"\nLoading extension: {EXT_PATH}")
    core.add_extension(EXT_PATH)
    core.set_property("GPU", {"CONFIG_FILE": GPU_CONFIG})
    
    # Generate inputs
    inputs = get_inputs(SEQ_LEN)
    
    results = {
        "config": {
            "device": DEVICE,
            "seq_len": SEQ_LEN,
            "warmup": NUM_WARMUP,
            "iterations": NUM_ITERATIONS
        }
    }
    
    # Test 1: Teacher (Dense) Model
    print("\n" + "="*80)
    print("TEST 1: TEACHER (DENSE) MODEL")
    print("="*80)
    
    print("Loading model...")
    teacher_model = core.read_model(DENSE_MODEL_PATH)
    teacher_inputs = {k: v for k, v in inputs.items() if k in [i.any_name for i in teacher_model.inputs]}
    
    print(f"Compiling on {DEVICE}...")
    compiled_teacher = core.compile_model(teacher_model, DEVICE)
    
    teacher_results = benchmark_model(compiled_teacher, teacher_inputs, NUM_WARMUP, NUM_ITERATIONS)
    teacher_output = teacher_results.pop("output")
    
    print(f"\nResults:")
    print(f"  Latency: {teacher_results['mean_latency']:.2f} ± {teacher_results['std_latency']:.2f} ms")
    print(f"  FPS:     {teacher_results['fps']:.2f}")
    
    results["teacher"] = teacher_results
    
    # Free memory
    del compiled_teacher
    del teacher_model
    print("\n✓ Teacher model unloaded from memory")
    
    # Test 2: Original TSSN Model
    print("\n" + "="*80)
    print("TEST 2: ORIGINAL TSSN MODEL")
    print("="*80)
    
    print("Loading model...")
    tssn_original = core.read_model(TSSN_ORIGINAL_PATH)
    student_inputs = {k: v for k, v in inputs.items() if k in [i.any_name for i in tssn_original.inputs]}
    
    print(f"Compiling on {DEVICE}...")
    compiled_original = core.compile_model(tssn_original, DEVICE)
    
    original_results = benchmark_model(compiled_original, student_inputs, NUM_WARMUP, NUM_ITERATIONS)
    original_output = original_results.pop("output")
    
    # Compute accuracy vs teacher
    original_accuracy = compute_accuracy(teacher_output, original_output)
    
    print(f"\nPerformance:")
    print(f"  Latency: {original_results['mean_latency']:.2f} ± {original_results['std_latency']:.2f} ms")
    print(f"  FPS:     {original_results['fps']:.2f}")
    print(f"  Speedup: {teacher_results['mean_latency'] / original_results['mean_latency']:.2f}x vs Teacher")
    
    print(f"\nAccuracy vs Teacher:")
    print(f"  MSE:               {original_accuracy['mse']:.6e}")
    print(f"  MAE:               {original_accuracy['mae']:.6e}")
    print(f"  Max Diff:          {original_accuracy['max_diff']:.6f}")
    print(f"  Cosine Similarity: {original_accuracy['cosine_similarity']:.6f}")
    
    results["original_tssn"] = {
        "performance": original_results,
        "accuracy": original_accuracy
    }
    
    # Free memory
    del compiled_original
    del tssn_original
    print("\n✓ Original TSSN model unloaded from memory")
    
    # Test 3: Evolved TSSN Model
    print("\n" + "="*80)
    print("TEST 3: EVOLVED TSSN MODEL")
    print("="*80)
    
    if not os.path.exists(TSSN_EVOLVED_PATH):
        print("⚠️  Evolved model not found. Skipping.")
        results["evolved_tssn"] = None
    else:
        print("Loading model...")
        tssn_evolved = core.read_model(TSSN_EVOLVED_PATH)
        
        print(f"Compiling on {DEVICE}...")
        compiled_evolved = core.compile_model(tssn_evolved, DEVICE)
        
        evolved_results = benchmark_model(compiled_evolved, student_inputs, NUM_WARMUP, NUM_ITERATIONS)
        evolved_output = evolved_results.pop("output")
        
        # Compute accuracy vs teacher
        evolved_accuracy = compute_accuracy(teacher_output, evolved_output)
        
        # Compute change vs original TSSN
        original_vs_evolved = compute_accuracy(original_output, evolved_output)
        
        print(f"\nPerformance:")
        print(f"  Latency: {evolved_results['mean_latency']:.2f} ± {evolved_results['std_latency']:.2f} ms")
        print(f"  FPS:     {evolved_results['fps']:.2f}")
        print(f"  Speedup: {teacher_results['mean_latency'] / evolved_results['mean_latency']:.2f}x vs Teacher")
        
        perf_change = ((original_results['mean_latency'] - evolved_results['mean_latency']) / 
                      original_results['mean_latency']) * 100
        print(f"  Change:  {perf_change:+.2f}% vs Original TSSN")
        
        print(f"\nAccuracy vs Teacher:")
        print(f"  MSE:               {evolved_accuracy['mse']:.6e}")
        print(f"  MAE:               {evolved_accuracy['mae']:.6e}")
        print(f"  Max Diff:          {evolved_accuracy['max_diff']:.6f}")
        print(f"  Cosine Similarity: {evolved_accuracy['cosine_similarity']:.6f}")
        
        # Improvement analysis
        mse_improvement = ((original_accuracy['mse'] - evolved_accuracy['mse']) / 
                          original_accuracy['mse']) * 100
        
        print(f"\nImprovement vs Original TSSN:")
        print(f"  MSE Change:    {mse_improvement:+.2f}%")
        print(f"  Output Diff:   {original_vs_evolved['mse']:.6e} (how much evolved changed)")
        
        results["evolved_tssn"] = {
            "performance": evolved_results,
            "accuracy": evolved_accuracy,
            "vs_original": original_vs_evolved,
            "improvements": {
                "mse_improvement_pct": float(mse_improvement),
                "perf_change_pct": float(perf_change)
            }
        }
        
        # Free memory
        del compiled_evolved
        del tssn_evolved
        print("\n✓ Evolved TSSN model unloaded from memory")
    
    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    if results.get("evolved_tssn"):
        evol = results["evolved_tssn"]["improvements"]
        print(f"\n📊 EVOLUTION IMPACT:")
        print(f"  Accuracy:    {evol['mse_improvement_pct']:+.2f}% MSE change")
        print(f"  Performance: {evol['perf_change_pct']:+.2f}% latency change")
        
        if evol['mse_improvement_pct'] > 0:
            print(f"\n✅ SUCCESS: Evolution IMPROVED accuracy by {evol['mse_improvement_pct']:.2f}%")
        else:
            print(f"\n⚠️  Evolution DECREASED accuracy by {abs(evol['mse_improvement_pct']):.2f}%")
        
        if abs(evol['perf_change_pct']) < 2:
            print(f"✅ Performance maintained (within 2%)")
        elif evol['perf_change_pct'] > 0:
            print(f"🚀 BONUS: Performance improved by {evol['perf_change_pct']:.2f}%!")
        else:
            print(f"⚠️  Performance degraded by {abs(evol['perf_change_pct']):.2f}%")
    
    # Save results
    results_file = "individual_test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("="*80)

if __name__ == "__main__":
    main()
