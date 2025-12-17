"""Test the evolved model's accuracy and performance."""
import openvino as ov
import numpy as np
import time
import os

# Configuration
DENSE_MODEL_PATH = "gemma_ir/openvino_model.xml"
TSSN_ORIGINAL_PATH = "gemma_ir_tssn/openvino_model.xml"
TSSN_EVOLVED_PATH = "gemma_ir_tssn/evolved_checkpoint.xml"
EXT_PATH = os.path.abspath("src/custom_ops/build/Release/openvino_tssn_extension.dll")
GPU_CONFIG = os.path.abspath("src/custom_ops/composite_tssn_gpu.xml")
DEVICE = "GPU"

# Test parameters
NUM_WARMUP = 5
NUM_ITERATIONS = 20
SEQ_LENGTHS = [32, 64, 128]

def get_inputs(seq_len=32):
    """Generate random test inputs."""
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
    # Warmup
    for _ in range(warmup):
        compiled_model(inputs)
    
    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = compiled_model(inputs)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    return {
        "mean_latency": np.mean(latencies),
        "std_latency": np.std(latencies),
        "min_latency": np.min(latencies),
        "max_latency": np.max(latencies),
        "fps": 1000.0 / np.mean(latencies)
    }

def compute_accuracy(teacher_output, student_output):
    """Compute accuracy metrics between teacher and student."""
    mse = np.mean((teacher_output - student_output) ** 2)
    mae = np.mean(np.abs(teacher_output - student_output))
    
    # Relative error
    rel_error = np.mean(np.abs(teacher_output - student_output) / (np.abs(teacher_output) + 1e-8))
    
    # Cosine similarity
    teacher_flat = teacher_output.flatten()
    student_flat = student_output.flatten()
    cos_sim = np.dot(teacher_flat, student_flat) / (
        np.linalg.norm(teacher_flat) * np.linalg.norm(student_flat) + 1e-8
    )
    
    return {
        "mse": mse,
        "mae": mae,
        "relative_error": rel_error,
        "cosine_similarity": cos_sim
    }

def main():
    print("=" * 80)
    print("EVOLVED MODEL EVALUATION")
    print("=" * 80)
    
    # Initialize
    core = ov.Core()
    print(f"\n[1/6] Loading extension: {EXT_PATH}")
    core.add_extension(EXT_PATH)
    core.set_property("GPU", {"CONFIG_FILE": GPU_CONFIG})
    
    # Load models
    print(f"[2/6] Loading Teacher (Dense)...")
    teacher_model = core.read_model(DENSE_MODEL_PATH)
    
    print(f"[3/6] Loading Student Original (TSSN)...")
    tssn_original = core.read_model(TSSN_ORIGINAL_PATH)
    
    print(f"[4/6] Loading Student Evolved (TSSN)...")
    tssn_evolved = core.read_model(TSSN_EVOLVED_PATH)
    
    # Note: Compile models one at a time to avoid GPU memory issues
    print(f"[5/6] Models loaded successfully")
    print(f"[6/6] Running tests (compiling on-demand)...\n")
    
    # Test across different sequence lengths
    all_results = []
    
    for seq_len in SEQ_LENGTHS:
        print(f"\n{'='*80}")
        print(f"SEQUENCE LENGTH: {seq_len}")
        print(f"{'='*80}\n")
        
        # Generate inputs
        inputs = get_inputs(seq_len)
        
        # Filter inputs for each model
        teacher_inputs = {k: v for k, v in inputs.items() if k in [i.any_name for i in teacher_model.inputs]}
        student_inputs = {k: v for k, v in inputs.items() if k in [i.any_name for i in tssn_original.inputs]}
        
        # Get outputs (compile one at a time to manage GPU memory)
        print("Running inference on Teacher...")
        compiled_teacher = core.compile_model(teacher_model, DEVICE)
        teacher_output = compiled_teacher(teacher_inputs)[0]
        del compiled_teacher  # Free GPU memory
        
        print("Running inference on Original TSSN...")
        compiled_original = core.compile_model(tssn_original, DEVICE)
        original_output = compiled_original(student_inputs)[0]
        del compiled_original  # Free GPU memory
        
        print("Running inference on Evolved TSSN...")
        compiled_evolved = core.compile_model(tssn_evolved, DEVICE)
        evolved_output = compiled_evolved(student_inputs)[0]
        
        # Compute accuracy
        print("\n--- ACCURACY METRICS ---")
        original_acc = compute_accuracy(teacher_output, original_output)
        evolved_acc = compute_accuracy(teacher_output, evolved_output)
        
        print(f"\nOriginal TSSN vs Teacher:")
        print(f"  MSE:               {original_acc['mse']:.6e}")
        print(f"  MAE:               {original_acc['mae']:.6e}")
        print(f"  Relative Error:    {original_acc['relative_error']:.6f}")
        print(f"  Cosine Similarity: {original_acc['cosine_similarity']:.6f}")
        
        print(f"\nEvolved TSSN vs Teacher:")
        print(f"  MSE:               {evolved_acc['mse']:.6e}")
        print(f"  MAE:               {evolved_acc['mae']:.6e}")
        print(f"  Relative Error:    {evolved_acc['relative_error']:.6f}")
        print(f"  Cosine Similarity: {evolved_acc['cosine_similarity']:.6f}")
        
        # Improvement
        mse_improvement = ((original_acc['mse'] - evolved_acc['mse']) / original_acc['mse']) * 100
        cos_improvement = ((evolved_acc['cosine_similarity'] - original_acc['cosine_similarity']) / 
                          (1 - original_acc['cosine_similarity'])) * 100
        
        print(f"\n🎯 IMPROVEMENT:")
        print(f"  MSE Reduction:     {mse_improvement:+.2f}%")
        print(f"  Cosine Sim Gain:   {cos_improvement:+.2f}%")
        
        # Benchmark performance (recompile for benchmarking)
        print("\n--- PERFORMANCE BENCHMARK ---")
        print(f"Running {NUM_ITERATIONS} iterations with {NUM_WARMUP} warmup...")
        
        print("Benchmarking Teacher...")
        compiled_teacher = core.compile_model(teacher_model, DEVICE)
        teacher_perf = benchmark_model(compiled_teacher, teacher_inputs, NUM_WARMUP, NUM_ITERATIONS)
        del compiled_teacher
        
        print("Benchmarking Original TSSN...")
        compiled_original = core.compile_model(tssn_original, DEVICE)
        original_perf = benchmark_model(compiled_original, student_inputs, NUM_WARMUP, NUM_ITERATIONS)
        del compiled_original
        
        print("Benchmarking Evolved TSSN...")
        evolved_perf = benchmark_model(compiled_evolved, student_inputs, NUM_WARMUP, NUM_ITERATIONS)
        del compiled_evolved
        
        print(f"\nTeacher (Dense):")
        print(f"  Latency: {teacher_perf['mean_latency']:.2f} ± {teacher_perf['std_latency']:.2f} ms")
        print(f"  FPS:     {teacher_perf['fps']:.2f}")
        
        print(f"\nOriginal TSSN:")
        print(f"  Latency: {original_perf['mean_latency']:.2f} ± {original_perf['std_latency']:.2f} ms")
        print(f"  FPS:     {original_perf['fps']:.2f}")
        print(f"  Speedup: {teacher_perf['mean_latency'] / original_perf['mean_latency']:.2f}x")
        
        print(f"\nEvolved TSSN:")
        print(f"  Latency: {evolved_perf['mean_latency']:.2f} ± {evolved_perf['std_latency']:.2f} ms")
        print(f"  FPS:     {evolved_perf['fps']:.2f}")
        print(f"  Speedup: {teacher_perf['mean_latency'] / evolved_perf['mean_latency']:.2f}x")
        
        # Performance comparison
        perf_change = ((original_perf['mean_latency'] - evolved_perf['mean_latency']) / 
                      original_perf['mean_latency']) * 100
        
        print(f"\n⚡ PERFORMANCE CHANGE:")
        print(f"  Evolved vs Original: {perf_change:+.2f}%")
        
        # Store results
        all_results.append({
            "seq_len": seq_len,
            "original_acc": original_acc,
            "evolved_acc": evolved_acc,
            "mse_improvement": mse_improvement,
            "teacher_perf": teacher_perf,
            "original_perf": original_perf,
            "evolved_perf": evolved_perf,
            "perf_change": perf_change
        })
    
    # Final Summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}\n")
    
    avg_mse_improvement = np.mean([r['mse_improvement'] for r in all_results])
    avg_perf_change = np.mean([r['perf_change'] for r in all_results])
    
    print(f"Average MSE Improvement:    {avg_mse_improvement:+.2f}%")
    print(f"Average Performance Change: {avg_perf_change:+.2f}%")
    
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    if avg_mse_improvement > 0:
        print(f"✅ SUCCESS: Evolution improved accuracy by {avg_mse_improvement:.2f}%")
    else:
        print(f"⚠️  WARNING: Evolution decreased accuracy by {abs(avg_mse_improvement):.2f}%")
    
    if abs(avg_perf_change) < 5:
        print(f"✅ Performance maintained (within 5%): {avg_perf_change:+.2f}%")
    elif avg_perf_change > 5:
        print(f"🚀 BONUS: Performance improved by {avg_perf_change:.2f}%!")
    else:
        print(f"⚠️  Performance degraded by {abs(avg_perf_change):.2f}%")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
