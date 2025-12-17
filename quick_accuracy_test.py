"""Quick accuracy test on CPU - evolved vs original."""
import openvino as ov
import numpy as np
import os

# Configuration
TSSN_ORIGINAL_PATH = "gemma_ir_tssn/openvino_model.xml"
TSSN_EVOLVED_PATH = "gemma_ir_tssn/evolved_checkpoint.xml"
EXT_PATH = os.path.abspath("src/custom_ops/build/Release/openvino_tssn_extension.dll")

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

print("="*80)
print("QUICK ACCURACY TEST: Evolved vs Original (CPU)")
print("="*80)

core = ov.Core()
print("\nLoading extension...")
core.add_extension(EXT_PATH)

print("Loading models...")
tssn_original = core.read_model(TSSN_ORIGINAL_PATH)
tssn_evolved = core.read_model(TSSN_EVOLVED_PATH)

print("Compiling on CPU...")
compiled_original = core.compile_model(tssn_original, "CPU")
compiled_evolved = core.compile_model(tssn_evolved, "CPU")

# Test with fixed seed for reproducibility
np.random.seed(42)
inputs = get_inputs(32)

print("\nRunning inference...")
output_original = compiled_original(inputs)[0]
output_evolved = compiled_evolved(inputs)[0]

# Compute differences
mse = np.mean((output_original - output_evolved) ** 2)
mae = np.mean(np.abs(output_original - output_evolved))
max_diff = np.max(np.abs(output_original - output_evolved))

# Check how many outputs are similar
similarity_threshold = 0.01
similar_ratio = np.mean(np.abs(output_original - output_evolved) < similarity_threshold)

print("\n" + "="*80)
print("RESULTS: Evolved vs Original TSSN")
print("="*80)
print(f"MSE between outputs:       {mse:.6e}")
print(f"MAE between outputs:       {mae:.6e}")
print(f"Max absolute difference:   {max_diff:.6f}")
print(f"Similar values (< 0.01):   {similar_ratio*100:.2f}%")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

if mse < 1e-10:
    print("⚠️  Models produce IDENTICAL outputs - evolution may not have changed behavior")
elif mse < 1e-6:
    print("✅ Models produce VERY SIMILAR outputs - evolution made minor refinements")
elif mse < 1e-3:
    print("✅ Models produce SIMILAR outputs - evolution made moderate changes")
else:
    print("⚠️  Models produce DIFFERENT outputs - evolution significantly altered behavior")

print("\nNote: To evaluate if evolution IMPROVED the model, we need to compare")
print("both outputs against the dense teacher model.")
print("="*80)
