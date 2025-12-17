"""Analyze the evolution results by comparing original and evolved models."""
import openvino as ov
import numpy as np
import os

ORIGINAL_MODEL = "gemma_ir_tssn/openvino_model.xml"
EVOLVED_MODEL = "gemma_ir_tssn/evolved_checkpoint.xml"
EXT_PATH = os.path.abspath("src/custom_ops/build/Release/openvino_tssn_extension.dll")

print("=== Evolution Analysis ===\n")

# Load models
print("Loading extension...")
core = ov.Core()
core.add_extension(EXT_PATH)
print("Loading models...")
original = core.read_model(ORIGINAL_MODEL)
evolved = core.read_model(EVOLVED_MODEL)

# Find TSSN layers
print("Finding TSSN layers...")
orig_tssn = []
evol_tssn = []

for op in original.get_ops():
    if op.get_type_name() == "CompositeTSSN":
        orig_tssn.append(op)

for op in evolved.get_ops():
    if op.get_type_name() == "CompositeTSSN":
        evol_tssn.append(op)

print(f"Found {len(orig_tssn)} TSSN layers in both models.\n")

# Compare function IDs
changes = 0
total_neurons = 0
changed_neurons = 0

for i, (orig_op, evol_op) in enumerate(zip(orig_tssn, evol_tssn)):
    # Get function IDs (input 6)
    orig_func = orig_op.input_value(6).get_node().data
    evol_func = evol_op.input_value(6).get_node().data
    
    total_neurons += orig_func.size
    
    # Compare
    diff = np.sum(orig_func != evol_func)
    if diff > 0:
        changes += 1
        changed_neurons += diff
        print(f"Layer {i} ({orig_op.get_friendly_name()}):")
        print(f"  Changed neurons: {diff}/{orig_func.size} ({100*diff/orig_func.size:.2f}%)")
        
        # Show some examples
        changed_idx = np.where(orig_func != evol_func)[0][:5]
        print(f"  Example changes (first 5):")
        for idx in changed_idx:
            print(f"    Neuron {idx}: {orig_func.flat[idx]} -> {evol_func.flat[idx]}")
        print()

print("\n=== Summary ===")
print(f"Total TSSN layers: {len(orig_tssn)}")
print(f"Layers with mutations: {changes}")
print(f"Total neurons: {total_neurons}")
print(f"Changed neurons: {changed_neurons} ({100*changed_neurons/total_neurons:.4f}%)")

if changes > 0:
    print("\n✅ SUCCESS: The evolution run DID make changes!")
    print(f"The model was successfully evolved with mutations in {changes} layers.")
else:
    print("\n⚠️  WARNING: No changes detected. Evolution may have failed or found no improvements.")
