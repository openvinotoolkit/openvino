import os
import sys

# Add OpenVINO Python module path explicitly
# This is required because we are using a custom build, not a pip install
openvino_python_path = r"C:\Users\ssdaj\openvino\bin\intel64\Release\python"
if os.path.exists(openvino_python_path) and openvino_python_path not in sys.path:
    sys.path.insert(0, openvino_python_path)

# Add OpenVINO and TBB DLL paths explicitly for Windows
if os.name == 'nt':
    openvino_bin = r"C:\Users\ssdaj\openvino\bin\intel64\Release"
    tbb_bin = r"C:\Users\ssdaj\openvino\temp\Windows_AMD64\tbb\bin"
    
    # Set OPENVINO_LIB_PATHS environment variable to satisfy OpenVINO check
    if 'OPENVINO_LIB_PATHS' not in os.environ:
        os.environ['OPENVINO_LIB_PATHS'] = openvino_bin

    if os.path.exists(openvino_bin):
        os.add_dll_directory(openvino_bin)
    
    if os.path.exists(tbb_bin):
        os.add_dll_directory(tbb_bin)

import openvino as ov
import numpy as np
from openvino.runtime import opset10 as ops

def apply_incision(model_path, output_path, sparsity_target=0.1, strategy="magnitude", block_size=(32, 32)):
    print(f"Loading model from {model_path}...")
    core = ov.Core()
    model = core.read_model(model_path)

    print(f"Applying {sparsity_target*100}% {strategy} pruning to FFN Down Projection layers...")
    
    pruned_count = 0
    total_weights_pruned = 0
    
    # Iterate over all operations
    for op in model.get_ops():
        if op.get_type_name() == "MatMul":
            # Check inputs
            inputs = op.inputs()
            if len(inputs) < 2:
                continue
            
            # We are looking for the weights, which are usually the second input
            weights_node = inputs[1].get_source_output().get_node()
            
            if weights_node.get_type_name() == "Constant":
                weights = weights_node.get_data()
                shape = weights.shape
                
                # Target FFN Down Projection: [768, 1152] (or transposed [1152, 768] depending on layout)
                # Based on inspect_ir, down_proj has inputs ['Dynamic', [768, 1152]]
                # This implies the weights are [768, 1152] if no transpose, or [1152, 768] if transposed.
                # Let's target specifically the [768, 1152] or [1152, 768] shapes that correspond to FFN.
                # The audit said FFN is [768, 1152].
                
                is_target = False
                if len(shape) == 2:
                    if (shape[0] == 768 and shape[1] == 1152) or (shape[0] == 1152 and shape[1] == 768):
                        # Check name to be sure it's an MLP/FFN
                        friendly_name = op.get_friendly_name()
                        if "mlp" in friendly_name and "down_proj" in friendly_name:
                            is_target = True
                
                if is_target:
                    if strategy == "magnitude":
                        # Calculate threshold
                        abs_weights = np.abs(weights)
                        threshold = np.percentile(abs_weights, sparsity_target * 100)
                        # Create mask
                        mask = abs_weights >= threshold
                    elif strategy == "random":
                        # Random pruning
                        mask = np.random.rand(*weights.shape) >= sparsity_target
                    elif strategy == "block":
                        # Block sparsity
                        H, W = weights.shape
                        bh, bw = block_size
                        
                        # Ensure divisibility
                        if H % bh != 0 or W % bw != 0:
                            print(f"Warning: Layer {op.get_friendly_name()} shape {shape} not divisible by block size {block_size}. Skipping block pruning for this layer.")
                            mask = np.ones_like(weights, dtype=bool)
                        else:
                            # Reshape to (H//bh, bh, W//bw, bw) -> (H//bh, W//bw, bh, bw)
                            blocks = weights.reshape(H // bh, bh, W // bw, bw).transpose(0, 2, 1, 3)
                            # Calculate L1 norm of each block
                            block_norms = np.sum(np.abs(blocks), axis=(2, 3))
                            
                            # Calculate threshold
                            threshold = np.percentile(block_norms, sparsity_target * 100)
                            block_mask = block_norms >= threshold
                            
                            # Expand mask: (H//bh, W//bw) -> (H, W)
                            mask = np.kron(block_mask, np.ones((bh, bw), dtype=bool))
                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")
                    
                    # Apply mask
                    pruned_weights = weights * mask
                    
                    # Count pruned
                    n_pruned = weights.size - np.count_nonzero(pruned_weights)
                    total_weights_pruned += n_pruned
                    
                    # Create new Constant node
                    new_weights_node = ops.constant(pruned_weights, dtype=weights.dtype)
                    new_weights_node.set_friendly_name(weights_node.get_friendly_name() + "_pruned")
                    
                    # Replace the input of the MatMul with the new Constant
                    # We need to replace the output of the old constant with the new one
                    # But simpler: just replace the input connection of the MatMul
                    # op.input(1).replace_source_output(new_weights_node.output(0))
                    # Actually, using ov.replace_node or similar is better, but replacing input is fine.
                    
                    # We need to find the input port index. It's 1.
                    op.input(1).replace_source_output(new_weights_node.output(0))
                    
                    pruned_count += 1
                    # print(f"Pruned {op.get_friendly_name()} (Threshold: {threshold:.6f})")

    print(f"Incision complete.")
    print(f"Targeted Layers: {pruned_count}")
    print(f"Total Weights Pruned: {total_weights_pruned}")
    
    # Serialize
    print(f"Saving pruned model to {output_path}...")
    ov.save_model(model, output_path, compress_to_fp16=False) # Keep original precision for now

if __name__ == "__main__":
    output_dir = "model_ir_pruned"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Phase 5: Determine Carrying Capacity (Magnitude Pruning)
    sparsities = [0.6, 0.7, 0.8, 0.9]
    for s in sparsities:
        print(f"\n--- Generating {int(s*100)}% Pruned Model (Magnitude) ---")
        apply_incision("model_ir/openvino_model.xml", f"{output_dir}/openvino_model_magnitude_{int(s*100)}.xml", sparsity_target=s, strategy="magnitude")

    # Block Sparsity
    print(f"\n--- Generating 70% Block Pruned Model (32x32) ---")
    apply_incision("model_ir/openvino_model.xml", f"{output_dir}/openvino_model_block_70.xml", sparsity_target=0.7, strategy="block", block_size=(32, 32))

