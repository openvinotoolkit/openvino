import openvino as ov
import numpy as np
import pickle
import os

def extract_seed(model_path, output_path, sparsity_target=0.1):
    print(f"Loading model from {model_path}...")
    core = ov.Core()
    model = core.read_model(model_path)

    print(f"Extracting Parasite Seed (Target Sparsity: {sparsity_target*100}%)...")
    
    seed_data = {}
    total_pruned = 0
    
    for op in model.get_ops():
        if op.get_type_name() == "MatMul":
            inputs = op.inputs()
            if len(inputs) < 2:
                continue
            
            weights_node = inputs[1].get_source_output().get_node()
            
            if weights_node.get_type_name() == "Constant":
                weights = weights_node.get_data()
                shape = weights.shape
                
                # Target FFN Down Projection [768, 1152]
                is_target = False
                if len(shape) == 2:
                    if (shape[0] == 768 and shape[1] == 1152) or (shape[0] == 1152 and shape[1] == 768):
                        friendly_name = op.get_friendly_name()
                        if "mlp" in friendly_name and "down_proj" in friendly_name:
                            is_target = True
                
                if is_target:
                    # Calculate threshold
                    abs_weights = np.abs(weights)
                    threshold = np.percentile(abs_weights, sparsity_target * 100)
                    
                    # Identify pruned weights (where |w| < threshold)
                    mask = abs_weights < threshold
                    
                    # Get indices and values
                    indices = np.where(mask)
                    values = weights[mask]
                    
                    seed_data[op.get_friendly_name()] = {
                        "shape": shape,
                        "indices": indices,
                        "values": values,
                        "count": len(values)
                    }
                    
                    total_pruned += len(values)
                    # print(f"Layer {op.get_friendly_name()}: Extracted {len(values)} seeds.")

    print(f"Extraction complete.")
    print(f"Total Parasite Seeds: {total_pruned}")
    
    with open(output_path, 'wb') as f:
        pickle.dump(seed_data, f)
    print(f"Seed data saved to {output_path}")

if __name__ == "__main__":
    extract_seed("model_ir/openvino_model.xml", "parasite_seed.pkl", sparsity_target=0.1)
