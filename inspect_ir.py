import openvino as ov
import sys

def inspect_topology(model_path):
    core = ov.Core()
    try:
        model = core.read_model(model_path)
    except Exception as e:
        print(f"Error reading model: {e}")
        return

    print(f"Inspecting model: {model.friendly_name}")
    
    matmul_count = 0
    gelu_count = 0
    fused_gelu_count = 0
    gather_count = 0
    
    print("\n--- MatMul Nodes ---")
    for op in model.get_ops():
        if op.get_type_name() == "MatMul":
            matmul_count += 1
            # Check inputs
            inputs = op.inputs()
            shapes = [list(inp.get_partial_shape().to_shape()) if inp.get_partial_shape().is_static else "Dynamic" for inp in inputs]
            
            # Try to identify if it's a weight matrix (usually the second input is a Constant)
            is_weight_const = False
            weight_shape = None
            if len(inputs) > 1:
                node = inputs[1].get_source_output().get_node()
                if node.get_type_name() == "Constant":
                    is_weight_const = True
                    weight_shape = shapes[1]

            print(f"MatMul: {op.friendly_name}, Inputs: {shapes}, Constant Weights: {is_weight_const}")

        elif op.get_type_name() == "Gelu":
            gelu_count += 1
        elif "Gelu" in op.get_type_name() and op.get_type_name() != "Gelu":
            # Catching fused versions if they have different names in OV (e.g. internal ops)
            fused_gelu_count += 1
            print(f"Potential Fused Gelu: {op.get_type_name()} - {op.friendly_name}")

        elif op.get_type_name() == "Gather":
            # Check if it's an embedding layer (input 0 is usually the dictionary)
            inputs = op.inputs()
            if len(inputs) > 0:
                inp0 = inputs[0].get_source_output().get_node()
                if inp0.get_type_name() == "Constant":
                    shape = list(inp0.get_output_partial_shape(0).to_shape()) if inp0.get_output_partial_shape(0).is_static else "Dynamic"
                    print(f"Gather (Embedding?): {op.friendly_name}, Dictionary Shape: {shape}")

    print(f"\n--- Summary ---")
    print(f"Total MatMul nodes: {matmul_count}")
    print(f"Total Gelu nodes: {gelu_count}")
    print(f"Total Fused/Other Gelu nodes: {fused_gelu_count}")
    print(f"Total Gather nodes: {gather_count}")

    # Check for the specific MatFormer structure criteria
    # "Distinct MatMul nodes with weights of shape [Hidden, 16384]"
    # EmbeddingGemma-300m hidden size is likely 256, 768, or similar. 
    # The vocab size or intermediate size might be large.
    
if __name__ == "__main__":
    inspect_topology("model_ir/openvino_model.xml")
