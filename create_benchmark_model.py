import numpy as np
import openvino as ov
import struct

def create_sparse_data(input_dim, output_dim, sparsity):
    total_weights = input_dim * output_dim
    n_synapses = int(total_weights * (1 - sparsity))
    
    print(f"Generating {n_synapses} synapses for {input_dim}x{output_dim} layer ({sparsity*100:.1f}% sparse)...")
    
    # Generate random indices
    # We need unique pairs.
    # Strategy: Sample from total range and convert to (row, col)
    flat_indices = np.random.choice(total_weights, n_synapses, replace=False)
    flat_indices.sort()
    
    rows = flat_indices // output_dim
    cols = flat_indices % output_dim
    
    indices = np.vstack([rows, cols]).astype(np.int32)
    
    # Random weights {-1, 1}
    weights = np.random.choice([-1, 1], n_synapses).astype(np.float32)
    
    # Random sensitivity [0, 1]
    sensitivity = np.random.rand(n_synapses).astype(np.float32)
    
    return indices, weights, sensitivity

def save_model(input_dim, output_dim, sparsity, func_ids, model_name="composite_tssn_benchmark"):
    indices, weights, sensitivity = create_sparse_data(input_dim, output_dim, sparsity)
    
    # Create .bin file
    bin_filename = f"{model_name}.bin"
    xml_filename = f"{model_name}.xml"
    
    with open(bin_filename, "wb") as f:
        # We need to track offsets
        offset = 0
        
        # 1. Indices
        indices_bytes = indices.tobytes()
        f.write(indices_bytes)
        indices_offset = offset
        indices_size = len(indices_bytes)
        offset += indices_size
        
        # 2. Weights
        weights_bytes = weights.tobytes()
        f.write(weights_bytes)
        weights_offset = offset
        weights_size = len(weights_bytes)
        offset += weights_size
        
        # 3. Sensitivity
        sensitivity_bytes = sensitivity.tobytes()
        f.write(sensitivity_bytes)
        sensitivity_offset = offset
        sensitivity_size = len(sensitivity_bytes)
        offset += sensitivity_size
        
    print(f"Binary file {bin_filename} created. Size: {offset} bytes.")
    
    # Create .xml file
    func_ids_str = ",".join(map(str, func_ids))
    
    xml_content = f"""<?xml version="1.0" ?>
<net name="{model_name}" version="10">
	<layers>
		<layer id="0" name="input" type="Parameter" version="opset1">
			<data shape="1,{input_dim}" element_type="f32"/>
			<output>
				<port id="0" precision="FP32" names="input">
					<dim>1</dim>
					<dim>{input_dim}</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="indices" type="Const" version="opset1">
			<data element_type="i32" shape="2,{indices.shape[1]}" offset="{indices_offset}" size="{indices_size}"/>
			<output>
				<port id="0" precision="I32">
					<dim>2</dim>
					<dim>{indices.shape[1]}</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="weights" type="Const" version="opset1">
			<data element_type="f32" shape="{weights.shape[0]}" offset="{weights_offset}" size="{weights_size}"/>
			<output>
				<port id="0" precision="FP32">
					<dim>{weights.shape[0]}</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="sensitivity" type="Const" version="opset1">
			<data element_type="f32" shape="{sensitivity.shape[0]}" offset="{sensitivity_offset}" size="{sensitivity_size}"/>
			<output>
				<port id="0" precision="FP32">
					<dim>{sensitivity.shape[0]}</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="composite_tssn" type="CompositeTSSN" version="extension">
			<data output_dim="{output_dim}" func_ids="{func_ids_str}"/>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>{input_dim}</dim>
				</port>
				<port id="1" precision="I32">
					<dim>2</dim>
					<dim>{indices.shape[1]}</dim>
				</port>
				<port id="2" precision="FP32">
					<dim>{weights.shape[0]}</dim>
				</port>
				<port id="3" precision="FP32">
					<dim>{sensitivity.shape[0]}</dim>
				</port>
			</input>
			<output>
				<port id="4" precision="FP32">
					<dim>1</dim>
					<dim>{output_dim}</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="result" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>{output_dim}</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
		<edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
		<edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
		<edge from-layer="4" from-port="4" to-layer="5" to-port="0"/>
	</edges>
</net>
"""
    with open(xml_filename, "w") as f:
        f.write(xml_content)
    print(f"XML file {xml_filename} created.")

if __name__ == "__main__":
    # 96% Sparsity, Composite IDs [233, 179]
    save_model(1024, 1024, 0.96, [233, 179])
