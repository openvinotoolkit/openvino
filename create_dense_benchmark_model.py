import numpy as np
import struct

def create_dense_data(dim):
    n_synapses = dim
    print(f"Generating {n_synapses} synapses for {dim}x{dim} layer (Dense Identity)...")
    
    # Identity mapping: i -> i
    rows = np.arange(dim, dtype=np.int32)
    cols = np.arange(dim, dtype=np.int32)
    
    indices = np.vstack([rows, cols]).astype(np.int32)
    weights = np.random.choice([-1, 1], n_synapses).astype(np.float32)
    sensitivity = np.random.rand(n_synapses).astype(np.float32)
    
    return indices, weights, sensitivity

def save_model(dim, func_ids, model_name):
    indices, weights, sensitivity = create_dense_data(dim)
    
    bin_filename = f"{model_name}.bin"
    xml_filename = f"{model_name}.xml"
    
    with open(bin_filename, "wb") as f:
        offset = 0
        
        indices_bytes = indices.tobytes()
        f.write(indices_bytes)
        indices_offset = offset
        indices_size = len(indices_bytes)
        offset += indices_size
        
        weights_bytes = weights.tobytes()
        f.write(weights_bytes)
        weights_offset = offset
        weights_size = len(weights_bytes)
        offset += weights_size
        
        sensitivity_bytes = sensitivity.tobytes()
        f.write(sensitivity_bytes)
        sensitivity_offset = offset
        sensitivity_size = len(sensitivity_bytes)
        offset += sensitivity_size
        
    func_ids_str = ",".join(map(str, func_ids))
    
    xml_content = f"""<?xml version="1.0" ?>
<net name="{model_name}" version="10">
	<layers>
		<layer id="0" name="input" type="Parameter" version="opset1">
			<data shape="1,{dim}" element_type="f32"/>
			<output>
				<port id="0" precision="FP32" names="input">
					<dim>1</dim>
					<dim>{dim}</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="indices" type="Const" version="opset1">
			<data element_type="i32" shape="2,{dim}" offset="{indices_offset}" size="{indices_size}"/>
			<output>
				<port id="0" precision="I32">
					<dim>2</dim>
					<dim>{dim}</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="weights" type="Const" version="opset1">
			<data element_type="f32" shape="{dim}" offset="{weights_offset}" size="{weights_size}"/>
			<output>
				<port id="0" precision="FP32">
					<dim>{dim}</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="sensitivity" type="Const" version="opset1">
			<data element_type="f32" shape="{dim}" offset="{sensitivity_offset}" size="{sensitivity_size}"/>
			<output>
				<port id="0" precision="FP32">
					<dim>{dim}</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="composite_tssn" type="CompositeTSSN" version="extension">
			<data output_dim="{dim}" func_ids="{func_ids_str}"/>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>{dim}</dim>
				</port>
				<port id="1" precision="I32">
					<dim>2</dim>
					<dim>{dim}</dim>
				</port>
				<port id="2" precision="FP32">
					<dim>{dim}</dim>
				</port>
				<port id="3" precision="FP32">
					<dim>{dim}</dim>
				</port>
			</input>
			<output>
				<port id="4" precision="FP32" names="output">
					<dim>1</dim>
					<dim>{dim}</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="result" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>{dim}</dim>
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
    print(f"Saved {xml_filename} and {bin_filename}")

if __name__ == "__main__":
    # Create Dense MUL model (Func ID 0)
    save_model(1024, [0], "dense_mul_benchmark")
    
    # Create Dense MIN/MAX model (Func IDs 113, 4049)
    save_model(1024, [113, 4049], "dense_minmax_benchmark")
