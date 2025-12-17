import numpy as np
import struct

def create_sparse_data(input_dim, output_dim, sparsity, distribution="hyperbolic"):
    total_weights = input_dim * output_dim
    n_synapses = int(total_weights * (1 - sparsity))
    
    print(f"Generating {n_synapses} synapses for {input_dim}x{output_dim} layer ({sparsity*100:.1f}% sparse, {distribution})...")
    
    if distribution == "hyperbolic":
        # Hyperbolic / Scale-Free Topology
        # Use Zipfian distribution for input indices (rows) to create "hubs"
        # Some inputs will be connected to MANY outputs (super-spreaders)
        # We use a rank-based probability: P(i) ~ 1/(i+1)^alpha
        alpha = 1.2
        ranks = np.arange(1, input_dim + 1)
        probs = 1.0 / (ranks ** alpha)
        probs /= probs.sum()
        
        # Sample inputs (rows) based on popularity
        rows = np.random.choice(input_dim, n_synapses, p=probs).astype(np.int32)
        
        # Sample outputs (cols) uniformly (or could also be hyperbolic for "rich-club")
        cols = np.random.randint(0, output_dim, n_synapses).astype(np.int32)
        
        # Remove duplicates (multiple synapses between same pair)
        # This might reduce n_synapses slightly, but that's fine for sparsity
        # We use a set of tuples to filter
        # For speed, we can use a structured array
        synapses = np.zeros(n_synapses, dtype=[('r', 'i4'), ('c', 'i4')])
        synapses['r'] = rows
        synapses['c'] = cols
        synapses = np.unique(synapses)
        
        rows = synapses['r']
        cols = synapses['c']
        n_synapses = len(rows)
        print(f"  -> Actual synapses after dedup: {n_synapses}")
        
    else:
        # Uniform Random
        flat_indices = np.random.choice(total_weights, n_synapses, replace=False)
        rows = flat_indices // output_dim
        cols = flat_indices % output_dim
    
    # Sort by output index (cols) to support CSC-like access on GPU
    # np.lexsort((secondary, primary)) -> Sort by rows then cols
    sort_idx = np.lexsort((rows, cols))
    rows = rows[sort_idx]
    cols = cols[sort_idx]
    
    indices = np.vstack([rows, cols]).astype(np.int32)
    weights = np.random.choice([-1, 1], n_synapses).astype(np.float32)
    sensitivity = np.random.rand(n_synapses).astype(np.float32)
    
    # Compute CSC info
    counts = np.bincount(cols, minlength=output_dim).astype(np.int32)
    # Starts is cumulative sum of counts (exclusive)
    starts = np.zeros(output_dim, dtype=np.int32)
    starts[1:] = np.cumsum(counts)[:-1]
    
    # Generate Function IDs per neuron (output)
    # 0: SUM, 1: MIN, 2: MAX, 3: T_WAVE, 4: IF
    function_ids = np.random.randint(0, 5, output_dim).astype(np.int32)
    
    return indices, weights, sensitivity, counts, starts, function_ids

def save_model(input_dim, output_dim, sparsity, func_ids, model_name):
    indices, weights, sensitivity, counts, starts, function_ids = create_sparse_data(input_dim, output_dim, sparsity)
    
    # Save data to bin for reference, but we will use Parameters in XML for debugging
    bin_filename = f"{model_name}.bin"
    xml_filename = f"{model_name}.xml"
    
    # We still save the bin file because we might want to load it in python to get the data
    with open(bin_filename, "wb") as f:
        # ... (Same writing logic)
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
        
        counts_bytes = counts.tobytes()
        f.write(counts_bytes)
        counts_offset = offset
        counts_size = len(counts_bytes)
        offset += counts_size
        
        starts_bytes = starts.tobytes()
        f.write(starts_bytes)
        starts_offset = offset
        starts_size = len(starts_bytes)
        offset += starts_size
        
        function_ids_bytes = function_ids.tobytes()
        f.write(function_ids_bytes)
        function_ids_offset = offset
        function_ids_size = len(function_ids_bytes)
        offset += function_ids_size

    func_ids_str = ",".join(map(str, func_ids))
    
    # Change Const to Parameter for debugging
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
        <layer id="100" name="input_identity" type="Convert" version="opset1">
            <data destination_type="f32"/>
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>{input_dim}</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>{input_dim}</dim>
                </port>
            </output>
        </layer>
		<layer id="1" name="indices" type="Parameter" version="opset1">
			<data shape="2,{indices.shape[1]}" element_type="i32"/>
			<output>
				<port id="0" precision="I32" names="indices">
					<dim>2</dim>
					<dim>{indices.shape[1]}</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="weights" type="Parameter" version="opset1">
			<data shape="{weights.shape[0]}" element_type="f32"/>
			<output>
				<port id="0" precision="FP32" names="weights">
					<dim>{weights.shape[0]}</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="sensitivity" type="Parameter" version="opset1">
			<data shape="{sensitivity.shape[0]}" element_type="f32"/>
			<output>
				<port id="0" precision="FP32" names="sensitivity">
					<dim>{sensitivity.shape[0]}</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="counts" type="Parameter" version="opset1">
			<data shape="{counts.shape[0]}" element_type="i32"/>
			<output>
				<port id="0" precision="I32" names="counts">
					<dim>{counts.shape[0]}</dim>
				</port>
			</output>
		</layer>
		<layer id="5" name="starts" type="Parameter" version="opset1">
			<data shape="{starts.shape[0]}" element_type="i32"/>
			<output>
				<port id="0" precision="I32" names="starts">
					<dim>{starts.shape[0]}</dim>
				</port>
			</output>
		</layer>
		<layer id="6" name="function_ids" type="Parameter" version="opset1">
			<data shape="{function_ids.shape[0]}" element_type="i32"/>
			<output>
				<port id="0" precision="I32" names="function_ids">
					<dim>{function_ids.shape[0]}</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="composite_tssn" type="CompositeTSSN" version="extension">
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
				<port id="4" precision="I32">
					<dim>{counts.shape[0]}</dim>
				</port>
				<port id="5" precision="I32">
					<dim>{starts.shape[0]}</dim>
				</port>
				<port id="6" precision="I32">
					<dim>{function_ids.shape[0]}</dim>
				</port>
			</input>
			<output>
				<port id="7" precision="FP32" names="output">
					<dim>1</dim>
					<dim>{output_dim}</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="output" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>{output_dim}</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="100" to-port="0"/>
		<edge from-layer="100" from-port="1" to-layer="7" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="7" to-port="1"/>
		<edge from-layer="2" from-port="0" to-layer="7" to-port="2"/>
		<edge from-layer="3" from-port="0" to-layer="7" to-port="3"/>
		<edge from-layer="4" from-port="0" to-layer="7" to-port="4"/>
		<edge from-layer="5" from-port="0" to-layer="7" to-port="5"/>
		<edge from-layer="6" from-port="0" to-layer="7" to-port="6"/>
		<edge from-layer="7" from-port="7" to-layer="8" to-port="0"/>
	</edges>
</net>
"""
    with open(xml_filename, "w") as f:
        f.write(xml_content)
    print(f"Saved {xml_filename} and {bin_filename}")
    
    return indices, weights, sensitivity, counts, starts, function_ids

if __name__ == "__main__":
    # Create model with evolved functions
    # IDs from simulation: 427, 415, 252, 463
    save_model(1024, 1024, 0.9, [427, 415, 252, 463], "evolved_tssn_benchmark")
