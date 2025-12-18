# 05_INFECTION: Apply Cyberspore to Gemma
import sys
import os
import shutil
import platform
from pathlib import Path
import openvino as ov
from openvino.runtime import opset10 as ops
import numpy as np

# --- Setup Environment ---
cwd = Path.cwd()
IS_WINDOWS = platform.system() == "Windows"
IS_COLAB = 'COLAB_GPU' in os.environ or not IS_WINDOWS

# Windows-specific paths (only used when building locally)
if IS_WINDOWS:
    release_bin = cwd / 'bin/intel64/Release'
    tbb_bin = cwd / 'temp/Windows_AMD64/tbb/bin'
    local_python_pkg = cwd / 'bin/intel64/Release/python'
    
    if release_bin.exists() and tbb_bin.exists():
        os.environ['OPENVINO_LIB_PATHS'] = f"{release_bin.absolute()};{tbb_bin.absolute()}"
        
    if local_python_pkg.exists():
        sys.path.insert(0, str(local_python_pkg.absolute()))

# --- Configuration ---
GEMMA_IR_PATH = cwd / "gemma_ir" / "openvino_model.xml"
INFECTED_IR_PATH = cwd / "gemma_ir_tssn" / "openvino_model.xml"

# Cross-platform extension path
if IS_WINDOWS:
    EXTENSION_PATH = cwd / "src/custom_ops/build/Release/openvino_tssn_extension.dll"
else:
    EXTENSION_PATH = cwd / "src/custom_ops/build/libopenvino_tssn_extension.so"

# Ensure output directory exists
INFECTED_IR_PATH.parent.mkdir(parents=True, exist_ok=True)

# Copy tokenizer files
for f in GEMMA_IR_PATH.parent.glob("*.json"):
    shutil.copy(f, INFECTED_IR_PATH.parent / f.name)

print(f"🦠 Infecting Gemma Model...")
print(f"   Source: {GEMMA_IR_PATH}")
print(f"   Target: {INFECTED_IR_PATH}")

def create_tssn_node(core, input_node, weights_data, output_dim, name_prefix):
    """
    Creates a CompositeTSSN node using XML injection.
    Assumes Dense connectivity (Identity Indices).
    weights_data: Flattened weights [Out * In] (Row-Major: Neuron 0 all inputs, Neuron 1 all inputs...)
    """
    input_dim = weights_data.size // output_dim
    n_synapses = weights_data.size
    
    # 1. Create Auxiliary Constants
    # Indices: [2, N]
    # Row 0: Input Index (0, 1, ... In-1, 0, 1, ... In-1)
    # Row 1: Output Index (0, 0, ... 0, 1, 1, ... 1)
    indices = np.zeros((2, n_synapses), dtype=np.int32)
    
    # Vectorized indices generation
    in_idxs = np.tile(np.arange(input_dim, dtype=np.int32), output_dim)
    out_idxs = np.repeat(np.arange(output_dim, dtype=np.int32), input_dim)
    
    indices[0, :] = in_idxs
    indices[1, :] = out_idxs
    
    # Sensitivity: All 1.0
    sensitivity = np.ones(n_synapses, dtype=np.float32)
    
    # Counts: All InputDim
    counts = np.full(output_dim, input_dim, dtype=np.int32)
    
    # Starts: 0, InputDim, 2*InputDim...
    starts = np.arange(0, n_synapses, input_dim, dtype=np.int32)
    
    # FuncIDs: [0] (Standard Mul)
    func_ids = np.array([0], dtype=np.int64)
    
    # 2. Serialize to Binary
    # Sanitize name_prefix for Windows filename compatibility
    safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name_prefix)
    bin_path = cwd / f"temp_{safe_name}.bin"
    with open(bin_path, "wb") as f:
        f.write(indices.tobytes())
        f.write(weights_data.tobytes())
        f.write(sensitivity.tobytes())
        f.write(counts.tobytes())
        f.write(starts.tobytes())
        f.write(func_ids.tobytes())
        
    # Offsets
    off_indices = 0
    off_weights = off_indices + indices.nbytes
    off_sens = off_weights + weights_data.nbytes
    off_counts = off_sens + sensitivity.nbytes
    off_starts = off_counts + counts.nbytes
    off_func = off_starts + starts.nbytes
    
    # 3. Create XML
    # Note: We use a dummy Parameter for input to match the shape
    xml_str = f"""<?xml version="1.0"?>
<net name="TSSN_Injection" version="10">
	<layers>
		<layer id="0" name="input" type="Parameter" version="opset1">
			<data shape="1,{input_dim}" element_type="f32"/>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>{input_dim}</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="indices" type="Const" version="opset1">
			<data element_type="i32" shape="2,{n_synapses}" offset="{off_indices}" size="{indices.nbytes}"/>
			<output>
				<port id="0" precision="I32">
					<dim>2</dim>
					<dim>{n_synapses}</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="weights" type="Const" version="opset1">
			<data element_type="f32" shape="{n_synapses}" offset="{off_weights}" size="{weights_data.nbytes}"/>
			<output>
				<port id="0" precision="FP32">
					<dim>{n_synapses}</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="sensitivity" type="Const" version="opset1">
			<data element_type="f32" shape="{n_synapses}" offset="{off_sens}" size="{sensitivity.nbytes}"/>
			<output>
				<port id="0" precision="FP32">
					<dim>{n_synapses}</dim>
				</port>
			</output>
		</layer>
        <layer id="4" name="counts" type="Const" version="opset1">
			<data element_type="i32" shape="{output_dim}" offset="{off_counts}" size="{counts.nbytes}"/>
			<output>
				<port id="0" precision="I32">
					<dim>{output_dim}</dim>
				</port>
			</output>
		</layer>
        <layer id="5" name="starts" type="Const" version="opset1">
			<data element_type="i32" shape="{output_dim}" offset="{off_starts}" size="{starts.nbytes}"/>
			<output>
				<port id="0" precision="I32">
					<dim>{output_dim}</dim>
				</port>
			</output>
		</layer>
        <layer id="6" name="func_ids" type="Const" version="opset1">
			<data element_type="i64" shape="{func_ids.size}" offset="{off_func}" size="{func_ids.nbytes}"/>
			<output>
				<port id="0" precision="I64">
					<dim>{func_ids.size}</dim>
				</port>
			</output>
		</layer>
		<layer id="7" name="{name_prefix}" type="CompositeTSSN" version="extension">
			<data output_dim="{output_dim}" func_ids="0"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>{input_dim}</dim>
				</port>
				<port id="1">
					<dim>2</dim>
					<dim>{n_synapses}</dim>
				</port>
				<port id="2">
					<dim>{n_synapses}</dim>
				</port>
				<port id="3">
					<dim>{n_synapses}</dim>
				</port>
                <port id="4">
					<dim>{output_dim}</dim>
				</port>
                <port id="5">
					<dim>{output_dim}</dim>
				</port>
                <port id="6">
					<dim>{func_ids.size}</dim>
				</port>
			</input>
			<output>
				<port id="7" precision="FP32">
					<dim>1</dim>
					<dim>{output_dim}</dim>
				</port>
			</output>
		</layer>
		<layer id="8" name="result" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>{output_dim}</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="7" to-port="0"/>
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
    # 4. Read Mini-Model
    # Save XML to file to avoid bytes issue
    xml_path = cwd / f"temp_{safe_name}.xml"
    with open(xml_path, "w") as f:
        f.write(xml_str)
        
    mini_model = core.read_model(model=str(xml_path), weights=str(bin_path))
    
    # 5. Extract the TSSN Node
    # The node we want is the one with type CompositeTSSN
    tssn_op = None
    for op in mini_model.get_ops():
        if op.get_type_name() == "CompositeTSSN":
            tssn_op = op
            break
            
    if tssn_op is None:
        raise RuntimeError("Failed to create CompositeTSSN node from XML")
        
    return tssn_op, bin_path

try:
    core = ov.Core()
    if EXTENSION_PATH.exists():
        core.add_extension(str(EXTENSION_PATH))
        print(f"   ✅ Loaded Extension: {EXTENSION_PATH.name}")
    else:
        print(f"   ❌ Extension not found at {EXTENSION_PATH}")
        sys.exit(1)

    print("   📖 Reading Model...")
    model = core.read_model(GEMMA_IR_PATH)
    
    # --- Infection Logic ---
    model_ops = model.get_ops()
    replacements = []
    temp_files = []
    
    print("   🔍 Scanning for FFN layers (MatMul)...")
    
    for op in model_ops:
        if op.get_type_name() == "MatMul":
            # Check inputs
            # Input 0: Activation
            # Input 1: Weights (Constant)
            if len(op.inputs()) != 2:
                continue
                
            weights_node = op.input_value(1).get_node()
            if weights_node.get_type_name() == "Convert": # Handle FP16->FP32 convert
                weights_node = weights_node.input_value(0).get_node()
                
            if weights_node.get_type_name() != "Constant":
                continue
                
            # It's a candidate.
            # Check shapes
            # Activation: [Batch, ..., In]
            # Weights: [In, Out] or [Out, In]
            
            weights_data = weights_node.data
            transpose_b = op.get_attributes().get('transpose_b', False)
            
            if transpose_b:
                # Weights are [Out, In]
                # Perfect for our row-major iteration
                out_dim = weights_data.shape[0]
                in_dim = weights_data.shape[1]
                flat_weights = weights_data.flatten()
            else:
                # Weights are [In, Out]
                # Need to transpose to [Out, In]
                out_dim = weights_data.shape[1]
                in_dim = weights_data.shape[0]
                flat_weights = weights_data.T.flatten()
                
            # Heuristic: Only replace FFN layers (usually large)
            # Gemma FFNs are quite large.
            if in_dim < 128 or out_dim < 128:
                continue
                
            print(f"   💉 Injecting TSSN into {op.get_friendly_name()} ({in_dim}x{out_dim})...")
            
            # Create Replacement
            try:
                # Check original output type
                orig_type = op.get_output_element_type(0)
                print(f"      Original Type: {orig_type}")

                new_op, bin_file = create_tssn_node(core, op.input_value(0), flat_weights, out_dim, op.get_friendly_name() + "_tssn")
                temp_files.append(bin_file)
                
                # Connect Input 0 of new_op to Input 0 of old op
                source_output = op.input_value(0)
                new_op.input(0).replace_source_output(source_output)
                
                # Replace old op with new op
                op.output(0).replace(new_op.output(0))
                
                replacements.append(op.get_friendly_name())
                
            except Exception as e:
                print(f"      ⚠️ Failed to inject: {e}")
                import traceback
                traceback.print_exc()

    print(f"   ✅ Replaced {len(replacements)} layers.")
    
    print("   💾 Saving 'Infected' Model...")
    ov.save_model(model, INFECTED_IR_PATH)
    print("   ✅ Infection Complete.")
    
    # Cleanup
    for f in temp_files:
        try:
            os.remove(f)
        except:
            pass

except Exception as e:
    print(f"   ❌ Infection Failed: {e}")
    import traceback
    traceback.print_exc()
