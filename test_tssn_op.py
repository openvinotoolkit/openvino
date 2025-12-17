import openvino as ov
import numpy as np
import os
import sys
from pathlib import Path

# Setup paths
cwd = Path.cwd()
release_bin = cwd / 'bin/intel64/Release'
tbb_bin = cwd / 'temp/Windows_AMD64/tbb/bin'
local_python_pkg = cwd / 'bin/intel64/Release/python'

if release_bin.exists() and tbb_bin.exists():
    os.environ['OPENVINO_LIB_PATHS'] = f"{release_bin.absolute()};{tbb_bin.absolute()}"
    
if local_python_pkg.exists():
    sys.path.insert(0, str(local_python_pkg.absolute()))

def test_tssn():
    print("--- Starting TSSN Op Test ---")
    core = ov.Core()
    
    ext_path = cwd / "src/custom_ops/build/Release/openvino_tssn_extension.dll"
    if not ext_path.exists():
        print(f"Error: Extension not found at {ext_path}")
        return
        
    print(f"Loading extension: {ext_path}")
    core.add_extension(str(ext_path))
    
    # Create a simple graph
    # Input: [1, 16]
    input_dim = 16
    output_dim = 8
    n_synapses = input_dim * output_dim # Dense
    
    input_node = ov.runtime.op.Parameter(ov.Type.f32, ov.Shape([1, input_dim]))
    
    # Indices (Identity for dense)
    # [2, N]
    indices_data = np.zeros((2, n_synapses), dtype=np.int32)
    for i in range(input_dim):
        for j in range(output_dim):
            idx = i * output_dim + j
            indices_data[0, idx] = i # Input index
            indices_data[1, idx] = j # Output index
            
    indices_const = ov.runtime.op.Constant(indices_data)
    
    # Weights (Ternary)
    weights_data = np.random.choice([-1.0, 0.0, 1.0], size=(n_synapses,)).astype(np.float32)
    weights_const = ov.runtime.op.Constant(weights_data)
    
    # Sensitivity
    sens_data = np.ones((n_synapses,), dtype=np.float32)
    sens_const = ov.runtime.op.Constant(sens_data)
    
    # Create Node
    # We need to use the registered op name. 
    # Since we don't have python bindings for the op class constructor easily available without the python module,
    # we might need to use core.read_model if we had an XML.
    # But wait, we can't easily create a custom op from python without the python binding class.
    # However, we can try to use `ov.Extension` to register it and then maybe `core.create_op`? No, that's not standard.
    
    # Alternative: Create a dummy XML string and read it.
    
    xml_str = f"""<?xml version="1.0"?>
<net name="TSSN_Test" version="10">
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
			<data element_type="i32" shape="2,{n_synapses}" offset="0" size="{indices_data.nbytes}"/>
			<output>
				<port id="0" precision="I32">
					<dim>2</dim>
					<dim>{n_synapses}</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="weights" type="Const" version="opset1">
			<data element_type="f32" shape="{n_synapses}" offset="{indices_data.nbytes}" size="{weights_data.nbytes}"/>
			<output>
				<port id="0" precision="FP32">
					<dim>{n_synapses}</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="sensitivity" type="Const" version="opset1">
			<data element_type="f32" shape="{n_synapses}" offset="{indices_data.nbytes + weights_data.nbytes}" size="{sens_data.nbytes}"/>
			<output>
				<port id="0" precision="FP32">
					<dim>{n_synapses}</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="tssn" type="CompositeTSSN" version="extension">
			<data output_dim="{output_dim}" func_ids=""/>
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
				<port id="0">
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
    # Write binary data
    bin_path = cwd / "test_tssn.bin"
    with open(bin_path, "wb") as f:
        f.write(indices_data.tobytes())
        f.write(weights_data.tobytes())
        f.write(sens_data.tobytes())
        
    # Write XML
    xml_path = cwd / "test_tssn.xml"
    with open(xml_path, "w") as f:
        f.write(xml_str)
        
    print("Created test model files.")
    
    # Read Model
    print("Reading model...")
    model = core.read_model(xml_path)
    
    # Compile Model (CPU)
    print("Compiling model (CPU)...")
    compiled_model = core.compile_model(model, "CPU")
    
    # Infer
    print("Running inference...")
    input_data = np.random.randn(1, input_dim).astype(np.float32)
    res = compiled_model([input_data])[0]
    print("Inference successful!")
    print("Output shape:", res.shape)
    
    # Check if kernel file exists
    kernel_path = cwd / "composite_tssn_kernel.cl"
    if kernel_path.exists():
        print("✅ composite_tssn_kernel.cl exists.")
        # Check timestamp or content if needed
    else:
        print("❌ composite_tssn_kernel.cl does NOT exist.")

if __name__ == "__main__":
    test_tssn()
