import os
import sys
import numpy as np
import time

# Add OpenVINO Python module path explicitly
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
from openvino.runtime import Core, Type, Tensor
from openvino.runtime import opset10 as ops
from transformers import AutoTokenizer

def create_parasite_model(batch, seq_len, hidden_dim):
    xml = f"""
    <net name="Parasite_TSSN" version="10">
        <layers>
            <layer id="0" name="Input_X" type="Parameter" version="opset1">
                <data shape="{batch},{seq_len},{hidden_dim}" element_type="i8"/>
                <output>
                    <port id="0" precision="I8" names="x">
                        <dim>{batch}</dim>
                        <dim>{seq_len}</dim>
                        <dim>{hidden_dim}</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="Input_H" type="Parameter" version="opset1">
                <data shape="{batch},{hidden_dim}" element_type="i8"/>
                <output>
                    <port id="0" precision="I8" names="h">
                        <dim>{batch}</dim>
                        <dim>{hidden_dim}</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="Weight_A" type="Parameter" version="opset1">
                <data shape="{hidden_dim}" element_type="i8"/>
                <output>
                    <port id="0" precision="I8"/>
                </output>
            </layer>
            <layer id="3" name="Weight_B" type="Parameter" version="opset1">
                <data shape="{hidden_dim}" element_type="i8"/>
                <output>
                    <port id="0" precision="I8"/>
                </output>
            </layer>
            <layer id="4" name="Weight_C" type="Parameter" version="opset1">
                <data shape="{hidden_dim}" element_type="i8"/>
                <output>
                    <port id="0" precision="I8"/>
                </output>
            </layer>
            <layer id="5" name="TSSN_Node" type="TSSN" version="extension">
                <input>
                    <port id="0">
                        <dim>{batch}</dim>
                        <dim>{seq_len}</dim>
                        <dim>{hidden_dim}</dim>
                    </port>
                    <port id="1">
                        <dim>{batch}</dim>
                        <dim>{hidden_dim}</dim>
                    </port>
                    <port id="2">
                        <dim>{hidden_dim}</dim>
                    </port>
                    <port id="3">
                        <dim>{hidden_dim}</dim>
                    </port>
                    <port id="4">
                        <dim>{hidden_dim}</dim>
                    </port>
                </input>
                <output>
                    <port id="0" precision="I8">
                        <dim>{batch}</dim>
                        <dim>{seq_len}</dim>
                        <dim>{hidden_dim}</dim>
                    </port>
                    <port id="1" precision="I8">
                        <dim>{batch}</dim>
                        <dim>{hidden_dim}</dim>
                    </port>
                </output>
            </layer>
            <layer id="6" name="Result_Y" type="Result" version="opset1">
                <input>
                    <port id="0">
                        <dim>{batch}</dim>
                        <dim>{seq_len}</dim>
                        <dim>{hidden_dim}</dim>
                    </port>
                </input>
            </layer>
            <layer id="7" name="Result_H" type="Result" version="opset1">
                <input>
                    <port id="0">
                        <dim>{batch}</dim>
                        <dim>{hidden_dim}</dim>
                    </port>
                </input>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="5" to-port="0"/>
            <edge from-layer="1" from-port="0" to-layer="5" to-port="1"/>
            <edge from-layer="2" from-port="0" to-layer="5" to-port="2"/>
            <edge from-layer="3" from-port="0" to-layer="5" to-port="3"/>
            <edge from-layer="4" from-port="0" to-layer="5" to-port="4"/>
            <edge from-layer="5" from-port="0" to-layer="6" to-port="0"/>
            <edge from-layer="5" from-port="1" to-layer="7" to-port="0"/>
        </edges>
    </net>
    """
    return xml

def run_healing_test(sparsity_level):
    print(f"\n=== Phase 5: Carrying Capacity Test (Sparsity: {int(sparsity_level*100)}%) ===")
    
    core = ov.Core()
    
    # Load Extension
    extension_path = r"src\custom_ops\build\Release\openvino_tssn_extension.dll"
    if os.path.exists(extension_path):
        core.add_extension(extension_path)
    else:
        print("Error: C++ Extension not found!")
        return

    # Load Pruned Model
    model_path = f"model_ir_pruned/openvino_model_magnitude_{int(sparsity_level*100)}.xml"
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return
        
    print(f"Loading pruned model from {model_path}...")
    model = core.read_model(model_path)
    
    # Load Original Model (Oracle)
    print("Loading Oracle (Original Model)...")
    model_orig = core.read_model("model_ir/openvino_model.xml")
    
    # Target Layer
    target_layer_name = "__module.layers.23.mlp.down_proj/aten::linear/MatMul"
    input_node_name = "__module.layers.23.mlp/aten::mul/Multiply"
    
    # Setup Probe on Pruned Model
    input_op = None
    for op in model.get_ops():
        if op.get_friendly_name() == input_node_name:
            input_op = op
            break
    model.add_outputs(input_op.output(0))
    
    # Enable Sparse Weights Decompression
    # This ensures the CPU skips zero computations
    # config = {"SPARSE_WEIGHTS_DECOMPRESSION_RATE": sparsity_level}
    # print(f"Compiling Host Model with SPARSE_WEIGHTS_DECOMPRESSION_RATE={sparsity_level}...")
    # try:
    #     compiled_model = core.compile_model(model, "CPU", config)
    # except RuntimeError as e:
    #     print(f"Warning: Failed to set SPARSE_WEIGHTS_DECOMPRESSION_RATE: {e}")
    #     print("Falling back to default compilation...")
    compiled_model = core.compile_model(model, "CPU")
    
    # Setup Probe on Oracle
    input_op_orig = None
    for op in model_orig.get_ops():
        if op.get_friendly_name() == input_node_name:
            input_op_orig = op
            break
    model_orig.add_outputs(input_op_orig.output(0))
    compiled_model_orig = core.compile_model(model_orig, "CPU")
    
    # Generate Input Data
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b") # Using gemma-2b tokenizer as proxy
    text = "The quick brown fox jumps over the lazy dog. " * 5
    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    seq_len = input_ids.shape[1]
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
    
    print(f"Input IDs shape: {input_ids.shape}")
    
    # Inspect Model Inputs
    print("Model Inputs:")
    for input in model.inputs:
        print(f"  {input.get_any_name()}: {input.get_partial_shape()}")

    # Run Host Inference
    infer_inputs = {
        "input_ids": input_ids, 
        "attention_mask": attention_mask,
        "position_ids": position_ids
    }
    
    try:
        res = compiled_model(infer_inputs)
    except Exception as e:
        print(f"Host Inference failed: {e}")
        return None
        
    Y_host_pruned = res[compiled_model.output(0)]
    X_host = res[compiled_model.output(1)] # The probe output
    
    # Run Oracle Inference
    try:
        res_orig = compiled_model_orig(infer_inputs)
    except Exception as e:
        print(f"Oracle Inference failed: {e}")
        return None
        
    Y_oracle = res_orig[compiled_model_orig.output(0)]
    
    # Calculate Baseline Error
    mse_baseline = np.mean((Y_oracle - Y_host_pruned)**2)
    print(f"Baseline Damage (MSE): {mse_baseline:.8f}")
    
    # --- Parasite Initialization ---
    batch_size = X_host.shape[0]
    seq_len = X_host.shape[1]
    parasite_dim = X_host.shape[-1]
    target_dim = Y_oracle.shape[-1]
    
    parasite_xml = create_parasite_model(batch_size, seq_len, parasite_dim)
    
    # Zero Initialization
    A = np.zeros((parasite_dim,), dtype=np.int8)
    B = np.zeros((parasite_dim,), dtype=np.int8)
    C = np.zeros((parasite_dim,), dtype=np.int8)
    H = np.zeros((batch_size, parasite_dim), dtype=np.int8)
    
    parasite_model = core.read_model(model=parasite_xml.encode(), weights=bytes())
    compiled_parasite = core.compile_model(parasite_model, "CPU")
    
    # --- Healing Loop ---
    epochs = 2000 # Reduced for speed in loop
    best_mse = mse_baseline
    start_time = time.time()
    
    print("Starting Healing Loop...")
    for epoch in range(epochs):
        # Mutate
        A_mut = A.copy()
        B_mut = B.copy()
        C_mut = C.copy()
        
        mutation_rate = 0.01
        mask_A = np.random.rand(*A.shape) < mutation_rate
        A_mut[mask_A] = np.random.choice([-1, 0, 1], size=np.count_nonzero(mask_A))
        
        mask_B = np.random.rand(*B.shape) < mutation_rate
        B_mut[mask_B] = np.random.choice([-1, 0, 1], size=np.count_nonzero(mask_B))
        
        mask_C = np.random.rand(*C.shape) < mutation_rate
        C_mut[mask_C] = np.random.choice([-1, 0, 1], size=np.count_nonzero(mask_C))
        
        # Quantize Input
        scale = 127.0 / (np.max(np.abs(X_host)) + 1e-9)
        X_quant = (X_host * scale).astype(np.int8)
        
        # Run Parasite
        res_p = compiled_parasite([X_quant, H, A_mut, B_mut, C_mut])
        Y_parasite_quant = res_p[compiled_parasite.output(0)]
        Y_parasite = (Y_parasite_quant.astype(np.float32) / scale)[:, :, :target_dim]
        
        # Chimera Output
        Y_chimera = Y_host_pruned + Y_parasite
        
        # Error
        mse = np.mean((Y_oracle - Y_chimera)**2)
        
        if mse < best_mse:
            best_mse = mse
            A, B, C = A_mut, B_mut, C_mut
            # print(f"Epoch {epoch}: Improved MSE: {mse:.9f}")
            
    end_time = time.time()
    recovery_pct = (mse_baseline - best_mse) / mse_baseline * 100 if mse_baseline > 0 else 0
    print(f"Final MSE: {best_mse:.8f}")
    print(f"Recovery: {recovery_pct:.2f}%")
    print(f"Time: {end_time - start_time:.2f}s")
    
    return mse_baseline, best_mse, recovery_pct

if __name__ == "__main__":
    sparsities = [0.6, 0.7, 0.8, 0.9]
    results = []
    
    for s in sparsities:
        res = run_healing_test(s)
        if res:
            results.append((s, *res))
            
    print("\n=== Summary Report ===")
    print("Sparsity | Baseline MSE | Final MSE | Recovery %")
    for r in results:
        print(f"{r[0]*100:.0f}%      | {r[1]:.8f}     | {r[2]:.8f}  | {r[3]:.2f}%")
