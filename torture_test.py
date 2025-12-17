import openvino as ov
import numpy as np
import time
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# --- TSSN Simulation Class (Reused) ---
class TSSNLayer:
    def __init__(self, weights_shape):
        self.shape = weights_shape
        self.indices = (np.array([], dtype=int), np.array([], dtype=int))
        self.ternary_weights = np.array([])
        self.sensitivity = np.array([])
        
    def expand(self, new_indices, new_values):
        self.indices = new_indices
        self.ternary_weights = np.sign(new_values)
        self.sensitivity = np.abs(new_values) 
            
    def forward(self, x):
        W_eff = np.zeros(self.shape)
        W_eff[self.indices] = self.ternary_weights * self.sensitivity
        return np.dot(x, W_eff)

def run_torture_test():
    print("Initializing Torture Test...")
    print("Objective: Validate PCN contribution and check for 'Lobotomy'.")
    
    # 1. Setup
    core = ov.Core()
    model = core.read_model("model_ir/openvino_model.xml")
    
    # Target Layer
    target_layer_name = "__module.layers.23.mlp.down_proj/aten::linear/MatMul"
    input_node_name = "__module.layers.23.mlp/aten::mul/Multiply"
    
    # Extract Weights
    target_op = None
    for op in model.get_ops():
        if op.get_friendly_name() == target_layer_name:
            target_op = op
            break
    
    weights_node = target_op.input_value(1).get_node()
    W_orig = weights_node.get_data()
    
    # Setup Probe
    input_op = None
    for op in model.get_ops():
        if op.get_friendly_name() == input_node_name:
            input_op = op
            break
    model.add_outputs(input_op.output(0))
    compiled_model = core.compile_model(model, "CPU")
    probe_output_key = compiled_model.outputs[-1]
    
    # Data Setup
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    except:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    # Calibrate Shapes
    dummy_input = tokenizer("test", return_tensors="np")
    input_ids = dummy_input["input_ids"]
    
    infer_request_input = {}
    for inp in compiled_model.inputs:
        name = inp.get_any_name()
        if "input_ids" in name: infer_request_input[inp] = input_ids
        elif "attention_mask" in name: infer_request_input[inp] = dummy_input["attention_mask"]
        elif "position_ids" in name: infer_request_input[inp] = np.arange(input_ids.shape[1], dtype=np.int64).reshape(1, -1)
    if not infer_request_input: infer_request_input = {compiled_model.inputs[0]: input_ids}
        
    results = compiled_model(infer_request_input)
    x_dummy = results[probe_output_key]
    x_dim = x_dummy.shape[-1]
    
    if W_orig.shape[0] != x_dim:
        if W_orig.shape[1] == x_dim: W_dense = W_orig.T
        else: return
    else: W_dense = W_orig

    # --- Prepare 99% Pruned State ---
    target_sparsity = 0.99
    print(f"\nPreparing State: {target_sparsity*100}% Sparsity")
    
    threshold = np.percentile(np.abs(W_dense), target_sparsity * 100)
    mask = np.abs(W_dense) >= threshold
    W_host = W_dense * mask
    
    # Initialize Parasite (PCN)
    parasite = TSSNLayer(W_dense.shape)
    current_indices = np.where(~mask)
    current_values = W_dense[~mask]
    parasite.expand(current_indices, current_values)
    
    print(f"Host Active Weights: {np.sum(mask)}")
    print(f"Parasite Active Weights: {len(current_values)}")
    
    # --- Test 1: The Placebo Control ---
    print("\n--- Test 1: The Placebo Control ---")
    print("Running inference with Host ONLY (No PCN)...")
    
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is evolving rapidly."
    ]
    
    placebo_errors = []
    chimera_errors = []
    
    for text in test_sentences:
        inputs = tokenizer(text, return_tensors="np")
        input_ids = inputs["input_ids"]
        
        req_input = {}
        for inp in compiled_model.inputs:
            name = inp.get_any_name()
            if "input_ids" in name: req_input[inp] = input_ids
            elif "attention_mask" in name: req_input[inp] = inputs["attention_mask"]
            elif "position_ids" in name: req_input[inp] = np.arange(input_ids.shape[1], dtype=np.int64).reshape(1, -1)
        if not req_input: req_input = {compiled_model.inputs[0]: input_ids}
        
        res = compiled_model(req_input)
        x = res[probe_output_key]
        x_flat = x.reshape(-1, x.shape[-1])
        
        y_true = np.dot(x_flat, W_dense)
        y_host = np.dot(x_flat, W_host) # Placebo
        y_parasite = parasite.forward(x_flat)
        y_chimera = y_host + y_parasite # Chimera
        
        placebo_mse = np.mean((y_true - y_host)**2)
        chimera_mse = np.mean((y_true - y_chimera)**2)
        
        placebo_errors.append(placebo_mse)
        chimera_errors.append(chimera_mse)
        
    avg_placebo_mse = np.mean(placebo_errors)
    avg_chimera_mse = np.mean(chimera_errors)
    
    print(f"Placebo MSE (Host Only): {avg_placebo_mse:.8f}")
    print(f"Chimera MSE (Host + PCN): {avg_chimera_mse:.8f}")
    
    if avg_placebo_mse < 1e-5:
        print("RESULT: FAIL. The Host is doing all the work. PCN is redundant.")
    else:
        print("RESULT: PASS. The Host failed without the PCN.")

    # --- Test 2: The Adversarial Probe (Resolution Stress Test) ---
    print("\n--- Test 2: The Adversarial Probe ---")
    print("Checking for loss of fine-grained detail...")
    
    adversarial_pairs = [
        ("The reaction was exothermic.", "The reaction was endothermic."),
        ("He is a good man.", "He is a bad man."),
        ("The stock market went up.", "The stock market went down.")
    ]
    
    for s1, s2 in adversarial_pairs:
        # Get embeddings (using the layer output as proxy for embedding)
        # Note: This is just one layer's output, not the full model embedding, 
        # but if this layer collapses, the downstream embedding will likely be affected.
        
        def get_layer_output(text, mode="oracle"):
            inputs = tokenizer(text, return_tensors="np")
            input_ids = inputs["input_ids"]
            req_input = {}
            for inp in compiled_model.inputs:
                name = inp.get_any_name()
                if "input_ids" in name: req_input[inp] = input_ids
                elif "attention_mask" in name: req_input[inp] = inputs["attention_mask"]
                elif "position_ids" in name: req_input[inp] = np.arange(input_ids.shape[1], dtype=np.int64).reshape(1, -1)
            if not req_input: req_input = {compiled_model.inputs[0]: input_ids}
            
            res = compiled_model(req_input)
            x = res[probe_output_key]
            x_flat = x.reshape(-1, x.shape[-1])
            
            if mode == "oracle":
                return np.dot(x_flat, W_dense)
            elif mode == "host":
                return np.dot(x_flat, W_host)
            elif mode == "chimera":
                return np.dot(x_flat, W_host) + parasite.forward(x_flat)
        
        # We compare the mean vector of the sequence for simplicity
        v1_oracle = np.mean(get_layer_output(s1, "oracle"), axis=0).reshape(1, -1)
        v2_oracle = np.mean(get_layer_output(s2, "oracle"), axis=0).reshape(1, -1)
        
        v1_host = np.mean(get_layer_output(s1, "host"), axis=0).reshape(1, -1)
        v2_host = np.mean(get_layer_output(s2, "host"), axis=0).reshape(1, -1)
        
        sim_oracle = cosine_similarity(v1_oracle, v2_oracle)[0][0]
        sim_host = cosine_similarity(v1_host, v2_host)[0][0]
        
        print(f"Pair: '{s1}' vs '{s2}'")
        print(f"  Oracle Similarity: {sim_oracle:.4f}")
        print(f"  Host Only Similarity: {sim_host:.4f}")
        
        if abs(sim_host - sim_oracle) < 0.01:
             print("  Status: PRESERVED (Host retained nuance)")
        else:
             print("  Status: DEGRADED (Host lost nuance)")

    # --- Test 3: Latency Check ---
    print("\n--- Test 3: Latency Check (Simulation) ---")
    # We can't easily measure true sparse speedup in Python, but we can measure the overhead
    # of the Python-side PCN to see if it's "expensive" in this simulation.
    
    start = time.time()
    for _ in range(100):
        np.dot(x_flat, W_host)
    host_time = time.time() - start
    
    start = time.time()
    for _ in range(100):
        parasite.forward(x_flat)
    parasite_time = time.time() - start
    
    print(f"100 Runs - Host (Dense Masked): {host_time:.4f}s")
    print(f"100 Runs - Parasite (Simulated Sparse): {parasite_time:.4f}s")
    print("Note: Parasite time is high because it's Python. C++ Kernel required for speedup.")

if __name__ == "__main__":
    run_torture_test()
