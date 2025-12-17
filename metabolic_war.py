import openvino as ov
import numpy as np
import matplotlib.pyplot as plt
import json
import time

# --- TSSN Simulation Class ---
class TSSNLayer:
    def __init__(self, weights_shape, seed_indices, seed_values):
        self.shape = weights_shape
        # Initialize ternary weights from the seed (sign of original weights)
        # We use a sparse-like approach: only active at seed indices
        self.indices = seed_indices
        self.ternary_weights = np.sign(seed_values) # {-1, 0, 1}
        
        # Learnable sensitivity parameter (initialized to magnitude of original weights)
        # This allows the TSSN to start "close" to the original function
        self.sensitivity = np.abs(seed_values)
        
        # Momentum for optimization
        self.velocity = np.zeros_like(self.sensitivity)
        
    def forward(self, x):
        # x shape: [batch, input_dim]
        # We need to perform a sparse matrix multiplication
        # For simulation in Python, we can reconstruct the sparse matrix or iterate
        # Since we are targeting one layer, let's try a masked dense approach for simplicity/speed in NumPy
        
        # Construct effective weight matrix for this batch
        # This is slow for a real loop, but fine for a feasibility demo
        
        # Optimization: We only care about the dot product at the specific indices
        # But x is dense.
        # Let's do: y = x @ W_sparse
        
        # Create a zero matrix
        W_eff = np.zeros(self.shape)
        # Fill with current effective weights
        W_eff[self.indices] = self.ternary_weights * self.sensitivity
        
        return np.dot(x, W_eff)

    def update(self, x, error_grad, learning_rate=0.01):
        # error_grad: [batch, output_dim] (dL/dy)
        # x: [batch, input_dim]
        # We need dL/ds (gradient w.r.t sensitivity)
        
        # y = x @ (W_ternary * s)
        # dy/ds = x * W_ternary
        # dL/ds = dL/dy * dy/ds
        
        # This is effectively backprop through the sparse weights
        # For each active weight w_ij (at indices):
        # grad_s_ij = sum_over_batch( error_grad_j * x_i * w_ternary_ij )
        
        # Let's implement this efficiently
        # We need the gradients only at self.indices
        
        # Calculate full gradient matrix
        grad_W = np.dot(x.T, error_grad) # [input_dim, output_dim]
        
        # Extract gradients at seed indices
        grad_s = grad_W[self.indices] * self.ternary_weights
        
        # Apply update (SGD + Momentum)
        # We want to maximize correlation with error (Hebbian-like) to reduce it
        # If error is positive (target > output), and input*weight is positive, we want to INCREASE sensitivity.
        # grad_s is proportional to error * input * weight.
        # So we should ADD grad_s.
        self.velocity = 0.9 * self.velocity + learning_rate * grad_s
        self.sensitivity += self.velocity
        
        # Ensure sensitivity stays positive (magnitude)
        self.sensitivity = np.maximum(self.sensitivity, 0.0)

# --- Main Simulation ---
def run_metabolic_war():
    print("Initializing Phase 3: The First Infection...")
    
    # 1. Setup OpenVINO Probe
    core = ov.Core()
    model = core.read_model("model_ir/openvino_model.xml")
    
    # Target Layer 23 (as found by probe)
    target_layer_name = "__module.layers.23.mlp.down_proj/aten::linear/MatMul"
    input_node_name = "__module.layers.23.mlp/aten::mul/Multiply"
    
    print(f"Targeting Layer: {target_layer_name}")
    
    # Get the original weights
    target_op = None
    for op in model.get_ops():
        if op.get_friendly_name() == target_layer_name:
            target_op = op
            break
            
    if target_op is None:
        print("Error: Target layer not found.")
        return

    weights_node = target_op.input_value(1).get_node()
    W_orig = weights_node.get_data() # Shape [1152, 768] or similar
    
    # Add output probe
    input_op = None
    for op in model.get_ops():
        if op.get_friendly_name() == input_node_name:
            input_op = op
            break
            
    model.add_outputs(input_op.output(0))
    compiled_model = core.compile_model(model, "CPU")
    probe_output_key = compiled_model.outputs[-1]
    
    # 3. Data (Synthetic Stream for "C4" proxy)
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    except:
        print("Warning: Could not load Gemma tokenizer. Using bert-base-uncased as fallback.")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is evolving rapidly.",
        "OpenVINO enables efficient inference on the edge.",
        "The metabolic cost of intelligence must be minimized.",
        "Neuromorphic computing offers a path to sustainability.",
        "Project Cyberspore aims to infect and refactor LLMs.",
        "Matryoshka representation learning creates nested embeddings.",
        "Sparse ternary neurons reduce energy consumption.",
        "Homeostatic regulation replaces backpropagation.",
        "The future of AI is deterministic and emergent."
    ] * 10 # 100 steps

    # Run one inference to determine shapes
    print("Calibrating shapes...")
    dummy_input = tokenizer(sentences[0], return_tensors="np")
    input_ids = dummy_input["input_ids"]
    print(f"Input IDs Shape: {input_ids.shape}")
    
    print("Model Inputs:")
    for inp in compiled_model.inputs:
        print(f"  {inp.get_any_name()}: {inp.get_partial_shape()}")
        
    # Try to map inputs by name
    infer_request_input = {}
    for inp in compiled_model.inputs:
        name = inp.get_any_name()
        if "input_ids" in name:
            infer_request_input[inp] = input_ids
        elif "attention_mask" in name:
            infer_request_input[inp] = dummy_input["attention_mask"]
        elif "position_ids" in name:
            # Generate position ids if needed
            seq_len = input_ids.shape[1]
            infer_request_input[inp] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            
    # If mapping failed (e.g. names are generic), fallback to index 0
    if not infer_request_input:
        print("Warning: Could not map inputs by name. Using index 0 for input_ids.")
        infer_request_input = {compiled_model.inputs[0]: input_ids}

    results = compiled_model(infer_request_input)
    x_dummy = results[probe_output_key]
    x_dim = x_dummy.shape[-1]
    
    print(f"Input Dimension: {x_dim}")
    print(f"Weight Shape: {W_orig.shape}")
    
    # Align Weights
    if W_orig.shape[0] != x_dim:
        if W_orig.shape[1] == x_dim:
            print("Transposing weights to match input...")
            W_dense = W_orig.T
        else:
            print(f"Error: Weight shape {W_orig.shape} does not match input dim {x_dim}")
            return
    else:
        W_dense = W_orig

    # 2. Prepare Host and Parasite
    sparsity = 0.1
    threshold = np.percentile(np.abs(W_dense), sparsity * 100)
    mask = np.abs(W_dense) >= threshold
    
    W_host = W_dense * mask
    
    # Parasite Seeds (where mask is False)
    seed_indices = np.where(~mask)
    seed_values = W_dense[~mask]
    
    print(f"Infecting {len(seed_values)} synapses...")
    # Initialize with low sensitivity to demonstrate adaptation (learning from scratch-ish)
    # If we initialized with perfect values, error would be 0 immediately.
    parasite = TSSNLayer(W_dense.shape, seed_indices, seed_values)
    parasite.sensitivity = np.random.uniform(0.0, 0.01, size=parasite.sensitivity.shape)
    
    # 4. The Metabolic War Loop
    history_loss = []
    history_metabolic = []
    
    print("Beginning Metabolic War Simulation...")
    start_time = time.time()
    
    for i, text in enumerate(sentences):
        # Get input activations
        inputs = tokenizer(text, return_tensors="np")
        input_ids = inputs["input_ids"]
        
        # Prepare inputs robustly
        infer_request_input = {}
        for inp in compiled_model.inputs:
            name = inp.get_any_name()
            if "input_ids" in name:
                infer_request_input[inp] = input_ids
            elif "attention_mask" in name:
                infer_request_input[inp] = inputs["attention_mask"]
            elif "position_ids" in name:
                seq_len = input_ids.shape[1]
                infer_request_input[inp] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        
        if not infer_request_input:
             infer_request_input = {compiled_model.inputs[0]: input_ids}

        # Run Probe
        results = compiled_model(infer_request_input)
        x = results[probe_output_key] # [1, seq_len, 1152] (assuming)
        
        # Flatten batch/seq for simple training
        x_flat = x.reshape(-1, x.shape[-1])
        
        # Oracle (True Target)
        y_true = np.dot(x_flat, W_dense)
        
        # Host (Pruned)
        y_host = np.dot(x_flat, W_host)
        
        # Parasite (TSSN)
        y_parasite = parasite.forward(x_flat)
        
        # Chimera
        y_chimera = y_host + y_parasite
        
        # Error
        diff = y_true - y_chimera
        loss = np.mean(diff**2)
        
        # Update Parasite (Fast Loop)
        parasite.update(x_flat, diff, learning_rate=0.05)
        
        # Metabolic Cost (Simulated)
        metabolic_cost = 0.9 + (0.1 * np.mean(parasite.sensitivity > 0.001)) 
        
        history_loss.append(loss)
        history_metabolic.append(metabolic_cost)
        
        if i % 10 == 0:
            print(f"Step {i}: Func Error (MSE) = {loss:.6f}, Metabolic Cost = {metabolic_cost:.4f}")

    print(f"Simulation Complete. Final MSE: {loss:.8f}")
    
    # Save Results
    results = {
        "steps": list(range(len(history_loss))),
        "functional_error": [float(l) for l in history_loss],
        "metabolic_cost": [float(m) for m in history_metabolic]
    }
    
    with open("metabolic_war_results.json", "w") as f:
        json.dump(results, f)
        
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history_loss, label="Functional Error (MSE)")
    plt.xlabel("Adaptation Steps")
    plt.ylabel("MSE")
    plt.title("Phase 3: The Metabolic War - TSSN Adaptation")
    plt.legend()
    plt.grid(True)
    plt.savefig("metabolic_war_plot.png")
    print("Plot saved to metabolic_war_plot.png")

if __name__ == "__main__":
    run_metabolic_war()
