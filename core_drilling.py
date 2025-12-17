import openvino as ov
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from transformers import AutoTokenizer

# --- TSSN Simulation Class ---
class TSSNLayer:
    def __init__(self, weights_shape):
        self.shape = weights_shape
        self.indices = (np.array([], dtype=int), np.array([], dtype=int))
        self.ternary_weights = np.array([])
        self.sensitivity = np.array([])
        self.velocity = np.array([])
        
    def expand(self, new_indices, new_values):
        """Injects new TSSNs into the layer."""
        self.indices = new_indices
        self.ternary_weights = np.sign(new_values)
        # Initialize sensitivity with original magnitude (Head Start)
        self.sensitivity = np.abs(new_values) 
        self.velocity = np.zeros_like(self.sensitivity)
            
    def forward(self, x):
        W_eff = np.zeros(self.shape)
        W_eff[self.indices] = self.ternary_weights * self.sensitivity
        return np.dot(x, W_eff)

    def update(self, x, error_grad, learning_rate=0.01):
        grad_W = np.dot(x.T, error_grad)
        grad_s = grad_W[self.indices] * self.ternary_weights
        
        # Momentum SGD
        self.velocity = 0.9 * self.velocity + learning_rate * grad_s
        self.sensitivity += self.velocity
        self.sensitivity = np.maximum(self.sensitivity, 0.0)

def run_core_drilling():
    print("Initializing Protocol: Core Drilling...")
    print("Objective: Attack the 'Inner Core' to force PCN adaptation.")
    
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
    ]
    
    # Calibrate Shapes
    dummy_input = tokenizer(sentences[0], return_tensors="np")
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

    # --- Attack Strategy: Inverse Magnitude Pruning ---
    # We want to remove the TOP 20% of weights (The Core).
    # This leaves the bottom 80% (The Noise).
    
    target_pruning_rate = 0.20
    print(f"\n--- Attack Phase: Removing Top {target_pruning_rate*100}% Weights ---")
    
    # Calculate threshold for top 20%
    # We want to KEEP weights where abs(w) < percentile(80)
    keep_threshold = np.percentile(np.abs(W_dense), (1.0 - target_pruning_rate) * 100)
    
    mask = np.abs(W_dense) < keep_threshold
    W_host = W_dense * mask
    
    print(f"Original Weights: {W_dense.size}")
    print(f"Host Active Weights (Bottom 80%): {np.sum(mask)}")
    print(f"Pruned Weights (Top 20%): {W_dense.size - np.sum(mask)}")
    
    # --- Baseline Damage Assessment ---
    print("\nAssessing Damage (Host Only)...")
    damage_errors = []
    
    for text in sentences:
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
        y_host = np.dot(x_flat, W_host)
        
        mse = np.mean((y_true - y_host)**2)
        damage_errors.append(mse)
        
    avg_damage = np.mean(damage_errors)
    print(f"Baseline Damage (MSE): {avg_damage:.6f}")
    
    if avg_damage < 0.001:
        print("WARNING: Damage insufficient. Even the top 20% might be redundant?")
    else:
        print("SUCCESS: Significant functional deficit created.")

    # --- Infection & Healing ---
    print("\n--- Infection Phase ---")
    parasite = TSSNLayer(W_dense.shape)
    
    # Inject into the voids (The Top 20%)
    current_indices = np.where(~mask)
    current_values = W_dense[~mask]
    
    print(f"Injecting {len(current_values)} TSSNs into the Core...")
    
    # Initialize with "Amnesia" to force learning?
    # If we initialize with perfect values, we just prove we can copy.
    # Let's initialize with random noise to prove *learning*.
    # Or initialize with 0.1 * magnitude to simulate a "weak" seed.
    
    parasite.indices = current_indices
    parasite.ternary_weights = np.sign(current_values)
    # "Weak Seed" Initialization: 10% of original sensitivity
    parasite.sensitivity = np.abs(current_values) * 0.1 
    parasite.velocity = np.zeros_like(parasite.sensitivity)
    
    print("TSSNs initialized with 10% sensitivity (Weak Seed).")
    
    print("\n--- Healing Phase (Metabolic War) ---")
    history_loss = []
    
    for epoch in range(5):
        epoch_loss = []
        for text in sentences:
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
            y_host = np.dot(x_flat, W_host)
            y_parasite = parasite.forward(x_flat)
            y_chimera = y_host + y_parasite
            
            diff = y_true - y_chimera
            loss = np.mean(diff**2)
            
            # Update
            parasite.update(x_flat, diff, learning_rate=0.1) # Aggressive learning
            epoch_loss.append(loss)
            
        avg_epoch_loss = np.mean(epoch_loss)
        print(f"Epoch {epoch+1}: MSE = {avg_epoch_loss:.6f}")
        history_loss.append(avg_epoch_loss)
        
    print(f"\nFinal MSE: {history_loss[-1]:.6f}")
    print(f"Recovery: {(1 - history_loss[-1]/avg_damage)*100:.2f}% of function restored.")
    
    # Plot
    plt.figure()
    plt.plot(history_loss, marker='o')
    plt.title("Core Drilling Recovery")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.savefig("core_drilling_plot.png")
    print("Plot saved to core_drilling_plot.png")

if __name__ == "__main__":
    run_core_drilling()
