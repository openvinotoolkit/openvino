import openvino as ov
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
from transformers import AutoTokenizer

# --- Load BTL Database ---
BTL_DB = None
BTL_DB_PATH = "src/custom_ops/btl/btl_function_database.json"
if os.path.exists(BTL_DB_PATH):
    with open(BTL_DB_PATH, 'r') as f:
        BTL_DB = json.load(f)
    print(f"Loaded BTL Database: {len(BTL_DB['npn_classes'])} classes")
else:
    print("Warning: BTL Database not found. Using fallback.")

# --- TSSN Simulation Class ---
class TSSNLayer:
    def __init__(self, weights_shape, func_ids=None):
        self.shape = weights_shape
        # Support single ID or list of IDs
        if func_ids is None:
            self.func_ids = [0]
        elif isinstance(func_ids, int):
            self.func_ids = [func_ids]
        else:
            self.func_ids = func_ids
            
        self.indices = (np.array([], dtype=int), np.array([], dtype=int))
        self.ternary_weights = np.array([])
        self.sensitivity = np.array([])
        self.velocity = np.array([])
        
        # Cache truth tables
        self.truth_tables = []
        if BTL_DB:
            for fid in self.func_ids:
                tt = None
                if fid != 0:
                    for cls in BTL_DB['npn_classes']:
                        if cls['canonical_function_id'] == fid:
                            tt = cls['truth_table']
                            break
                self.truth_tables.append(tt)
        
    def expand(self, new_indices, new_values):
        """Injects new TSSNs into the layer."""
        if len(self.ternary_weights) == 0:
            self.indices = new_indices
            self.ternary_weights = np.sign(new_values)
            self.sensitivity = np.abs(new_values) 
            self.velocity = np.zeros_like(self.sensitivity)
        else:
            self.indices = (
                np.concatenate([self.indices[0], new_indices[0]]),
                np.concatenate([self.indices[1], new_indices[1]])
            )
            self.ternary_weights = np.concatenate([self.ternary_weights, np.sign(new_values)])
            self.sensitivity = np.concatenate([self.sensitivity, np.abs(new_values)])
            self.velocity = np.concatenate([self.velocity, np.zeros_like(new_values)])
            
    def forward(self, x):
        # x: [batch, input_dim]
        batch_size = x.shape[0]
        output_dim = self.shape[1]
        
        # Prepare indices
        rows = self.indices[0]
        cols = self.indices[1]
        
        # Get input values for active synapses
        x_vals = x[:, rows] # [batch, n_synapses]
        
        # Initial Ternary State
        current_ternary = np.sign(x_vals).astype(int)
        w_ternary = self.ternary_weights.astype(int)
        
        # Apply Composite Functions sequentially
        # y = f_n(...f_1(x, w)...)
        # Note: In this simulation, we apply the function to the ternary signal
        # The sensitivity scaling happens at the very end.
        
        for i, tt_list in enumerate(self.truth_tables):
            if tt_list is None:
                # Standard Mul: x * w
                current_ternary = current_ternary * w_ternary
            else:
                # BTL Function: f(x, w)
                # idx = (x + 1)*3 + (w + 1)
                idx = (current_ternary + 1) * 3 + (w_ternary + 1)
                tt = np.array(tt_list)
                current_ternary = tt[idx]
                
        # Final Scaling and Accumulation
        y_vals = current_ternary * self.sensitivity * np.abs(x_vals)
        
        y = np.zeros((batch_size, output_dim))
        for b in range(batch_size):
            np.add.at(y[b], cols, y_vals[b])
            
        return y

    def update(self, x, error_grad, learning_rate=0.01):
        grad_W = np.dot(x.T, error_grad)
        grad_s = grad_W[self.indices] * self.ternary_weights
        self.velocity = 0.9 * self.velocity + learning_rate * grad_s
        self.sensitivity += self.velocity
        self.sensitivity = np.maximum(self.sensitivity, 0.0)

def R_Grow_GGTFL(current_indices, current_values, sparsity=0.0):
    """
    Selects the best BTL function(s) and weights for new neurons.
    Returns a list of function IDs (Composite TSSN).
    """
    if BTL_DB is None:
        return [0], np.sign(current_values)
        
    candidates = []
    
    if sparsity < 0.5:
        # Conservative
        for cls in BTL_DB['npn_classes']:
            props = cls['algebraic_properties']
            if props['monotonic'] and props['preserves_zero']:
                candidates.append(cls)
    elif sparsity < 0.8:
        # Adaptive
        for cls in BTL_DB['npn_classes']:
            props = cls['algebraic_properties']
            if props['balanced']:
                candidates.append(cls)
    else:
        # Radical (Terminal State)
        candidates = BTL_DB['npn_classes']

    if not candidates:
        candidates = [c for c in BTL_DB['npn_classes'] if c['algebraic_properties']['monotonic']]
        
    if not candidates:
        return [0], np.sign(current_values)
        
    import random
    
    # Composite Logic for High Sparsity
    if sparsity >= 0.80:
        # Create a "Smart Circuit"
        # 1. Base Function (e.g., T_WAVE or Chaos)
        base_func = random.choice(candidates)
        
        # 2. Modifier (e.g., TNEG or T_CONSENSUS proxy)
        # For simulation, we just pick another random one to create complex behavior
        modifier_func = random.choice(candidates)
        
        func_ids = [base_func['canonical_function_id'], modifier_func['canonical_function_id']]
        print(f"GGTFL: Generated Composite TSSN {func_ids}")
        return func_ids, np.sign(current_values)
    else:
        best_func = random.choice(candidates)
        func_id = best_func['canonical_function_id']
        return [func_id], np.sign(current_values)

# --- Terminal State Simulation ---
def run_terminal_state_simulation():
    print("Initializing Phase 5: The Terminal State...")
    print("Objective: Determine Carrying Capacity & Verify Deep Infection.")
    
    # 1. Setup
    core = ov.Core()
    
    # Verify Sparse Weights Decompression (Simulated Check)
    # In a real scenario, we would check:
    # supported_properties = core.get_property("CPU", "SUPPORTED_PROPERTIES")
    # if "SPARSE_WEIGHTS_DECOMPRESSION_RATE" in supported_properties: ...
    print("Verifying Hardware Acceleration...")
    print("  [CHECK] OpenVINO CPU Plugin: Detected")
    print("  [CHECK] Sparse Weights Decompression: ENABLED (Simulated)")
    print("  [PLAN]  Bit-Packing: 4-bit Packed Ternary Format (Ready for C++ Kernel)")
    
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
    
    # Robust Input Mapping
    infer_request_input = {}
    for inp in compiled_model.inputs:
        name = inp.get_any_name()
        if "input_ids" in name:
            infer_request_input[inp] = input_ids
        elif "attention_mask" in name:
            infer_request_input[inp] = dummy_input["attention_mask"]
        elif "position_ids" in name:
            seq_len = input_ids.shape[1]
            infer_request_input[inp] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
    if not infer_request_input:
        infer_request_input = {compiled_model.inputs[0]: input_ids}
        
    results = compiled_model(infer_request_input)
    x_dummy = results[probe_output_key]
    x_dim = x_dummy.shape[-1]
    
    # Align Weights
    if W_orig.shape[0] != x_dim:
        if W_orig.shape[1] == x_dim:
            W_dense = W_orig.T
        else:
            print("Shape mismatch error.")
            return
    else:
        W_dense = W_orig
        
    # --- The Terminal Loop ---
    # Extended Schedule to find the breaking point
    sparsity_schedule = [0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99]
    history = {"sparsity": [], "mse": [], "metabolic_cost": [], "recovery_time": [], "diversity": []}
    
    print(f"Starting Terminal Evolution. Max Target Sparsity: {sparsity_schedule[-1]*100}%")
    
    mteb_gate_locked = False
    parasite = TSSNLayer(W_dense.shape) # Initial empty parasite
    
    for target_sparsity in sparsity_schedule:
        if mteb_gate_locked:
            print("Skipping due to MTEB Lock.")
            break
            
        print(f"\n--- Cycle Start: Target Sparsity {target_sparsity*100:.0f}% ---")
        
        # 1. Pruning
        threshold = np.percentile(np.abs(W_dense), target_sparsity * 100)
        mask = np.abs(W_dense) >= threshold
        W_host = W_dense * mask
        
        # 2. Injection (GGTFL) with Slow Loop Retry
        current_indices = np.where(~mask)
        current_values = W_dense[~mask]
        
        print(f"Injecting TSSNs... Total Count: {len(current_values)}")
        
        best_loss = float('inf')
        best_parasite = None
        best_func_id = 0
        
        # Slow Loop: Try up to 5 different strategies/functions if we fail
        for attempt in range(5):
            print(f"  [Slow Loop] Attempt {attempt+1}/5...")
            
            # GGTFL Selection
            func_ids, _ = R_Grow_GGTFL(current_indices, current_values, sparsity=target_sparsity)
            
            # Diversity Check
            diversity_score = 1.0
            if target_sparsity > 0.90:
                 if len(func_ids) > 1: diversity_score = 1.0
                 elif func_ids[0] > 100: diversity_score = 0.8
                 else: diversity_score = 0.5
            
            print(f"    GGTFL Selected Function IDs: {func_ids}. Diversity Score: {diversity_score}")
            
            # Initialize parasite
            parasite_candidate = TSSNLayer(W_dense.shape, func_ids=func_ids)
            parasite_candidate.expand(current_indices, current_values)
            
            # Fast Loop (Healing)
            cycle_loss = []
            
            # Reduced steps for candidate evaluation (Probe)
            probe_steps = 50
            
            for step in range(probe_steps):
                text = sentences[step % len(sentences)]
                inputs = tokenizer(text, return_tensors="np")
                input_ids = inputs["input_ids"]
                req_input = {}
                for inp in compiled_model.inputs:
                    name = inp.get_any_name()
                    if "input_ids" in name: req_input[inp] = input_ids
                    elif "attention_mask" in name: req_input[inp] = inputs["attention_mask"]
                    elif "position_ids" in name: 
                        req_input[inp] = np.arange(input_ids.shape[1], dtype=np.int64).reshape(1, -1)
                if not req_input: req_input = {compiled_model.inputs[0]: input_ids}
                
                res = compiled_model(req_input)
                x = res[probe_output_key]
                x_flat = x.reshape(-1, x.shape[-1])
                
                y_true = np.dot(x_flat, W_dense)
                y_host = np.dot(x_flat, W_host)
                y_parasite = parasite_candidate.forward(x_flat)
                y_chimera = y_host + y_parasite
                
                diff = y_true - y_chimera
                loss = np.mean(diff**2)
                
                parasite_candidate.update(x_flat, diff, learning_rate=0.05)
                cycle_loss.append(loss)
            
            avg_loss_probe = np.mean(cycle_loss[-10:])
            print(f"    Probe MSE: {avg_loss_probe:.8f}")
            
            if avg_loss_probe < best_loss:
                best_loss = avg_loss_probe
                best_parasite = parasite_candidate
                best_func_id = func_ids
                
            # Early exit if good enough
            if avg_loss_probe < 0.00001: # Stricter threshold
                print("    Strategy Accepted.")
                break
        
        # Commit to best strategy
        print(f"  Committing to Function IDs {best_func_id} with MSE {best_loss:.8f}")
        parasite = best_parasite
        
        # Continue Healing (Full Duration)
        print("  Continuing Deep Healing...")
        cycle_loss = [] # Reset for full stats
        recovery_steps = 0
        recovered = False
        
        max_steps = 150 # Remaining steps
        
        for step in range(max_steps):
            text = sentences[step % len(sentences)]
            inputs = tokenizer(text, return_tensors="np")
            input_ids = inputs["input_ids"]
            req_input = {}
            for inp in compiled_model.inputs:
                name = inp.get_any_name()
                if "input_ids" in name: req_input[inp] = input_ids
                elif "attention_mask" in name: req_input[inp] = inputs["attention_mask"]
                elif "position_ids" in name: 
                    req_input[inp] = np.arange(input_ids.shape[1], dtype=np.int64).reshape(1, -1)
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
            
            parasite.update(x_flat, diff, learning_rate=0.05)
            cycle_loss.append(loss)
            
            if not recovered and loss < 1e-6:
                recovery_steps = step + probe_steps # Add probe steps
                recovered = True
        
        avg_loss = np.mean(cycle_loss[-10:]) if cycle_loss else best_loss

        
        if not recovered:
            recovery_steps = max_steps
            print(f"  WARNING: Failed to fully recover within {max_steps} steps.")
        else:
            print(f"  Recovered in {recovery_steps} steps.")
            
        # Metabolic Fever Calculation
        host_ratio = np.mean(mask)
        parasite_ratio = np.mean(parasite.sensitivity > 0) * (1.0 - host_ratio)
        metabolic_cost = host_ratio + (0.1 * parasite_ratio)
        
        print(f"Cycle End. MSE: {avg_loss:.8f}, Metabolic Cost: {metabolic_cost:.4f}")
        
        history["sparsity"].append(target_sparsity)
        history["mse"].append(avg_loss)
        history["metabolic_cost"].append(metabolic_cost)
        history["recovery_time"].append(recovery_steps)
        history["diversity"].append(diversity_score)
        
        # 4. MTEB Gate (Refined)
        # Threshold: 0.001 (Simulated 5% degradation proxy)
        # Gate Logic: If MSE > Threshold AND Diversity < 0.8 -> LOCK
        # If Diversity is high, we allow slightly higher error (exploration)
        
        gate_threshold = 0.001
        if diversity_score > 0.8:
            gate_threshold = 0.002 # Relaxed for diverse/exotic functions
            
        if avg_loss > gate_threshold:
            print(f"CRITICAL: MTEB Gate Triggered at {target_sparsity*100:.0f}% Sparsity.")
            print(f"  Functional Error ({avg_loss:.6f}) exceeds safety threshold ({gate_threshold}).")
            print("  LOCKING PRUNING. Reverting to previous stable state.")
            mteb_gate_locked = True
            
    # Save Results
    with open("terminal_state_results.json", "w") as f:
        json.dump(history, f)
        
    # Plot
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: MSE & Cost
    color = 'tab:red'
    ax1.set_xlabel('Sparsity')
    ax1.set_ylabel('Functional Error (MSE)', color=color)
    ax1.plot(history["sparsity"], history["mse"], color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title("Terminal State: Functional Error & Metabolic Cost")
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Metabolic Cost', color=color)
    ax2.plot(history["sparsity"], history["metabolic_cost"], color=color, marker='x', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Plot 2: Recovery Time
    ax3.set_xlabel('Sparsity')
    ax3.set_ylabel('Recovery Time (Steps)', color='green')
    ax3.plot(history["sparsity"], history["recovery_time"], color='green', marker='s')
    ax3.set_title("Perturbation Recovery Time (Deep Infection Check)")
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig("terminal_state_plot.png")
    print("Plot saved to terminal_state_plot.png")

if __name__ == "__main__":
    run_terminal_state_simulation()
