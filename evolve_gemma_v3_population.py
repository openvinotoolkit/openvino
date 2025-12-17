"""
EVOLUTION V3: Population-Based GPU Acceleration
-----------------------------------------------
Optimized for "Sustained" GPU Load.
Uses OpenVINO AsyncInferQueue to evaluate a population of mutations in parallel.
"""
import openvino as ov
import numpy as np
import time
import os
import json
import copy
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
DENSE_MODEL_PATH = "gemma_ir/openvino_model.xml"
TSSN_MODEL_PATH = "gemma_ir_tssn/openvino_model.xml"
EXT_PATH = os.path.abspath("src/custom_ops/build/Release/openvino_tssn_extension.dll")
GPU_CONFIG = os.path.abspath("src/custom_ops/composite_tssn_gpu.xml")
DEVICE = "GPU"

# Evolution Hyperparameters
POPULATION_SIZE = 32      # Number of parallel mutations to evaluate
MAX_GENERATIONS = 5000    # Total generations (Total evaluations = MAX_GEN * POP_SIZE)
MUTATION_RATE = 0.05      # 5% of neurons per mutation (Lower rate for fine-tuning)
CHECKPOINT_INTERVAL = 5   # Save every N generations
LOG_FILE = "evolution_v3_gpu.log"
CHECKPOINT_DIR = "evolution_checkpoints_v3"
RESULTS_FILE = "evolution_results_v3.json"

def log_message(message, also_print=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)
    if also_print:
        print(f"[{timestamp}] {message}")

def get_inputs(seq_len=32):
    """Generate consistent random test inputs."""
    np.random.seed(42) # Fixed seed for consistent baseline comparison
    input_ids = np.random.randint(0, 256000, (1, seq_len), dtype=np.int64)
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
    beam_idx = np.zeros((1,), dtype=np.int32)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "beam_idx": beam_idx
    }

def main():
    log_message("="*80)
    log_message(f"EVOLUTION V3: Population-Based GPU Acceleration (Pop Size: {POPULATION_SIZE})")
    log_message("="*80)
    
    # 1. Initialize OpenVINO
    core = ov.Core()
    log_message(f"Loading extension: {EXT_PATH}")
    core.add_extension(EXT_PATH)
    core.set_property("GPU", {"CONFIG_FILE": GPU_CONFIG})
    
    # 2. Load Models
    log_message("Loading Teacher (Dense)...")
    teacher_model = core.read_model(DENSE_MODEL_PATH)
    
    log_message("Loading Student (TSSN)...")
    student_model = core.read_model(TSSN_MODEL_PATH)
    
    # 3. Refactor Student for Dynamic Parameters
    tssn_ops = []
    layer_params = {} # layer_name -> Parameter Node
    current_genome = {} # layer_name -> numpy array (The "DNA")
    
    log_message("Refactoring Graph for Dynamic Execution...")
    for op in student_model.get_ops():
        if op.get_type_name() == "CompositeTSSN":
            tssn_ops.append(op)
            layer_name = op.get_friendly_name()
            
            # Get Function IDs Constant (Input 6)
            func_ids_input = op.input_value(6)
            func_ids_node = func_ids_input.get_node()
            
            # Extract initial data
            initial_data = func_ids_node.data.copy()
            current_genome[layer_name] = initial_data
            
            # Create Parameter
            param_name = f"{layer_name}_func_ids"
            new_param = ov.runtime.op.Parameter(ov.Type.i32, ov.Shape(initial_data.shape))
            new_param.set_friendly_name(param_name)
            new_param.output(0).set_names({param_name})
            
            # Replace Constant with Parameter
            op.input(6).replace_source_output(new_param.output(0))
            student_model.add_parameters([new_param])
            layer_params[layer_name] = new_param
            
    log_message(f"Found {len(tssn_ops)} TSSN layers. Genome extracted.")
    
    # 4. Prepare Inputs
    inputs = get_inputs()
    
    def get_safe_input_names(model):
        names = []
        for i in model.inputs:
            try:
                names.append(i.any_name)
            except RuntimeError:
                names.append(i.node.friendly_name)
        return names

    teacher_input_names = get_safe_input_names(teacher_model)
    student_input_names = get_safe_input_names(student_model)

    teacher_inputs = {k: v for k, v in inputs.items() if k in teacher_input_names}
    student_base_inputs = {k: v for k, v in inputs.items() if k in student_input_names}
    
    # 5. Compile Models
    log_message("Compiling Teacher...")
    compiled_teacher = core.compile_model(teacher_model, DEVICE)
    teacher_output = compiled_teacher(teacher_inputs)[0] # Ground Truth
    
    log_message(f"Compiling Student (Async Queue Size: {POPULATION_SIZE})...")
    compiled_student = core.compile_model(student_model, DEVICE)
    
    # Create Async Queue
    infer_queue = ov.AsyncInferQueue(compiled_student, POPULATION_SIZE)
    
    # 6. Baseline Evaluation
    # Construct full input map for baseline
    full_student_inputs = student_base_inputs.copy()
    for layer_name, data in current_genome.items():
        full_student_inputs[layer_params[layer_name]] = data
        
    # Run single inference for baseline
    request = infer_queue[0]
    request.infer(full_student_inputs)
    student_output = request.get_output_tensor(0).data
    
    best_mse = np.mean((teacher_output - student_output)**2)
    log_message(f"Baseline MSE: {best_mse:.6e}")
    
    # 7. Evolution Loop
    results_history = {"baseline_mse": float(best_mse), "generations": []}
    
    # Pre-allocate memory for population genomes to avoid allocation overhead
    # List of dictionaries, where each dict is a full genome candidate
    # Actually, we only mutate one layer at a time usually, but for population we might want diversity.
    # Strategy: In each generation, pick ONE layer to mutate, but generate POP_SIZE variations of it.
    
    start_time = time.time()
    
    try:
        for generation in range(1, MAX_GENERATIONS + 1):
            gen_start = time.time()
            
            # A. Select Target Layer for this Generation
            target_op = tssn_ops[np.random.randint(0, len(tssn_ops))]
            target_layer_name = target_op.get_friendly_name()
            target_param = layer_params[target_layer_name]
            
            base_gene = current_genome[target_layer_name]
            
            # B. Prepare Population Inputs
            # We need to set up the queue. 
            # All requests need the base inputs + the static parts of the genome + the mutated part.
            # Optimization: Only update the changed tensor.
            
            # Callback to capture results
            gen_results = [None] * POPULATION_SIZE
            
            def completion_callback(request, userdata):
                idx = userdata
                output = request.get_output_tensor(0).data
                mse = np.mean((teacher_output - output)**2)
                gen_results[idx] = mse

            infer_queue.set_callback(completion_callback)
            
            # Dispatch Jobs
            mutations_data = [] # Store the mutated arrays to retrieve the winner later
            
            for i in range(POPULATION_SIZE):
                # 1. Create Mutation
                mutated_gene = base_gene.copy()
                mask = np.random.rand(*mutated_gene.shape) < MUTATION_RATE
                if np.any(mask):
                    mutated_gene[mask] = np.random.randint(0, 5, size=np.sum(mask))
                
                mutations_data.append(mutated_gene)
                
                # 2. Set Inputs
                # We need to ensure the request has all inputs.
                # Ideally, we set the static inputs once.
                # But AsyncInferQueue reuses requests, so we must be careful.
                
                # Set Base Inputs (if not already set - simplistic check)
                # For safety in this loop, we set everything. 
                # (Optimization: Set base inputs once per request object outside loop)
                
                # Note: This loop might be the CPU bottleneck now.
                for k, v in student_base_inputs.items():
                    infer_queue[i].set_tensor(k, ov.Tensor(v))
                
                # Set Genome (Static parts)
                # This is expensive to copy every time. 
                # TODO: In V4, optimize this by only updating the changed layer.
                # For now, we iterate all layers.
                for lname, data in current_genome.items():
                    if lname == target_layer_name:
                        infer_queue[i].set_tensor(layer_params[lname], ov.Tensor(mutated_gene))
                    else:
                        infer_queue[i].set_tensor(layer_params[lname], ov.Tensor(data))
                
                # 3. Start Async
                infer_queue.start_async(userdata=i)
            
            # C. Wait for all
            infer_queue.wait_all()
            
            # D. Selection
            # Find best in batch
            min_mse = min(gen_results)
            best_idx = gen_results.index(min_mse)
            
            gen_time = time.time() - gen_start
            
            # E. Update Global State
            improved = False
            if min_mse < best_mse:
                best_mse = min_mse
                current_genome[target_layer_name] = mutations_data[best_idx]
                improved = True
                log_message(f"[{generation}] 🌟 NEW BEST! MSE: {best_mse:.6e} (Layer: {target_layer_name})")
            
            # Stats
            mps = POPULATION_SIZE / gen_time
            if generation % 10 == 0:
                print(f"Gen {generation}: Best MSE {best_mse:.6e} | Speed: {mps:.2f} mutations/sec | Last Time: {gen_time:.3f}s")
            
            # F. Checkpoint
            if generation % CHECKPOINT_INTERVAL == 0:
                # Save genome
                param_dump = {k: v.tolist() for k, v in current_genome.items()}
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                with open(os.path.join(CHECKPOINT_DIR, f"genome_gen_{generation}.json"), "w") as f:
                    json.dump(param_dump, f)

    except KeyboardInterrupt:
        log_message("Evolution interrupted.")
    
    total_time = time.time() - start_time
    log_message(f"Total Time: {total_time:.2f}s")
    log_message(f"Final MSE: {best_mse:.6e}")

if __name__ == "__main__":
    main()
