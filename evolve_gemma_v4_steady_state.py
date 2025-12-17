"""
EVOLUTION V4: Steady-State GPU Saturation
-----------------------------------------
Optimized for MAXIMUM SUSTAINED GPU LOAD.
Uses a steady-state evolutionary algorithm with a circular buffer of async requests.
No 'wait_all()' barriers. As soon as one request finishes, a new mutation is spawned.
"""
import openvino as ov
import numpy as np
import time
import os
import json
import threading
from datetime import datetime
import queue
import memory_utils
import sys
from pathlib import Path

# --- Setup Environment (Must be before other imports/config) ---
cwd = Path.cwd()
release_bin = cwd / 'bin/intel64/Release'
tbb_bin = cwd / 'temp/Windows_AMD64/tbb/bin'
local_python_pkg = cwd / 'bin/intel64/Release/python'

if release_bin.exists() and tbb_bin.exists():
    # Prepend to PATH so DLLs are found
    os.environ['PATH'] = f"{release_bin.absolute()};{tbb_bin.absolute()};" + os.environ['PATH']
    os.environ['OPENVINO_LIB_PATHS'] = f"{release_bin.absolute()};{tbb_bin.absolute()}"
    
if local_python_pkg.exists():
    sys.path.insert(0, str(local_python_pkg.absolute()))

# --- Configuration ---
DENSE_MODEL_PATH = "gemma_ir/openvino_model.xml"
TSSN_MODEL_PATH = "gemma_ir_tssn/openvino_model.xml"
EXT_PATH = os.path.abspath("src/custom_ops/build/Release/openvino_tssn_extension.dll")
GPU_CONFIG = os.path.abspath("src/custom_ops/composite_tssn_gpu.xml")
DEVICE = "GPU"

# Evolution Hyperparameters
POPULATION_SIZE = 16      # Reduced queue depth for stability (was 64)
MAX_EVALUATIONS = 200000  # Total number of inferences to run
MUTATION_RATE = 0.05
CHECKPOINT_INTERVAL = 1000 # Save every N evaluations
LOG_FILE = "evolution_v4_gpu.log"
CHECKPOINT_DIR = "evolution_checkpoints_v4"

# Global State (Thread-Safe)
state_lock = threading.Lock()
global_best_mse = float('inf')
global_genome = {} # layer_name -> numpy array
global_genome_tensors = {} # layer_name -> ov.Tensor (Cache)
global_version = 0
global_change_log = [] # List of (version, layer_name) tuples
request_states = [] # List of dicts tracking request state

evaluations_count = 0
start_time = 0

def ternarize_weights(model):
    """
    Applies Balanced Ternary Quantization (BTQ) to TSSN layers.
    Ref: INTEL UHD 620 x TERNARY LOGIC REPORT
    """
    log_message("Applying Ternary Quantization to Weights...")
    ops = model.get_ops()
    count = 0
    for op in ops:
        if op.get_type_name() == "CompositeTSSN":
            # Input 2 is weights
            weights_input = op.input_value(2)
            weights_node = weights_input.get_node()
            
            # Handle Convert nodes (e.g. FP16 -> FP32)
            if weights_node.get_type_name() == "Convert":
                weights_input = weights_node.input_value(0)
                weights_node = weights_input.get_node()
            
            if weights_node.get_type_name() == "Constant":
                weights = weights_node.data.copy()
                
                # BTQ Logic
                threshold = 0.33 * np.abs(weights).max()
                ternary = np.zeros_like(weights)
                ternary[weights > threshold] = 1.0
                ternary[weights < -threshold] = -1.0
                
                # Update node
                # Create new constant with same type as original weights (likely FP16 or FP32)
                # Ensure 64-byte alignment for the new constant data
                aligned_ternary = memory_utils.ensure_aligned(ternary.astype(weights.dtype))
                new_const = ov.runtime.op.Constant(aligned_ternary)
                new_const.set_friendly_name(weights_node.get_friendly_name() + "_ternary")
                
                # Replace the original Constant's output consumers
                # This handles both direct connection and connection via Convert
                weights_node.output(0).replace(new_const.output(0))
                count += 1
            else:
                log_message(f"WARNING: TSSN Weights are not Constant! Found: {weights_node.get_type_name()} ({weights_node.get_friendly_name()})")
                
    log_message(f"Ternarized {count} weight tensors.")

def log_message(message, also_print=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)
    if also_print:
        print(f"[{timestamp}] {message}")

def get_inputs(seq_len=32):
    np.random.seed(42)
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
    global global_best_mse, global_genome, global_genome_tensors, global_version, request_states, evaluations_count, start_time
    
    log_message("="*80)
    log_message(f"EVOLUTION V4: Steady-State GPU Saturation (Queue: {POPULATION_SIZE})")
    log_message("="*80)
    
    # 1. Initialize OpenVINO
    core = ov.Core()
    log_message(f"Loading extension: {EXT_PATH}")
    core.add_extension(EXT_PATH)
    core.set_property("GPU", {"CONFIG_FILE": GPU_CONFIG})

    # System Audit (Report Section 3.2)
    log_message("Performing Ternary System Audit...")
    devices = core.available_devices
    log_message(f"Available Devices: {devices}")
    if "GPU" in devices:
        gpu_ops = core.get_property("GPU", "OPTIMIZATION_CAPABILITIES")
        log_message(f"GPU Capabilities: {gpu_ops}")
        if "FP16" in gpu_ops:
            log_message("[OK] Native FP16 Support Detected (Gen9+)")
        else:
            log_message("[WARN] FP16 Support Not Explicitly Advertised (Check Drivers)")
    
    # 2. Load Models
    teacher_model = core.read_model(DENSE_MODEL_PATH)
    student_model = core.read_model(TSSN_MODEL_PATH)
    
    # 3. Refactor Student
    ternarize_weights(student_model) # Apply Ternary Optimization
    
    tssn_ops = []
    layer_params = {} 
    
    log_message("Refactoring Graph...")
    for op in student_model.get_ops():
        if op.get_type_name() == "CompositeTSSN":
            tssn_ops.append(op)
            layer_name = op.get_friendly_name()
            
            func_ids_node = op.input_value(6).get_node()
            initial_data = func_ids_node.data.copy()
            global_genome[layer_name] = initial_data
            # Pre-create tensor cache (Zero-Copy Aligned)
            aligned_data = memory_utils.ensure_aligned(initial_data)
            global_genome_tensors[layer_name] = ov.Tensor(aligned_data)
            
            param_name = f"{layer_name}_func_ids"
            new_param = ov.runtime.op.Parameter(ov.Type.i32, ov.Shape(initial_data.shape))
            new_param.set_friendly_name(param_name)
            new_param.output(0).set_names({param_name})
            
            op.input(6).replace_source_output(new_param.output(0))
            student_model.add_parameters([new_param])
            layer_params[layer_name] = new_param
            
    log_message(f"Found {len(tssn_ops)} TSSN layers.")
    
    # Initialize request states
    request_states = [{'version': -1, 'mutated_layer': None} for _ in range(POPULATION_SIZE)]

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
    
    # Pre-create base input tensors (Aligned for Zero-Copy)
    student_base_tensors = {}
    for k, v in student_base_inputs.items():
        aligned_v = memory_utils.ensure_aligned(v)
        student_base_tensors[k] = ov.Tensor(aligned_v)

    # Align teacher inputs as well (optional but good practice)
    teacher_inputs_aligned = {}
    for k, v in teacher_inputs.items():
        aligned_v = memory_utils.ensure_aligned(v)
        teacher_inputs_aligned[k] = aligned_v # Keep as numpy for compiled_teacher call
    
    # 5. Compile & Baseline
    log_message("Compiling Teacher (CPU)...")
    compiled_teacher = core.compile_model(teacher_model, "CPU")
    teacher_output = compiled_teacher(teacher_inputs_aligned)[0]
    
    log_message("Compiling Student...")
    # Intel Optimization: Stage 369 Configuration for Gen9/UHD620
    # Based on INTEL_UHD620_STAGE2_ULTRA_TERNARY_ACCELERATION.md
    student_config = {
        "PERFORMANCE_HINT": "THROUGHPUT",
        "NUM_STREAMS": "1",                 # Reduced to 1 for stability
        "INFERENCE_PRECISION_HINT": "f16",
        "GPU_ENABLE_LOOP_UNROLLING": "NO",  # Disabled to reduce kernel size/compilation time
        "CACHE_DIR": "./model_cache",       # Stage 369: Faster startup
        # "GPU_NV12_TWO_INPUTS": "YES",     # Removed: Not supported by current plugin version
    }
    compiled_student = core.compile_model(student_model, DEVICE, student_config)
    infer_queue = ov.AsyncInferQueue(compiled_student, POPULATION_SIZE)
    
    # Baseline Run
    # Set initial inputs for Request 0
    for k, v in student_base_tensors.items():
        infer_queue[0].set_tensor(k, v)
    for lname, tensor in global_genome_tensors.items():
        param_name = layer_params[lname].get_friendly_name()
        infer_queue[0].set_tensor(param_name, tensor)
        
    infer_queue[0].infer() # Inputs already set
    baseline_output = infer_queue[0].get_output_tensor(0).data
    global_best_mse = np.mean((teacher_output - baseline_output)**2)
    log_message(f"Baseline MSE: {global_best_mse:.6e}")
    
    # 6. Steady-State Loop Setup
    
    # We need a way to signal completion
    stop_event = threading.Event()
    
    def process_result(request, userdata):
        global global_best_mse, evaluations_count, global_version
        
        if request.results is None: # Check for errors if possible
            pass

        # Unpack userdata: (request_index, layer_name, mutated_gene)
        request_index, layer_name, mutated_gene = userdata
        
        if stop_event.is_set():
            return

        # 1. Evaluate
        try:
            output = request.get_output_tensor(0).data
        except Exception as e:
            log_message(f"Inference Error: {e}")
            return

        mse = np.mean((teacher_output - output)**2)
        
        # 2. Update State (Critical Section)
        with state_lock:
            evaluations_count += 1
            current_count = evaluations_count
            
            if mse < global_best_mse:
                global_best_mse = mse
                global_genome[layer_name] = mutated_gene
                # Update Tensor Cache
                global_genome_tensors[layer_name] = ov.Tensor(mutated_gene)
                global_version += 1
                global_change_log.append((global_version, layer_name))
                # Keep log size manageable (optional, but good for long runs)
                if len(global_change_log) > 10000:
                    global_change_log.pop(0)
                
                log_message(f"[{current_count}] 🌟 NEW BEST! MSE: {mse:.6e} (Layer: {layer_name})")
            
            # Stats
            if current_count % 100 == 0:
                elapsed = time.time() - start_time
                mps = current_count / elapsed
                print(f"Eval {current_count}: Best MSE {global_best_mse:.6e} | Speed: {mps:.2f} evals/sec")
                
            if current_count >= MAX_EVALUATIONS:
                stop_event.set()
                return

        # 3. Schedule Next Job (Immediate Turnaround on SAME request)
        if not stop_event.is_set():
            schedule_next_job(request_index)

    def schedule_next_job(request_index):
        try:
            # 1. Pick Mutation Target
            target_op = tssn_ops[np.random.randint(0, len(tssn_ops))]
            target_layer_name = target_op.get_friendly_name()
            
            # Get current best gene
            with state_lock:
                base_gene = global_genome[target_layer_name] 
            
            # 2. Mutate
            mutated_gene = base_gene.copy()
            mask = np.random.rand(*mutated_gene.shape) < MUTATION_RATE
            if np.any(mask):
                # Monotone Bias: Prefer 1 (MIN) and 2 (MAX) and 5 (CONSENSUS)
                choices = [0, 1, 2, 3, 4, 5]
                probs = [0.1, 0.3, 0.3, 0.1, 0.1, 0.1]
                mutated_gene[mask] = np.random.choice(choices, size=np.sum(mask), p=probs)
                
            # 3. Prepare Inputs (Optimized Delta Update)
            request = infer_queue[request_index]
            req_state = request_states[request_index]
            
            # Check if we need to initialize base inputs (first run for this request)
            if req_state['version'] == -1:
                 for k, v in student_base_tensors.items():
                    request.set_tensor(k, v)

            # Check Global Version
            with state_lock:
                current_global_version = global_version # Snapshot
                change_log_snapshot = list(global_change_log) # Safe copy
            
            if current_global_version > req_state['version']:
                # Delta Update Protocol
                start_version = req_state['version']
                
                needs_full_update = True
                layers_to_update = set()
                
                if start_version != -1:
                    if len(change_log_snapshot) > 0:
                        oldest_version = change_log_snapshot[0][0]
                        if start_version >= oldest_version - 1:
                            needs_full_update = False
                            for ver, lname in change_log_snapshot:
                                if ver > start_version:
                                    layers_to_update.add(lname)
                
                if needs_full_update:
                    # Full Update
                    with state_lock:
                        for lname, tensor in global_genome_tensors.items():
                            param_name = layer_params[lname].get_friendly_name()
                            request.set_tensor(param_name, tensor)
                else:
                    # Delta Update
                    with state_lock:
                        for lname in layers_to_update:
                            param_name = layer_params[lname].get_friendly_name()
                            request.set_tensor(param_name, global_genome_tensors[lname])
                
                # Update state
                req_state['version'] = current_global_version
                req_state['mutated_layer'] = None 
                
            # Now apply the mutation
            if req_state['mutated_layer'] is not None and req_state['version'] == current_global_version:
                # Revert the previously mutated layer to its "Best" state
                prev_layer = req_state['mutated_layer']
                if prev_layer != target_layer_name:
                    prev_param = layer_params[prev_layer].get_friendly_name()
                    with state_lock:
                        request.set_tensor(prev_param, global_genome_tensors[prev_layer])
            
            # Apply NEW mutation
            target_param = layer_params[target_layer_name].get_friendly_name()
            mutated_tensor = ov.Tensor(mutated_gene)
            request.set_tensor(target_param, mutated_tensor)
            
            # Update Request State
            req_state['mutated_layer'] = target_layer_name
                
            # 4. Launch
            new_userdata = (request_index, target_layer_name, mutated_gene)
            infer_queue.userdata[request_index] = new_userdata
            
            request.start_async()
        except Exception as e:
            log_message(f"CRITICAL ERROR in schedule_next_job for request {request_index}: {e}")
            try:
                infer_queue[request_index].start_async()
            except:
                pass

    # 6.5 Benchmark Mode (Report Section 4.2)
    log_message("Running Throughput Benchmark (Warmup)...")
    # Warmup
    for _ in range(10):
        infer_queue.start_async()
    infer_queue.wait_all()
    
    log_message("Benchmarking...")
    bench_start = time.time()
    bench_iters = 100
    for i in range(bench_iters):
        infer_queue.start_async()
    infer_queue.wait_all()
    
    # Explicitly wait for each request to ensure they are free for the Kickstart loop
    infer_queue.wait_all()

    bench_duration = time.time() - bench_start
    bench_fps = bench_iters / bench_duration
    log_message(f"Benchmark Result: {bench_fps:.2f} FPS (Target: >40 FPS for MobileNetV2-like, Gemma is heavier)")
    
    # 7. Kickstart
    log_message("Igniting GPU Engine...")
    
    # Set the callback on the QUEUE
    infer_queue.set_callback(process_result)
    
    start_time = time.time()
    
    # Fill the queue manually
    for i in range(POPULATION_SIZE):
        schedule_next_job(i)
        
    # 8. Wait
    try:
        while not stop_event.is_set():
            time.sleep(1)
            
            # Periodic Checkpoint
            if evaluations_count > 0 and evaluations_count % CHECKPOINT_INTERVAL == 0:
                with state_lock:
                    param_dump = {k: v.tolist() for k, v in global_genome.items()}
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                with open(os.path.join(CHECKPOINT_DIR, f"genome_{evaluations_count}.json"), "w") as f:
                    json.dump(param_dump, f)
                    
    except KeyboardInterrupt:
        log_message("Evolution interrupted.")
        stop_event.set()
        
    infer_queue.wait_all()
    log_message(f"Final MSE: {global_best_mse:.6e}")

if __name__ == "__main__":
    main()
