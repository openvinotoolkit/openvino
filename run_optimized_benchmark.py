import sys
import os
import time
import numpy as np
import argparse

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

def run_benchmark(model_path, device="CPU", batch_size=1, duration=10):
    print(f"Benchmarking {model_path} on {device} with Batch Size {batch_size}...")
    
    try:
        core = ov.Core()
        # Add extension if needed (for TSSN)
        ext_path = r"src/custom_ops/build/Release/openvino_tssn_extension.dll"
        if os.path.exists(ext_path):
            # print(f"Loading extension: {ext_path}")
            core.add_extension(ext_path)
            
        # Load GPU Custom Layer Config
        if "GPU" in device:
            config_path = os.path.abspath(r"src/custom_ops/composite_tssn_gpu.xml")
            if os.path.exists(config_path):
                print(f"Loading GPU Custom Layer Config: {config_path}")
                # Use the correct property key for OpenVINO 2.0+
                # It's usually "cldnn_config" or "CONFIG_FILE" depending on version
                # Let's try generic set_property
                try:
                    core.set_property("GPU", {"CONFIG_FILE": config_path})
                except Exception as e:
                    print(f"Failed to set CONFIG_FILE: {e}")
            else:
                print(f"Warning: GPU Config not found at {config_path}")
            
        model = core.read_model(model_path)
        
        # Reshape model for batch size
        new_shapes = {}
        for input in model.inputs:
            shape = input.partial_shape
            if shape.rank.is_static:
                new_shape = []
                for i, dim in enumerate(shape):
                    if i == 0:
                        new_shape.append(batch_size)
                    elif dim.is_static:
                        new_shape.append(dim.get_length())
                    else:
                        # Assume dim 1 is seq len
                        if i == 1:
                            new_shape.append(128)
                        else:
                            new_shape.append(1)
                new_shapes[input.any_name] = new_shape
        
        if new_shapes:
            # print(f"Reshaping model to: {new_shapes}")
            model.reshape(new_shapes)
            
        compiled_model = core.compile_model(model, device)
        
        # Create input data
        inputs = {}
        for input in compiled_model.inputs:
            # shape should be static now
            shape = input.shape
            # Fill with random data or zeros
            if input.element_type == ov.Type.i64:
                data = np.zeros(shape, dtype=np.int64)
            elif input.element_type == ov.Type.i32:
                data = np.zeros(shape, dtype=np.int32)
            elif input.element_type == ov.Type.boolean:
                data = np.zeros(shape, dtype=bool)
            else:
                data = np.random.rand(*shape).astype(np.float32)
            inputs[input.any_name] = data
            
        request = compiled_model.create_infer_request()
        
        # Warmup
        # print("Warmup...")
        for _ in range(5):
            request.infer(inputs)
            
        # Benchmark
        # print(f"Running for {duration} seconds...")
        start_time = time.time()
        num_iters = 0
        while time.time() - start_time < duration:
            request.infer(inputs)
            num_iters += 1
            
        total_time = time.time() - start_time
        fps = (num_iters * batch_size) / total_time
        
        print(f"Result: {fps:.2f} FPS")
        return fps
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        # import traceback
        # traceback.print_exc()
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="CPU", help="Device to run on (CPU, GPU)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    results = {}
    
    # 1. Baseline
    if os.path.exists("model_ir/openvino_model.xml"):
        fps = run_benchmark("model_ir/openvino_model.xml", args.device, args.batch_size)
        results["Baseline (FP32/FP16)"] = fps
    else:
        print("Baseline model not found at model_ir/openvino_model.xml")
        
    # 2. Quantized (INT8)
    if os.path.exists("model_ir_int8/openvino_model.xml"):
        fps = run_benchmark("model_ir_int8/openvino_model.xml", args.device, args.batch_size)
        results["Quantized (INT8)"] = fps
    else:
        print("Quantized model not found. Run quantize_model.py first.")
        
    # 3. Block Sparse
    if os.path.exists("model_ir_pruned/openvino_model_block_70.xml"):
        fps = run_benchmark("model_ir_pruned/openvino_model_block_70.xml", args.device, args.batch_size)
        results["Block Sparse (70%)"] = fps
    else:
        print("Block sparse model not found. Run apply_incision.py first.")

    # 4. TSSN (Custom Op)
    if os.path.exists("gemma_ir_tssn/openvino_model.xml"):
        fps = run_benchmark("gemma_ir_tssn/openvino_model.xml", args.device, args.batch_size)
        results["TSSN (Custom Op)"] = fps
    else:
        print("TSSN model not found at gemma_ir_tssn/openvino_model.xml")

    print("\n--- Summary ---")
    print(f"Device: {args.device}, Batch Size: {args.batch_size}")
    for name, fps in results.items():
        print(f"{name}: {fps:.2f} FPS")
