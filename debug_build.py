import openvino as ov
import os
import sys
import numpy as np

def log(msg):
    print(f"[DEBUG] {msg}")

def ternarize_weights(model):
    """
    Applies Balanced Ternary Quantization (BTQ) to TSSN layers.
    Ref: INTEL UHD 620 x TERNARY LOGIC REPORT
    """
    log("Applying Ternary Quantization to Weights...")
    ops = model.get_ops()
    count = 0
    for op in ops:
        if op.get_type_name() == "CompositeTSSN":
            # Input 2 is weights
            weights_input = op.input_value(2)
            weights_node = weights_input.get_node()
            if weights_node.get_type_name() == "Constant":
                weights = weights_node.data.copy()
                
                # BTQ Logic
                threshold = 0.33 * np.abs(weights).max()
                ternary = np.zeros_like(weights)
                ternary[weights > threshold] = 1.0
                ternary[weights < -threshold] = -1.0
                
                # Update node
                new_const = ov.runtime.op.Constant(ternary.astype(np.float32))
                new_const.set_friendly_name(weights_node.get_friendly_name() + "_ternary")
                op.input(2).replace_source_output(new_const.output(0))
                count += 1
    log(f"Ternarized {count} weight tensors.")

def main():
    try:
        core = ov.Core()
        ext_path = os.path.abspath("src/custom_ops/build/Release/openvino_tssn_extension.dll")
        xml_path = os.path.abspath("src/custom_ops/composite_tssn_gpu.xml")
        
        log(f"Loading extension: {ext_path}")
        core.add_extension(ext_path)
        
        log(f"Setting GPU config: {xml_path}")
        core.set_property("GPU", {"CONFIG_FILE": xml_path})
        
        model_path = "gemma_ir_tssn/openvino_model.xml"
        log(f"Reading model: {model_path}")
        model = core.read_model(model_path)
        
        ternarize_weights(model)
        
        log("Refactoring Graph (adding parameters)...")
        for op in model.get_ops():
            if op.get_type_name() == "CompositeTSSN":
                # Input 6 is function_ids
                func_ids_node = op.input_value(6).get_node()
                if func_ids_node.get_type_name() == "Constant":
                    initial_data = func_ids_node.data.copy()
                    
                    param_name = f"{op.get_friendly_name()}_func_ids"
                    new_param = ov.runtime.op.Parameter(ov.Type.i32, ov.Shape(initial_data.shape))
                    new_param.set_friendly_name(param_name)
                    new_param.output(0).set_names({param_name})
                    
                    op.input(6).replace_source_output(new_param.output(0))
                    model.add_parameters([new_param])
        
        student_config = {
            "PERFORMANCE_HINT": "THROUGHPUT",
            "NUM_STREAMS": "4",
            "INFERENCE_PRECISION_HINT": "f16",
            "GPU_ENABLE_LOOP_UNROLLING": "YES",
            "CACHE_DIR": "./model_cache",
        }
        
        log("Compiling model with config...")
        compiled_model = core.compile_model(model, "GPU", student_config)
        log("Compilation successful!")
        
    except Exception as e:
        log(f"Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
