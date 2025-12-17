import openvino as ov
import numpy as np
from transformers import AutoTokenizer
import os

def validate():
    model_path = "gemma_ir_tssn/openvino_model.xml"
    ext_path = r"src/custom_ops/build/Release/openvino_tssn_extension.dll"
    gpu_xml_path = r"src/custom_ops/composite_tssn_gpu.xml"
    
    print(f"Loading extension from {ext_path}...")
    core = ov.Core()
    core.add_extension(ext_path)
    
    # Load GPU Custom Layer Config
    if os.path.exists(gpu_xml_path):
        print(f"Loading GPU config from {gpu_xml_path}...")
        core.set_property("GPU", {"CONFIG_FILE": gpu_xml_path})
    else:
        print(f"Warning: GPU config not found at {gpu_xml_path}")
    
    print(f"Loading model from {model_path}...")
    model = core.read_model(model_path)
    
    # Compile for GPU
    print("Compiling model for GPU...")
    compiled_model = core.compile_model(model, "GPU")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gemma_ir_tssn")
    
    input_text = "The future of AI is"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Convert to numpy for OpenVINO
    ov_inputs = {k: v.numpy() for k, v in inputs.items()}
    
    # Check inputs and add missing ones
    for input in model.inputs:
        name = input.any_name
        print(f"Model expects input: {name}, Shape: {input.partial_shape}")
        if name not in ov_inputs:
            if "beam_idx" in name:
                # beam_idx usually has shape [Batch]
                batch_size = ov_inputs["input_ids"].shape[0]
                ov_inputs[name] = np.zeros((batch_size,), dtype=np.int32)
            elif "position_ids" in name:
                # Create position ids
                seq_len = ov_inputs["input_ids"].shape[1]
                ov_inputs[name] = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
    
    print("Running inference...")
    request = compiled_model.create_infer_request()
    results = request.infer(ov_inputs)
    
    # Get logits
    logits = results[compiled_model.output(0)]
    print(f"Logits shape: {logits.shape}")
    
    # Simple greedy decode
    next_token_id = np.argmax(logits[:, -1, :], axis=-1)
    print(f"Next token ID: {next_token_id}")
    print(f"Next token: {tokenizer.decode(next_token_id)}")
    
    print("Validation successful!")

if __name__ == "__main__":
    validate()
