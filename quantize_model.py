import os
import sys
import numpy as np

# Add OpenVINO Python module path explicitly if needed
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
import nncf
from transformers import AutoTokenizer

def quantize(model_path, output_path):
    print(f"Loading model from {model_path}...")
    core = ov.Core()
    model = core.read_model(model_path)
    
    print("Preparing calibration dataset...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gemma_ir_tssn")
    except:
        print("Could not load tokenizer from gemma_ir_tssn, trying 'bert-base-uncased' as fallback or skipping.")
        # If we can't load the specific tokenizer, we might have issues. 
        # But let's assume it works as per validate_gemma_tssn.py
        raise

    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "OpenVINO optimizes deep learning models for inference.",
        "Quantization reduces model size and improves performance.",
        "Sparsity can lead to significant speedups on supported hardware.",
        "The future of AI is bright and full of possibilities.",
        "Machine learning models are becoming increasingly complex.",
        "Efficiency is key for deploying models on edge devices.",
        "Neural networks are inspired by the human brain.",
        "Deep learning has revolutionized computer vision and NLP."
    ] * 10
    
    def transform_fn(data_item):
        # Ensure inputs match model expectations
        inputs = tokenizer(data_item, return_tensors="np", padding="max_length", truncation=True, max_length=128)
        
        # Convert to dict and ensure types
        model_inputs = {k: v for k, v in inputs.items()}
        
        # Add missing inputs
        for input in model.inputs:
            name = input.any_name
            if name not in model_inputs:
                if "beam_idx" in name:
                    batch_size = model_inputs["input_ids"].shape[0]
                    model_inputs[name] = np.zeros((batch_size,), dtype=np.int32)
                elif "position_ids" in name:
                    seq_len = model_inputs["input_ids"].shape[1]
                    # Create position ids [1, seq_len] or [batch, seq_len]
                    # Usually [1, seq_len] is broadcastable, but let's match batch size to be safe
                    batch_size = model_inputs["input_ids"].shape[0]
                    model_inputs[name] = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len).repeat(batch_size, axis=0)
        
        return model_inputs

    calibration_dataset = nncf.Dataset(sentences, transform_fn)
    
    print("Quantizing model (INT8)...")
    # preset=nncf.QuantizationPreset.PERFORMANCE is default
    quantized_model = nncf.quantize(model, calibration_dataset)
    
    print(f"Saving quantized model to {output_path}...")
    ov.save_model(quantized_model, output_path)

if __name__ == "__main__":
    input_model = "model_ir/openvino_model.xml"
    output_dir = "model_ir_int8"
    output_model = f"{output_dir}/openvino_model.xml"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    quantize(input_model, output_model)
