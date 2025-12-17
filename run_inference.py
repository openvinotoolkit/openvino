import openvino as ov
import numpy as np
from transformers import AutoTokenizer

# Paths
model_path = "model_ir/openvino_model.xml"
tokenizer_path = "./embeddinggemma_local"

print(f"Loading model from {model_path}...")
core = ov.Core()
compiled_model = core.compile_model(model_path, "CPU")
infer_request = compiled_model.create_infer_request()

print(f"Loading tokenizer from {tokenizer_path}...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Input text
text = "OpenVINO is a toolkit for optimizing and deploying AI inference."
print(f"\nInput text: '{text}'")

# Tokenize
inputs = tokenizer(text, return_tensors="np")

# Prepare input dict for OpenVINO
# The model expects: input_ids, attention_mask, position_ids
ov_inputs = {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
    # position_ids is often optional or can be generated, but let's see if the model needs it.
    # Based on the verification output, it is an input.
}

# Some models require position_ids explicitly if exported that way
if "position_ids" not in inputs:
    seq_len = inputs["input_ids"].shape[1]
    ov_inputs["position_ids"] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

print("Running inference...")
result = infer_request.infer(ov_inputs)

# The output key might vary, usually it's the last hidden state or pooler output
# Let's print available outputs
print("\nOutput keys:", list(result.keys()))

# Assuming the first output is the embeddings
embedding = list(result.values())[0]
print(f"Embedding shape: {embedding.shape}")
print(f"First 5 values: {embedding[0, 0, :5]}")
