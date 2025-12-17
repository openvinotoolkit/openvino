import openvino as ov
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import os

def evaluate_embedding_quality():
    # Paths
    dense_model_path = "embeddinggemma_local"
    sparse_model_path = "model_ir_pruned/gemma_embedding_pruned.xml"
    ext_path = r"src/custom_ops/build/Release/openvino_tssn_extension.dll"
    
    print("1. Loading Reference Dense Model (PyTorch)...")
    try:
        # Load original model for ground truth
        ref_tokenizer = AutoTokenizer.from_pretrained(dense_model_path, trust_remote_code=True)
        ref_model = AutoModel.from_pretrained(dense_model_path, trust_remote_code=True)
        ref_model.eval()
    except Exception as e:
        print(f"Failed to load reference model: {e}")
        return

    print("2. Loading Sparse TSSN Model (OpenVINO)...")
    core = ov.Core()
    if os.path.exists(ext_path):
        core.add_extension(ext_path)
    else:
        print(f"Error: Extension not found at {ext_path}")
        return
        
    try:
        model = core.read_model(sparse_model_path)
        compiled_model = core.compile_model(model, "CPU")
        infer_request = compiled_model.create_infer_request()
    except Exception as e:
        print(f"Failed to load sparse model: {e}")
        return

    # Test Sentences
    sentences = [
        "The future of artificial intelligence is sparse.",
        "OpenVINO optimizes deep learning models for inference.",
        "The quick brown fox jumps over the lazy dog.",
        "Semantic search requires high quality vector embeddings.",
        "Photosynthesis is the process by which plants use sunlight to synthesize foods."
    ]
    
    print(f"\n3. Comparing Embeddings for {len(sentences)} sentences...")
    
    similarities = []
    
    for text in sentences:
        print(f"\nProcessing: '{text}'")
        
        # --- Reference Inference ---
        inputs = ref_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = ref_model(**inputs)
            # Mean pooling for sentence embedding
            # Attention mask needed to ignore padding
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            ref_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            ref_embedding = ref_embedding.numpy()

        # --- Sparse Inference ---
        # Prepare inputs for OpenVINO
        ov_inputs = {k: v.numpy() for k, v in inputs.items()}
        
        # Check what inputs the model expects
        expected_inputs = {inp.get_any_name(): inp for inp in compiled_model.inputs}
        
        # Filter inputs
        final_ov_inputs = {}
        for name, tensor in ov_inputs.items():
            if name in expected_inputs:
                final_ov_inputs[name] = tensor
        
        # Add position_ids if expected and missing
        if "position_ids" in expected_inputs and "position_ids" not in final_ov_inputs:
             batch_size = ov_inputs["input_ids"].shape[0]
             seq_len = ov_inputs["input_ids"].shape[1]
             final_ov_inputs["position_ids"] = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)

        # Run inference
        results = infer_request.infer(final_ov_inputs)
        
        # Output 0 might be logits [Batch, Seq, Vocab] if the model was converted with a head
        # or last_hidden_state [Batch, Seq, Hidden] if it wasn't.
        # The error says Y.shape[1] == 262144, which is the vocab size of Gemma.
        # This means we are getting logits, not embeddings.
        # We need to find the last hidden state output, or if not available, use the logits (bad idea)
        # or re-convert the model without the head.
        
        # However, for now, let's check if there are other outputs.
        # If only one output and it's logits, we can't easily get the embedding from the compiled model
        # unless we modify the IR to output the layer before the head.
        
        # BUT, wait. The reference model `ref_model` (AutoModel) outputs hidden states (768).
        # The OpenVINO model `sparse_model_path` was converted from `AutoModelForCausalLM` (in fix_and_convert.py).
        # So the OpenVINO model HAS a head and outputs logits (262144).
        
        # To compare, we should use the reference model's logits? No, that's perplexity again.
        # We want embeddings.
        
        # Quick fix: The OpenVINO model has intermediate layers. We can try to request the output of the last transformer layer.
        # But `compiled_model` usually only exposes the final results.
        
        # Let's try to see if we can get the hidden state from the logits? No.
        
        # We need to re-convert the model as an Embedding model (no head) OR
        # use the reference model as CausalLM and compare logits (but we know that's bad).
        
        # Actually, the user wants to use it as an Embedding model.
        # So we should have converted it as `AutoModel` (without head), not `AutoModelForCausalLM`.
        # In `fix_and_convert.py`, we used `AutoModelForCausalLM`.
        
        # Let's modify this script to just print the shape mismatch and exit for now, 
        # as we need to fix the conversion first.
        
        ov_output = results[compiled_model.output(0)]
        print(f"  OpenVINO Output Shape: {ov_output.shape}")
        
        if ov_output.shape[-1] == 262144:
            print("  WARNING: Model is outputting Logits (Vocab Size), not Embeddings.")
            print("  Cannot compare with reference embeddings directly.")
            return

        ov_last_hidden_state = ov_output

        
        # Mean pooling manually for OpenVINO output
        # ov_last_hidden_state: [1, Seq, 768]
        # mask: [1, Seq]
        mask = ov_inputs["attention_mask"]
        mask_expanded = np.expand_dims(mask, -1) # [1, Seq, 1]
        
        sum_embeddings = np.sum(ov_last_hidden_state * mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
        sparse_embedding = sum_embeddings / sum_mask
        
        # --- Comparison ---
        # Cosine Similarity
        sim = cosine_similarity(ref_embedding, sparse_embedding)[0][0]
        similarities.append(sim)
        print(f"  Cosine Similarity: {sim:.4f}")
        
    avg_sim = np.mean(similarities)
    print(f"\nAverage Semantic Preservation: {avg_sim:.4f}")
    
    if avg_sim > 0.9:
        print("SUCCESS: Sparse model preserves semantic meaning well.")
    elif avg_sim > 0.7:
        print("WARNING: Significant semantic degradation. Fine-tuning required.")
    else:
        print("FAILURE: Model brain damage. Incision too aggressive or weights mismatched.")

if __name__ == "__main__":
    evaluate_embedding_quality()
