import openvino as ov
import numpy as np
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch

class OpenVINOModelWrapper:
    def __init__(self, model_path, tokenizer_path):
        self.core = ov.Core()
        print(f"Loading model from {model_path}...")
        self.model = self.core.compile_model(model_path, "CPU")
        self.infer_request = self.model.create_infer_request()
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.device = "cpu"

    def encode(self, sentences, batch_size=32, **kwargs):
        all_embeddings = []
        
        # Handle DataLoader or list
        if hasattr(sentences, 'batch_size'): # It's a DataLoader
             # Iterate over the dataloader directly
             for batch in sentences:
                 # If batch is a dict (common in HF DataLoaders), extract text
                 if isinstance(batch, dict):
                     # Try to find the text key. Usually 'sentence' or 'text'
                     if 'sentence' in batch:
                         batch = batch['sentence']
                     elif 'text' in batch:
                         batch = batch['text']
                 
                 self._process_batch(batch, all_embeddings)
            
        return np.vstack(all_embeddings)

    def _process_batch(self, batch_sentences, all_embeddings):
            # Tokenize
            inputs = self.tokenizer(
                batch_sentences, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="np"
            )
            
            ov_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
            
            # Add position_ids if needed (simple range)
            seq_len = inputs["input_ids"].shape[1]
            batch_len = inputs["input_ids"].shape[0]
            position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            ov_inputs["position_ids"] = np.repeat(position_ids, batch_len, axis=0)

            # Infer
            result = self.infer_request.infer(ov_inputs)
            
            # Get last_hidden_state (usually output 0)
            last_hidden_state = list(result.values())[0] # [Batch, Seq, Hidden]
            
            # Mean Pooling
            # Mask out padding tokens
            attention_mask = inputs["attention_mask"]
            input_mask_expanded = np.expand_dims(attention_mask, -1).astype(last_hidden_state.dtype)
            
            sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
            sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            
            embeddings = sum_embeddings / sum_mask
            
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            
            all_embeddings.append(embeddings)

if __name__ == "__main__":
    import sys
    model_path = "model_ir/openvino_model.xml"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    # Initialize wrapper
    print(f"Loading model from {model_path}...")
    model = OpenVINOModelWrapper(model_path, "./embeddinggemma_local")
    
    # Select a fast task
    task_name = "STS12" 
    print(f"Running MTEB task: {task_name}")
    
    # New API usage
    import mteb
    tasks = mteb.get_tasks(tasks=[task_name])
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder="mteb_results")
    
    print("\n--- MTEB Results ---")
    for task_res in results:
        print(f"Task: {task_res.task_name}")
        print(f"Score: {task_res.scores}")
