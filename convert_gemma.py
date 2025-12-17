import openvino as ov
from pathlib import Path
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

def convert_gemma():
    model_path = Path("embeddinggemma_local")
    output_dir = Path("gemma_ir")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading and converting model from {model_path} using Optimum Intel...")
    try:
        # Load and export model using Optimum Intel
        model = OVModelForCausalLM.from_pretrained(
            model_path, 
            export=True,
            trust_remote_code=True,
            use_cache=True # Keep cache for performance, Optimum handles it
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        print(f"Saving model to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"Model successfully saved to {output_dir}")
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    convert_gemma()

if __name__ == "__main__":
    convert_gemma()
