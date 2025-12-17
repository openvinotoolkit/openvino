import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer
import openvino as ov
from pathlib import Path
import shutil

def fix_and_convert():
    model_path = Path("embeddinggemma_local")
    output_dir = Path("gemma_ir_fixed")
    output_dir.mkdir(exist_ok=True)
    
    print("Loading state dict...")
    state_dict = load_file(model_path / "model.safetensors")
    
    new_state_dict = {}
    print("Fixing keys...")
    for k, v in state_dict.items():
        new_key = f"model.{k}"
        new_state_dict[new_key] = v
        
    # Tie weights (not needed for AutoModel usually, but good to have if we switch back)
    if "model.embed_tokens.weight" in new_state_dict:
        # print("Tying lm_head to embed_tokens...")
        # new_state_dict["lm_head.weight"] = new_state_dict["model.embed_tokens.weight"]
        pass
    
    print("Loading Config...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Update architecture
    config.architectures = ["Gemma3Model"] # Base model
    
    print("Instantiating Model...")
    try:
        # Use AutoModel to get the base model (embeddings output) instead of CausalLM (logits output)
        model = AutoModel.from_config(config, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to instantiate AutoModel: {e}")
        return

    print("Loading state dict into model...")
    # strict=False because we might have extra keys (lm_head) that we don't need for the base model
    # The state dict we prepared has "model.layers...", but AutoModel expects "layers..." usually?
    # Wait, AutoModelForCausalLM wraps the base model in "model".
    # AutoModel IS the base model.
    # So if we use AutoModel, we should NOT prefix with "model.".
    
    # Let's revert the key fixing for AutoModel
    print("Re-adjusting keys for AutoModel...")
    final_state_dict = {}
    for k, v in state_dict.items():
        # The original safetensors had keys like "layers.0..."
        # AutoModel expects "layers.0..."
        final_state_dict[k] = v
        
    missing, unexpected = model.load_state_dict(final_state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    
    model.eval()
    
    print("Converting to OpenVINO...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    dummy_input = tokenizer("Hello world", return_tensors="pt")
    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
    
    example_input = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids
    }
    
    try:
        print("Exporting to OpenVINO using Optimum...")
        
        # Save the "fixed" model to a temporary directory as a pure transformers model, 
        # and then use OVModelForFeatureExtraction to load and export it.
        # This is the cleanest way to ensure Optimum handles all the tracing/input generation correctly.
        
        temp_dir = Path("temp_fixed_model")
        print(f"Saving fixed PyTorch model to temporary dir: {temp_dir}...")
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)
        config.save_pretrained(temp_dir)
        
        print("Loading and exporting with OVModelForFeatureExtraction...")
        from optimum.intel import OVModelForFeatureExtraction
        
        ov_model = OVModelForFeatureExtraction.from_pretrained(
            temp_dir,
            export=True,
            compile=False
        )
        
        print(f"Saving OpenVINO model to {output_dir}...")
        ov_model.save_pretrained(output_dir)
        
        print("Done.")
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_and_convert()
