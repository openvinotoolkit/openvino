import openvino as ov
import os
import hashlib
import shutil
from pathlib import Path
from optimum.intel import OVModelForSequenceClassification

# --- Configuration ---
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
CACHE_DIR = "temp_ov_cache_test" # Use a unique name for the test cache
NUM_COMPILATIONS = 10
DEVICE = "CPU"

def get_file_hash(filepath):
    """Calculates the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# --- Main Execution ---
print(f"--- Cache Duplication Test for Issue #31672 ---")

# Step 1: Clean up and create cache directory
if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)
os.makedirs(CACHE_DIR)

# Step 2: Download and convert model
print(f"Acquiring model: {MODEL_ID}...")
model = OVModelForSequenceClassification.from_pretrained(MODEL_ID, export=True)
model_path = Path(model.model_save_dir) / "openvino_model.xml"

# Step 3: Repeatedly compile the model
print(f"\nCompiling model {NUM_COMPILATIONS} times...")
core = ov.Core()
core.set_property({"CACHE_DIR": CACHE_DIR})

for i in range(NUM_COMPILATIONS):
    model_obj = core.read_model(str(model_path))
    compiled_model = core.compile_model(model_obj, device_name=DEVICE)
    del compiled_model
    del model_obj

# Step 4: Analyze cache directory
blob_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.blob')]

if not blob_files:
    raise RuntimeError("No cache (.blob) files were generated.")

file_hashes = {get_file_hash(os.path.join(CACHE_DIR, f)) for f in blob_files}

# Step 5: Report result
print(f"\nFound {len(blob_files)} total cache files.")
print(f"Found {len(file_hashes)} unique cache files.")

if len(file_hashes) == 1:
    print("\n[SUCCESS] Test passed. Only one unique cache file was generated.")
else:
    print("\n[FAILURE] Test failed. Multiple unique cache files were generated.")
    # Exit with a non-zero code to make CI fail if the bug reappears
    exit(1)

# Final cleanup
shutil.rmtree(CACHE_DIR)
