import openvino as ov
import os

# Path to the built extension
ext_path = r"src/custom_ops/build/Release/openvino_tssn_extension.dll"

print(f"Loading extension from: {ext_path}")

core = ov.Core()
try:
    core.add_extension(ext_path)
    print("Extension loaded successfully!")
except Exception as e:
    print(f"Failed to load extension: {e}")
