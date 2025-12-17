import openvino as ov
import sys

try:
    core = ov.Core()
    model = core.read_model("model_ir/openvino_model.xml")
    print("Model loaded successfully!")
    print("Inputs:")
    for input in model.inputs:
        print(f"  {input.any_name}: {input.get_partial_shape()}")
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)
