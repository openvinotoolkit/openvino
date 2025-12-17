import openvino as ov
import numpy as np
import os

def test_injected_model():
    model_path = "injected_model.xml"
    ext_path = os.path.abspath("src/custom_ops/build/Release/openvino_tssn_extension.dll")
    gpu_xml_path = os.path.abspath("src/custom_ops/composite_tssn_gpu.xml")
    
    core = ov.Core()
    core.add_extension(ext_path)
    core.set_property("GPU", {"CONFIG_FILE": gpu_xml_path})
    
    print(f"Loading {model_path}...")
    model = core.read_model(model_path)
    
    print("Compiling...")
    compiled_model = core.compile_model(model, "GPU")
    
    input_data = np.random.rand(1, 1024).astype(np.float32)
    
    print("Running inference...")
    res = compiled_model([input_data])[0]
    print("Success! Output shape:", res.shape)

if __name__ == "__main__":
    test_injected_model()
