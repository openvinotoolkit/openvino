import numpy as np
from openvino.runtime import Core
import time

print("Using runtime config to disable layer fusion")

core = Core()

# 1️⃣ Disable fusion BEFORE compile_model
core.set_property({"DISABLE_LAYER_FUSION": "YES"})

model = core.read_model(model_path)

start_compile = time.time()
compiled_model = core.compile_model(model, "CPU")
compile_time = time.time() - start_compile

input_shape = compiled_model.input(0).shape
dummy_input = np.random.rand(*input_shape).astype(np.float32)

start_infer = time.time()
_ = compiled_model([dummy_input])
infer_time = time.time() - start_infer

print(f"\n✅ Compile Time: {compile_time:.4f} sec")
print(f"✅ First Inference Time: {infer_time:.4f} sec")

print("\n📋 Runtime Graph (Visible ops like ReLU, Conv, etc.):")
for op in compiled_model.get_runtime_model().get_ops():
    print(f"{op.get_type_name():<25} {op.friendly_name}")

# ////////////////////////////////////////////////////////////////////////////////////////////////////
import numpy as np
from openvino.runtime import Core
import os
import time

print(f"DISABLE_LAYER_FUSION env var is: {os.getenv('DISABLE_LAYER_FUSION', 'NOT SET')}")

core = Core()

model = core.read_model(model_path)

start_compile = time.time()
compiled_model = core.compile_model(model, "CPU")
compile_time = time.time() - start_compile

input_shape = compiled_model.input(0).shape
dummy_input = np.random.rand(*input_shape).astype(np.float32)

start_infer = time.time()
_ = compiled_model([dummy_input])
infer_time = time.time() - start_infer

print(f"\n✅ Compile Time: {compile_time:.4f} sec")
print(f"✅ First Inference Time: {infer_time:.4f} sec")

print("\n📋 Runtime Graph (Visible ops like ReLU, Conv, etc.):")
for op in compiled_model.get_runtime_model().get_ops():
    print(f"{op.get_type_name():<25} {op.friendly_name}")
# ////////////////////////////////////////////////////////////////////////////////////////////////////