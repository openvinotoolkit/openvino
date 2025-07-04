
# # from openvino.runtime import Core
# # import openvino
# # print(openvino.__file__)

# # core = Core()
# # model = core.read_model(r"C:\Users\LENOVO\Documents\OpenVINO_Work\openvino_disable_fusion\public\resnet-50-pytorch\FP16\resnet-50-pytorch.xml")
# # compiled_model = core.compile_model(model=model, device_name="CPU", config={"DISABLE_LAYER_FUSION": "YES"})
# # print("Model compiled with DISABLE_LAYER_FUSION") 
# import numpy as np
# import time
# from openvino.runtime import Core

# model_path = r"C:\Users\LENOVO\Documents\OpenVINO_Work\openvino_disable_fusion\public\resnet-50-pytorch\FP16\resnet-50-pytorch.xml"
# input_shape = (1, 3, 224, 224)  # شكل الإدخال المتوقع للـ ResNet-50
# dummy_input = np.random.rand(*input_shape).astype(np.float32)

# core = Core()

# # ========== 1. WITH Layer Fusion ==========
# start_compile = time.time()
# model = core.read_model(model_path)
# compiled_with_fusion = core.compile_model(model=model, device_name="CPU")  # بدون تعطيل
# compile_time_fusion = time.time() - start_compile

# input_tensor = compiled_with_fusion.input(0)
# start_infer = time.time()
# _ = compiled_with_fusion([dummy_input])
# infer_time_fusion = time.time() - start_infer

# # ========== 2. WITHOUT Layer Fusion ==========
# start_compile = time.time()
# model = core.read_model(model_path)
# compiled_without_fusion = core.compile_model(
#     model=model,
#     device_name="CPU",
#     config={"DISABLE_LAYER_FUSION": "YES"}
# )
# compile_time_no_fusion = time.time() - start_compile

# input_tensor = compiled_without_fusion.input(0)
# start_infer = time.time()
# _ = compiled_without_fusion([dummy_input])
# infer_time_no_fusion = time.time() - start_infer

# # ========== مقارنة النتائج ==========
# print("\n🔍 Comparison of Layer Fusion vs No Fusion:")
# print(f"✅ Compile Time WITH Fusion     : {compile_time_fusion:.4f} seconds")
# print(f"❌ Compile Time WITHOUT Fusion  : {compile_time_no_fusion:.4f} seconds")
# print(f"✅ Inference Time WITH Fusion   : {infer_time_fusion:.4f} seconds")
# print(f"❌ Inference Time WITHOUT Fusion: {infer_time_no_fusion:.4f} seconds")
import time
import numpy as np
from openvino.runtime import Core
def benchmark(compiled_model, input_data, runs=100):
    infer_request = compiled_model.create_infer_request()
    times = []

    for _ in range(runs):
        start = time.time()
        infer_request.infer({compiled_model.inputs[0]: input_data})
        times.append(time.time() - start)

    return sum(times) / len(times)
core = Core()

model_path = r"C:\Users\LENOVO\Documents\OpenVINO_Work\openvino_disable_fusion\public\resnet-50-pytorch\FP16\resnet-50-pytorch.xml"
model = core.read_model(model_path)

# حمّلي نفس الموديل مرتين: مرّة مع ومرّة من غير Layer Fusion
compiled_with_fusion = core.compile_model(model, "CPU")
compiled_without_fusion = core.compile_model(model, "CPU", config={"DISABLE_LAYER_FUSION": "YES"})

# جهّزي dummy input بنفس شكل input الموديل
input_shape = compiled_with_fusion.inputs[0].shape  # غالباً [1, 3, 224, 224]
input_data = np.random.rand(*input_shape).astype(np.float32)
avg_time_fusion = benchmark(compiled_with_fusion, input_data)
avg_time_no_fusion = benchmark(compiled_without_fusion, input_data)

print(f"✅ Avg Inference Time WITH Fusion    : {avg_time_fusion:.6f} seconds")
print(f"❌ Avg Inference Time WITHOUT Fusion : {avg_time_no_fusion:.6f} seconds")
