# from openvino.runtime import Core

# def test_disable_layer_fusion():
#     core = Core()

#     config = {"DISABLE_LAYER_FUSION": "YES"}
#     device = "CPU"

#     core.set_property(device, config)

#     model_path = r"C:\Users\LENOVO\Documents\OpenVINO_Work\openvino_disable_fusion\public\resnet-50-pytorch\FP32\resnet-50-pytorch.xml"
#     model = core.read_model(model_path)

#     compiled_model = core.compile_model(model, device)

#     print("Model compiled with DISABLE_LAYER_FUSION=YES")

# if __name__ == "__main__":
#     test_disable_layer_fusion()
import time
from openvino.runtime import Core
import numpy as np

# مسار الموديل
model_path = r"C:\Users\LENOVO\Documents\OpenVINO_Work\openvino_disable_fusion\public\resnet-50-pytorch\FP32\resnet-50-pytorch.xml"

# إنشاء الكور
core = Core()

# دالة لتشغيل inference مع أو بدون fusion
def run_inference(disable_fusion: bool):
    config = {"DISABLE_LAYER_FUSION": "YES" if disable_fusion else "NO"}
    print(f"\n🔧 Running with config: {config}")

    compiled_model = core.compile_model(model=model_path, device_name="CPU", config=config)

    input_layer = compiled_model.input(0)
    dummy_input = np.random.randn(*input_layer.shape).astype(np.float32)

    start = time.time()
    compiled_model.infer_new_request({input_layer.any_name: dummy_input})
    end = time.time()

    return end - start

# تشغيل التجربة مرتين
time_with_fusion = run_inference(False)
time_without_fusion = run_inference(True)

# طباعة النتائج
print(f"\n✅ وقت الـ inference مع الفيوجن مفعّل   : {time_with_fusion:.4f} ثانية")
print(f"❌ وقت الـ inference مع تعطيل الفيوجن : {time_without_fusion:.4f} ثانية")
print(f"📉 الفرق بينهم                         : {abs(time_with_fusion - time_without_fusion):.4f} ثانية")

# معلومات إضافية
print("\n🧠 معلومات عن المعالج:")
print("FULL_DEVICE_NAME:", core.get_property("CPU", "FULL_DEVICE_NAME"))
print("CACHE_DIR        :", core.get_property("CPU", "CACHE_DIR"))
