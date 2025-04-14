import openvino as ov

core = ov.Core()

model = core.read_model(model="path/to/your/model.xml")

compiled_model = core.compile_model(model=model, device_name="CPU")

infer_request = compiled_model.create_infer_request()

input_data = {"input_tensor_name": [1.0, 2.0, 3.0, 4.0]}

infer_request.infer(inputs=input_data)

output = infer_request.get_output_tensor().data

print("Inference result:", output)
