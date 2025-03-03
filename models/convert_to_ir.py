import openvino as ov

# Convert ONNX model to OpenVINO IR
ov_model = ov.convert_model('dropout_fixed.onnx')  

# Save the IR model
ir_path = "dropout_fixed.xml"
ov.save_model(ov_model, ir_path)

print(f"Model saved as {ir_path}")
