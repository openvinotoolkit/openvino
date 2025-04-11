import onnx
import numpy as np
from onnx import helper, TensorProto

# Define input tensor
input_tensor = helper.make_tensor_value_info(
    "input", TensorProto.FLOAT, [1, 3, 224, 224]
)

# **Instead of Dropout, use an Identity node to keep the model structure**
identity_node = helper.make_node("Identity", inputs=["input"], outputs=["output"])

# Define output tensor
output_tensor = helper.make_tensor_value_info(
    "output", TensorProto.FLOAT, [1, 3, 224, 224]
)

# Create the model graph
graph = helper.make_graph(
    nodes=[identity_node],  # 🔥 Using Identity instead of Dropout
    name="DropoutGraphFixed",
    inputs=[input_tensor],
    outputs=[output_tensor],
)

# Create the ONNX model
model = helper.make_model(graph, producer_name="dropout_model_generator")

# Save the model
onnx.save(model, "dropout_fixed.onnx")

print("dropout_fixed.onnx has been generated successfully!")
