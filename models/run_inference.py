from openvino.runtime import Core
import numpy as np

# Step 1: Load OpenVINO Runtime
ie = Core()

# Step 2: Read the IR Model
model_path = "ir_model/dropout_fixed.xml"
model = ie.read_model(model_path)

# Step 3: Compile the Model for Inference
compiled_model = ie.compile_model(model, "CPU")  # Use "GPU" if you have a GPU

# Step 4: Get Input Layer Information
input_layer = compiled_model.input(0)
input_shape = input_layer.shape  # Get input shape

# Step 5: Prepare Sample Input Data
sample_input = np.random.rand(*input_shape).astype(np.float32)  # Random input

# Step 6: Run Inference
output = compiled_model(sample_input)

# Step 7: Print the Output
print("Inference Output:", output)
