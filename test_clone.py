import openvino as ov
import numpy as np

core = ov.Core()

# Create a simple Add node
param = ov.opset10.parameter([1, 10], np.float32)
const = ov.opset10.constant(np.ones((1, 10), dtype=np.float32))
add_node = ov.opset10.add(param, const)

print(f"Node: {add_node}")
print(f"Type: {type(add_node)}")

# Try to clone
try:
    new_input = ov.opset10.parameter([1, 10], np.float32)
    # In Python, we usually re-create the op. 
    # But for a custom op where we don't have the constructor, we need a generic clone.
    # Check attributes
    if hasattr(add_node, 'clone_with_new_inputs'):
        print("Has clone_with_new_inputs")
        cloned = add_node.clone_with_new_inputs({0: new_input, 1: const})
        print(f"Cloned: {cloned}")
    else:
        print("No clone_with_new_inputs")
except Exception as e:
    print(f"Error: {e}")
