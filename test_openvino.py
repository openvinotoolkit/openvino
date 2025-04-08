import openvino.runtime.opset14 as ov
import numpy as np

print("Creating string tensor constant...")
str_const = ov.constant(np.array(['openvino'], dtype=str))
print("String constant created:", str_const)
