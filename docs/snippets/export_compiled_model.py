#! [export_compiled_model]

import openvino as ov

ov.Core().compile_model(device, modelPath, properties).export_model(compiled_blob)

#! [export_compiled_model]
