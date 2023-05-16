#! [compile_model]

from openvino.runtime import Core

ov.Core().compile_model(device, modelPath, properties).export_model(compiled_blob)

#! [compile_model]
