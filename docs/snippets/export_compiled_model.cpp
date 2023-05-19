//! [export_compiled_model]

#include <openvino/runtime/core.hpp>

ov::Core core;

ov::CompiledModel model = core.compile_model(device, modelPath, properties)
model.export_model(compiled_blob);

//!  [export_compiled_model]
