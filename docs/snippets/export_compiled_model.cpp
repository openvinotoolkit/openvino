//! [export_compiled_model]

#include <openvino/runtime/core.hpp>

ov::Core core;

core.compile_model(device, modelPath, properties).export_model(compiled_blob);

//!  [export_compiled_model]
