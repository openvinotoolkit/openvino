//! [compile_model]

#include <openvino/runtime/core.hpp>

ov::Core.compile_model(device, modelPath, properties).export_model(compiled_blob);

//!  [compile_model]
