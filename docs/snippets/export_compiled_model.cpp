//! [export_compiled_model]

#include <openvino/openvino.hpp>

ov::Core core;

std::stringstream stream;

ov::CompiledModel model = core.compile_model("modelPath", "deviceName");
model.export_model(stream);

//!  [export_compiled_model]
