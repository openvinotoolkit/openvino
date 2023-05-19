//! [export_compiled_model]

#include <openvino/runtime/core.hpp>

int main() {

ov::Core core;

std::stringstream stream;

ov::CompiledModel model = core.compile_model("modelPath", "deviceName");

model.export_model(stream);

    return 0;
}

//!  [export_compiled_model]
