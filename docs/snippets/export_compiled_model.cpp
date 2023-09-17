
#include <openvino/runtime/core.hpp>

int main() {

//! [export_compiled_model]
ov::Core core;

std::stringstream stream;

ov::CompiledModel model = core.compile_model("modelPath", "deviceName");

model.export_model(stream);

//!  [export_compiled_model]

return 0;
}

