#include <fstream>
#include <openvino/runtime/core.hpp>

int main() {

//! [export_compiled_model]
ov::Core core;
ov::CompiledModel model = core.compile_model("modelPath", "deviceName");
std::fstream stream("compiled_model.blob");
model.export_model(stream);
//!  [export_compiled_model]

return 0;
}

