#include <openvino/openvino.hpp>

int main() {
//! [part0]
ov::Core core;

// Read a model in IR, PaddlePaddle, or ONNX format:
std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

// Option 1
// Pre-configure MULTI globally with explicitly defined devices,
// and compile the model on MULTI using the newly specified default device list.
core.set_property("MULTI", ov::device::priorities("GPU.1,GPU.0")); 
ov::CompiledModel compileModel0 = core.compile_model(model, "MULTI");

// Option 2
// Specify the devices to be used by MULTI explicitly at compilation.
// The following lines are equivalent:
ov::CompiledModel compileModel1 = core.compile_model(model, "MULTI:GPU.1,GPU.0");
ov::CompiledModel compileModel2 = core.compile_model(model, "MULTI", ov::device::priorities("GPU.1,GPU.0"));



//! [part0]
return 0;
}
