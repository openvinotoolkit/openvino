#include <openvino/openvino.hpp>

int main() {
//! [part0]
ov::Core core;

// Read a network in IR, PaddlePaddle, or ONNX format:
std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

// compile a model on MULTI using the default list of device candidates.
ov::CompiledModel compileModel0 = core.compile_model(model, "MULTI");

// Optional
// You can also specify the devices to be used by MULTI.
// The following lines are equivalent:
ov::CompiledModel compileModel1 = core.compile_model(model, "MULTI:HDDL,GPU");
ov::CompiledModel compileModel2 = core.compile_model(model, "MULTI", ov::device::priorities("HDDL,GPU"));

// Optional
// MULTI is pre-configured (globally) with the explicit option.
core.set_property("MULTI", ov::device::priorities("HDDL,GPU"));

//! [part0]
return 0;
}
