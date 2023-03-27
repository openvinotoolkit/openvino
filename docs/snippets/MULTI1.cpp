#include <openvino/openvino.hpp>

int main() {
//! [part1]
ov::Core core;
std::shared_ptr<ov::Model> model = core.read_model("sample.xml");
ov::CompiledModel compileModel = core.compile_model(model, "MULTI:CPU,GPU");

// Once the priority list is set, you can alter it on the fly:
// reverse the order of priorities
compileModel.set_property(ov::device::priorities("GPU,CPU"));

// exclude some devices (in this case, CPU)
compileModel.set_property(ov::device::priorities("GPU"));

// bring back the excluded devices
compileModel.set_property(ov::device::priorities("GPU,CPU"));

// You cannot add new devices on the fly!
// Attempting to do so will trigger the following exception:
// [ ERROR ] [NOT_FOUND] You can only change device
// priorities but not add new devices with the model's
// ov::device::priorities. CPU device was not in the original device list!

//! [part1]
return 0;
}
