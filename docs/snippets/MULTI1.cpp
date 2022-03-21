#include <openvino/openvino.hpp>

int main() {
//! [part1]
ov::Core core;
std::shared_ptr<ov::Model> model = core.read_model("sample.xml");
ov::CompiledModel compileModel = core.compile_model(model, "MULTI:HDDL,GPU");
// reverse the order of priorities
compileModel.set_property(ov::device::priorities("GPU,HDDL"));
// you can even exclude some device (HDDL here)
compileModel.set_property(ov::device::priorities("GPU"));
// and then return it back
compileModel.set_property(ov::device::priorities("GPU,HDDL"));
// but you cannot add new devices on the fly, 
// the next line will trigger the following exception:
// [ ERROR ] [NOT_FOUND] You can only change device
// priorities but not add new devices with the model's
// ov::device::priorities. CPU device was not in the original device list!
compileModel.set_property(ov::device::priorities("CPU,GPU,HDDL"));
//! [part1]
return 0;
}
