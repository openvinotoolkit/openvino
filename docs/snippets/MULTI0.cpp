#include <openvino/openvino.hpp>

int main() {
//! [part0]
ov::Core core;
std::shared_ptr<ov::Model> model = core.read_model("sample.xml");
// the "MULTI" device is (globally) pre-configured with the explicit option
core.set_property("MULTI", ov::device::priorities("HDDL,GPU"));
ov::CompiledModel compileModel0 = core.compile_model(model, "MULTI");

// configuration of the "MULTI" is part of the compile configuration (and hence specific to the model):
ov::CompiledModel compileModel1 = core.compile_model(model, "MULTI", ov::device::priorities("HDDL,GPU"));

// same as previous, but configuration of the "MULTI" is part
// of the name (so config is empty), also model-specific:
ov::CompiledModel compileModel2 = core.compile_model(model, "MULTI:HDDL,GPU");
//! [part0]
return 0;
}
