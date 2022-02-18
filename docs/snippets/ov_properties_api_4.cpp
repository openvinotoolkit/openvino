#include <openvino/runtime/core.hpp>

int main() {
//! [part4]
ov::Core core;
auto model = core.read_model("sample.xml");
auto compiled_model = core.compile_model(model, "MYRIAD");
auto temperature = compiled_model.get_property(ov::device::thermal);
//! [part4]
return 0;
}
