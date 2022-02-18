#include <openvino/runtime/core.hpp>

int main() {
//! [part3]
ov::Core core;
auto model = core.read_model("sample.xml");
auto compiled_model = core.compile_model(model, "CPU");
auto nireq = compiled_model.get_property(ov::optimal_number_of_infer_requests);
//! [part3]
return 0;
}
