#include <openvino/runtime/core.hpp>

int main() {
//! [part5]
ov::Core core;
auto model = core.read_model("sample.xml");
auto compiled_model = core.compile_model(model, "CPU");
auto nthreads = compiled_model.get_property(ov::inference_num_threads);
//! [part5]
return 0;
}
