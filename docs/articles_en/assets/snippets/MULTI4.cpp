#include <openvino/openvino.hpp>

int main() {
ov::AnyMap cpu_config, gpu_config;
//! [part4]
ov::Core core;

// Read a network in IR, PaddlePaddle, or ONNX format:
std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

// When compiling the model on MULTI, configure GPU and CPU 
// (devices, priorities, and device configurations):
ov::CompiledModel compileModel = core.compile_model(model, "MULTI",
    ov::device::priorities("GPU", "CPU"),
    ov::device::properties("GPU", gpu_config),
    ov::device::properties("CPU", cpu_config));

// Optionally, query the optimal number of requests:
uint32_t nireq = compileModel.get_property(ov::optimal_number_of_infer_requests);
//! [part4]
return 0;
}
