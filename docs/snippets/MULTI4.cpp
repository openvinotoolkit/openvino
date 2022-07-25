#include <openvino/openvino.hpp>

int main() {
ov::AnyMap myriad_config, gpu_config;
//! [part4]
ov::Core core;

// Read a network in IR, PaddlePaddle, or ONNX format:
std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

// When compiling the model on MULTI, configure GPU and HDDL 
// (devices, priorities, and device configurations):
ov::CompiledModel compileModel = core.compile_model(model, "MULTI",
    ov::device::priorities("HDDL", "GPU"),
    ov::device::properties("GPU", gpu_config),
    ov::device::properties("HDDL", myriad_config));

// Optionally, query the optimal number of requests:
uint32_t nireq = compileModel.get_property(ov::optimal_number_of_infer_requests);
//! [part4]
return 0;
}
