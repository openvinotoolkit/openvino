#include <openvino/openvino.hpp>

int main() {
ov::AnyMap cpu_config = {};
ov::AnyMap myriad_config = {};
//! [part5]
ov::Core core;

// Read a network in IR, PaddlePaddle, or ONNX format:
std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

// Configure  CPU and the MYRIAD devices when compiled model
ov::CompiledModel compiled_model = core.compile_model(model, "AUTO",
    ov::device::properties("CPU", cpu_config),
    ov::device::properties("MYRIAD", myriad_config));
//! [part5]
    return 0;
}
