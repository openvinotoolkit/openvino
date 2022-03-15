#include <openvino/openvino.hpp>

int main() {
{
//! [part5]
ov::Core core;

// Read a network in IR, PaddlePaddle, or ONNX format:
std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

// Configure the CPU and the Myriad devices separately and load the network to Auto-Device plugin:
// set CPU and GPU config
core.set_property({ov::device::properties("CPU", ov::enable_profiling(true)),
    ov::device::properties("GPU", ov::enable_profiling(false))});
ov::CompiledModel compiled_model = core.compile_model(model, "AUTO");
//! [part5]
}
    return 0;
}
