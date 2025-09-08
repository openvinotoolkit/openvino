#include <openvino/openvino.hpp>

int main() {
ov::AnyMap cpu_config = {};
ov::AnyMap gpu_config = {};
//! [part5]
ov::Core core;

// Read a network in IR, TensorFlow, TensorFlow Lite, PaddlePaddle, or ONNX format:
std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

// Configure the CPU and the GPU devices when compiling model
ov::CompiledModel compiled_model = core.compile_model(model, "AUTO",
    ov::device::properties("CPU", cpu_config),
    ov::device::properties("GPU", gpu_config));
//! [part5]
    return 0;
}
