#include <openvino/openvino.hpp>

int auto7() {
{
//! [part7]
ov::Core core;

// read a network in IR, PaddlePaddle, or ONNX format
std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

// compile a model on AUTO and set log level to debug
ov::CompiledModel compiled_model = core.compile_model(model, "AUTO");
// query the runtime target devices on which the inferences are being executed
ov::Any execution_devices = compiled_model.get_property(ov::execution_devices);
//! [part7]
}
    return 0;
}
