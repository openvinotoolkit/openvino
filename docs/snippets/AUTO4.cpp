#include <openvino/openvino.hpp>

int main() {
    ov::Core core;

    // Read a network in IR, PaddlePaddle, or ONNX format:
    std::shared_ptr<ov::Model> model = core.read_model("sample.xml");
{
//! [part4]
// Example 1
ov::CompiledModel compiled_model0 = core.compile_model(model, "AUTO",
    ov::hint::model_priority(ov::hint::Priority::HIGH));
ov::CompiledModel compiled_model1 = core.compile_model(model, "AUTO",
    ov::hint::model_priority(ov::hint::Priority::MEDIUM));
ov::CompiledModel compiled_model2 = core.compile_model(model, "AUTO",
    ov::hint::model_priority(ov::hint::Priority::LOW));
/************
  Assume that all the devices (CPU and GPUs) can support all the models.
  Result: compiled_model0 will use GPU.1, compiled_model1 will use GPU.0, compiled_model2 will use CPU.
 ************/

// Example 2
ov::CompiledModel compiled_model3 = core.compile_model(model, "AUTO",
    ov::hint::model_priority(ov::hint::Priority::LOW));
ov::CompiledModel compiled_model4 = core.compile_model(model, "AUTO",
    ov::hint::model_priority(ov::hint::Priority::MEDIUM));
ov::CompiledModel compiled_model5 = core.compile_model(model, "AUTO",
    ov::hint::model_priority(ov::hint::Priority::LOW));
/************
  Assume that all the devices (CPU and GPUs) can support all the models.
  Result: compiled_model3 will use GPU.1, compiled_model4 will use GPU.1, compiled_model5 will use GPU.0.
 ************/
//! [part4]
}
    return 0;
}
