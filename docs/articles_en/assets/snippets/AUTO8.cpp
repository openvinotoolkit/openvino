#include <openvino/openvino.hpp>
#include <openvino/runtime/auto/properties.hpp>

int main() {
//! [part8]
ov::Core core;

// read a network in IR, PaddlePaddle, or ONNX format
std::shared_ptr<ov::Model> model = core.read_model("sample.xml");
// compile a model on AUTO and set utilization threshold
std::map<std::string, double> utilization_threshold = {{"CPU", 78.5}, {"GPU", 55}};
ov::CompiledModel compiled_model = core.compile_model(model, "AUTO",
    {ov::intel_auto::devices_utilization_threshold(utilization_threshold)});
//! [part8]
    return 0;
}
