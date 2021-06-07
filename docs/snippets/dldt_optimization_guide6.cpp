#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part6]
InferenceEngine::Core ie;
auto network = ie.ReadNetwork("Model.xml", "Model.bin");
InferenceEngine::InputsDataMap input_info(network.getInputsInfo());

auto executable_network = ie.LoadNetwork(network, "GPU");
auto infer_request = executable_network.CreateInferRequest();

for (auto & item : input_info) {
    std::string input_name = item.first;
    auto input = infer_request.GetBlob(input_name);
    /** Lock/Fill input tensor with data **/
    unsigned char* data = input->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
    // ...
}

infer_request.Infer();
//! [part6]
return 0;
}
