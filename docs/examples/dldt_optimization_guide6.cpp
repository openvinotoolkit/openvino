#include <inference_engine.hpp>
#include "ie_plugin_config.hpp"

#include "hetero/hetero_plugin_config.hpp"


int main() {
using namespace InferenceEngine;
Core ie;

auto network = ie.ReadNetwork("Model.xml", "Model.bin");

InferenceEngine::InputsDataMap input_info(network.getInputsInfo());



auto executable_network = ie.LoadNetwork(network, "GPU");

auto infer_request = executable_network.CreateInferRequest();



for (auto & item : inputInfo) {

	std::string input_name = item->first;

	auto input = infer_request.GetBlob(input_name);

	/** Lock/Fill input tensor with data **/

		   unsigned char* data =

	input->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();

// 	...

}



infer_request->Infer();

return 0;
}
