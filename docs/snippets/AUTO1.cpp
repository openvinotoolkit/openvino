#include <ie_core.hpp>

int main() {
//! [part1]
	/*********************
	 * With API Prior to 2022.1 Release
	 *********************/
	InferenceEngine::Core ie;

	// Read a network in IR, PaddlePaddle, or ONNX format:
	InferenceEngine::CNNNetwork network = ie.ReadNetwork("sample.xml");

	// Load a network to AUTO using the default list of device candidates.
	// The following lines are equivalent:
	InferenceEngine::ExecutableNetwork exec0 = ie.LoadNetwork(network);
	InferenceEngine::ExecutableNetwork exec1 = ie.LoadNetwork(network, "AUTO");
	InferenceEngine::ExecutableNetwork exec2 = ie.LoadNetwork(network, "AUTO", {});

	// You can also specify the devices to be used by AUTO in its selection process.
	// The following lines are equivalent:
	InferenceEngine::ExecutableNetwork exec3 = ie.LoadNetwork(network, "AUTO:GPU,CPU");
	InferenceEngine::ExecutableNetwork exec4 = ie.LoadNetwork(network, "AUTO", {{"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}});

	// the AUTO plugin is pre-configured (globally) with the explicit option:
	ie.SetConfig({{"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}, "AUTO");
//! [part1]
	return 0;
}
