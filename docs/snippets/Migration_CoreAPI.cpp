#include <ie_core.hpp>

int main() {
std::string deviceName = "Device name";
//! [part0]
InferenceEngine::InferencePlugin plugin = InferenceEngine::PluginDispatcher({ FLAGS_pp }).getPluginByDevice(FLAGS_d);
//! [part0]

//! [part1]
InferenceEngine::Core core;
//! [part1]

//! [part2]
InferenceEngine::CNNNetReader network_reader;
network_reader.ReadNetwork(fileNameToString(input_model));
network_reader.ReadWeights(fileNameToString(input_model).substr(0, input_model.size() - 4) + ".bin");
InferenceEngine::CNNNetwork network = network_reader.getNetwork();
//! [part2]

//! [part3]
InferenceEngine::CNNNetwork network = core.ReadNetwork(input_model);
//! [part3]

//! [part4]
InferenceEngine::CNNNetwork network = core.ReadNetwork("model.onnx");
//! [part4]

//! [part5]
plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
//! [part5]

//! [part6]
core.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
//! [part6]

//! [part7]
core.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
//! [part7]

//! [part8]
auto execNetwork = plugin.LoadNetwork(network, { });
//! [part8]

//! [part9]
auto execNetwork = core.LoadNetwork(network, deviceName, { });
//! [part9]
return 0;
}
