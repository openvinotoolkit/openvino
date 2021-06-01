#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part1]
InferenceEngine::Core core;
auto network = core.ReadNetwork("sample.xml");
auto exeNetwork = core.LoadNetwork(network, "CPU");
auto enforceBF16 = exeNetwork.GetConfig(PluginConfigParams::KEY_ENFORCE_BF16).as<std::string>();
//! [part1]

return 0;
}
