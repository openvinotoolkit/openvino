#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part5]
InferenceEngine::Core core;
auto network = core.ReadNetwork("sample.xml");
auto exeNetwork = core.LoadNetwork(network, "CPU");
auto ncores = exeNetwork.GetConfig(PluginConfigParams::KEY_CPU_THREADS_NUM).as<std::string>();
//! [part5]
return 0;
}
