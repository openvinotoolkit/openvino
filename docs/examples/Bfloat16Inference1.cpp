#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
InferenceEngine::Core core;

auto exeNetwork = core.LoadNetwork(network, "CPU");

auto enforceBF16 = exeNetwork.GetConfig(PluginConfigParams::KEY_ENFORCE_BF16).as<std::string>();

return 0;
}
