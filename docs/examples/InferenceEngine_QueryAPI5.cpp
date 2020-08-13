#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
InferenceEngine::Core core;
auto exeNetwork = core.LoadNetwork(network, "CPU");
auto ncores = exeNetwork.GetConfig(PluginConfigParams::KEY_CPU_THREADS_NUM).as<std::string>();
return 0;
}
