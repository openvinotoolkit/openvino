#include <ie_core.hpp>

int main() {
//! [part1]
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network = ie.ReadNetwork("sample.xml");
    // "AUTO" plugin is (globally) pre-configured with the explicit option:
    ie.SetConfig({{"AUTO_DEVICE_LIST", "CPU,GPU"}}, "AUTO");
    // the below 3 lines are equivalent (the first line leverages the pre-configured AUTO, while second and third explicitly pass the same settings)
    InferenceEngine::ExecutableNetwork exec0 = ie.LoadNetwork(network, "AUTO", {});
    InferenceEngine::ExecutableNetwork exec1 = ie.LoadNetwork(network, "AUTO", {{"AUTO_DEVICE_LIST", "CPU,GPU"}});
    InferenceEngine::ExecutableNetwork exec2 = ie.LoadNetwork(network, "AUTO:CPU,GPU");
//! [part1]
return 0;
}
