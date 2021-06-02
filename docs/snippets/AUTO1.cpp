#include <ie_core.hpp>

int main() {
//! [part1]
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network = ie.ReadNetwork("sample.xml");
    //NEW IE-CENTRIC API, the "AUTO" plugin is (globally) pre-configured with the explicit option:
    ie.SetConfig({{"AUTO_DEVICE_LIST", "CPU,GPU"}}, "AUTO");
    InferenceEngine::ExecutableNetwork exec0 = ie.LoadNetwork(network, "AUTO", {});
    InferenceEngine::ExecutableNetwork exec1 = ie.LoadNetwork(network, "AUTO", {{"AUTO_DEVICE_LIST", "CPU,GPU"}});
//! [part1]
return 0;
}
