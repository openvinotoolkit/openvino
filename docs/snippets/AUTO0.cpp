#include <ie_core.hpp>

int main() {
//! [part0]
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network = ie.ReadNetwork("sample.xml");
    //NEW IE-CENTRIC API, the "auto" plugin is (globally) pre-configured with the explicit option:
    InferenceEngine::ExecutableNetwork exec0 = ie.LoadNetwork(network, "AUTO");
//! [part0]
return 0;
}
