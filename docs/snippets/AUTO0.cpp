#include <ie_core.hpp>

int main() {
//! [part0]
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network = ie.ReadNetwork("sample.xml");
    // these 2 lines below are equivalent
    InferenceEngine::ExecutableNetwork exec0 = ie.LoadNetwork(network, "AUTO");
    InferenceEngine::ExecutableNetwork exec1 = ie.LoadNetwork(network, "");
//! [part0]
return 0;
}
