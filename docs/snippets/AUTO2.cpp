#include <ie_core.hpp>

int main() {
//! [part2]
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network = ie.ReadNetwork("sample.xml");
    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(network, "AUTO");
//! [part2]
return 0;
}
