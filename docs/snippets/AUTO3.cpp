#include <ie_core.hpp>

int main() {
//! [part3]
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network = ie.ReadNetwork("sample.xml");
    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(network, "AUTO:CPU,GPU");
//! [part3]
return 0;
}
