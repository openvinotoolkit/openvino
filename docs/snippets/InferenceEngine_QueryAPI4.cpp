#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part4]
InferenceEngine::Core core;
auto network = core.ReadNetwork("sample.xml");
auto exeNetwork = core.LoadNetwork(network, "MYRIAD");
float temperature = exeNetwork.GetMetric(METRIC_KEY(DEVICE_THERMAL)).as<float>();
//! [part4]
return 0;
}
