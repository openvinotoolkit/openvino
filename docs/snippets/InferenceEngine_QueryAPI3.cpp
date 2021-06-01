#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part3]
InferenceEngine::Core core;
auto network = core.ReadNetwork("sample.xml");
auto exeNetwork = core.LoadNetwork(network, "CPU");
auto nireq = exeNetwork.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
//! [part3]
return 0;
}
