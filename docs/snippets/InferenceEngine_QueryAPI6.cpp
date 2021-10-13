#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part6]
InferenceEngine::Core core;
auto network = core.ReadNetwork("sample.xml");
auto exeNetwork = core.LoadNetwork(network, "GPU");
std::map<std::string, uint64_t> statistics_map = exeNetwork.GetMetric(GPU_METRIC_KEY(MEMORY_STATISTICS));
//! [part6]
return 0;
}
