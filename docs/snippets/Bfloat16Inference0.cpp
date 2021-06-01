#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part0]
InferenceEngine::Core core;
auto cpuOptimizationCapabilities = core.GetMetric("CPU", METRIC_KEY(OPTIMIZATION_CAPABILITIES)).as<std::vector<std::string>>();
//! [part0]
return 0;
}
