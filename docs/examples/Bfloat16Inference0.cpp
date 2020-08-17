#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
InferenceEngine::Core core;

auto cpuOptimizationCapabilities = core.GetMetric("CPU", METRIC_KEY(OPTIMIZATION_CAPABILITIES)).as<std::vector<std::string>>();

return 0;
}
