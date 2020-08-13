#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
InferenceEngine::Core core;
std::string cpuDeviceName = core.GetMetric("GPU", METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>();
return 0;
}
