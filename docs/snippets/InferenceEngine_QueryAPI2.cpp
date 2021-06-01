#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part2]
InferenceEngine::Core core;
std::string cpuDeviceName = core.GetMetric("GPU", METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>();
//! [part2]
return 0;
}
