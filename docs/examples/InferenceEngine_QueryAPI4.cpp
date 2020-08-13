#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
InferenceEngine::Core core;
auto exeNetwork = core.LoadNetwork(network, "MYRIAD");
float temperature = exeNetwork.GetMetric(METRIC_KEY(DEVICE_THERMAL)).as<float>();
return 0;
}
