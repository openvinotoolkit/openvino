#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
InferenceEngine::Core core;

auto exeNetwork = core.LoadNetwork(network, "CPU");

auto nireq = exeNetwork.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();

return 0;
}
