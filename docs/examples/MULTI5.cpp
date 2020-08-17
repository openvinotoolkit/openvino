#include <inference_engine.hpp>
#include <multi/multi_device_config.hpp>


int main() {
using namespace InferenceEngine;
// 'device_name' can be "MULTI:HDDL,GPU" to configure the multi-device to use HDDL and GPU

ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, device_name, full_config);

// new metric allows to query the optimal number of requests:

uint32_t nireq = exeNetwork.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();

return 0;
}
