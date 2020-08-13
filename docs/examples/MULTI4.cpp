#include <inference_engine.hpp>
#include <multi/multi_device_config.hpp>

int main() {
using namespace InferenceEngine;
// configure the HDDL device first
Core ie; 
ie.SetConfig(hddl_config, "HDDL"); 
// configure the GPU device
ie.SetConfig(gpu_config, "GPU"); 
// load the network to the multi-device, while specifying the configuration (devices along with priorities):
ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, "MULTI", {{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "HDDL,GPU"}});
// new metric allows to query the optimal number of requests:
uint32_t nireq = exeNetwork.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
return 0;
}
