#include <inference_engine.hpp>
#include <multi-device/multi_device_config.hpp>


int main() {
using namespace InferenceEngine;
const std::map<std::string, std::string> hddl_config  = { { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } };
const std::map<std::string, std::string> gpu_config = { { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } };
//! [part4]
// configure the HDDL device first
Core ie; 
CNNNetwork cnnNetwork = ie.ReadNetwork("sample.xml");
ie.SetConfig(hddl_config, "HDDL"); 
// configure the GPU device
ie.SetConfig(gpu_config, "GPU"); 
// load the network to the multi-device, while specifying the configuration (devices along with priorities):
ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, "MULTI", {{MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "HDDL,GPU"}});
// new metric allows to query the optimal number of requests:
uint32_t nireq = exeNetwork.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
//! [part4]
return 0;
}
