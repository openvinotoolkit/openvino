#include <ie_core.hpp>

int main() {
const std::map<std::string, std::string> hddl_config  = { { InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES } };
const std::map<std::string, std::string> gpu_config = { { InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES } };
//! [part4]
// configure the HDDL device first
InferenceEngine::Core ie; 
InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork("sample.xml");
ie.SetConfig(hddl_config, "HDDL"); 
// configure the GPU device
ie.SetConfig(gpu_config, "GPU"); 
// load the network to the multi-device, while specifying the configuration (devices along with priorities):
InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, "MULTI", {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, "HDDL,GPU"}});
// new metric allows to query the optimal number of requests:
uint32_t nireq = exeNetwork.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
//! [part4]
return 0;
}
