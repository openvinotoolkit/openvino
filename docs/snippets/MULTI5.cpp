#include <ie_core.hpp>

int main() {
std::string device_name = "MULTI:HDDL,GPU";
const std::map< std::string, std::string > full_config = {};
//! [part5]
InferenceEngine::Core ie; 
InferenceEngine::CNNNetwork cnnNetwork = ie.ReadNetwork("sample.xml");
// 'device_name' can be "MULTI:HDDL,GPU" to configure the multi-device to use HDDL and GPU
InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, device_name, full_config);
// new metric allows to query the optimal number of requests:
uint32_t nireq = exeNetwork.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
//! [part5]
return 0;
}
