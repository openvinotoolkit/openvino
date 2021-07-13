#include <ie_core.hpp>

int main() {
    std::string device_name = "AUTO:CPU,GPU";
    const std::map< std::string, std::string > full_config = {};
    //! [part5]
    InferenceEngine::Core ie; 
    InferenceEngine::CNNNetwork network = ie.ReadNetwork("sample.xml");
    // 'device_name' can be "AUTO:CPU,GPU" to configure the auto-device to use CPU and GPU
    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(network, device_name, full_config);
    // new metric allows to query the optimization capabilities
    std::vector<std::string> device_cap = exeNetwork.GetMetric(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
    //! [part5]
    return 0;
}
