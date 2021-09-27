#include <ie_core.hpp>

int main() {
    const std::map<std::string, std::string> cpu_config  = { { InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES } };
    const std::map<std::string, std::string> gpu_config = { { InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES } };
    //! [part4]
    InferenceEngine::Core ie; 
    InferenceEngine::CNNNetwork network = ie.ReadNetwork("sample.xml");
    // configure the CPU device first
    ie.SetConfig(cpu_config, "CPU"); 
    // configure the GPU device
    ie.SetConfig(gpu_config, "GPU"); 
    // load the network to the auto-device
    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(network, "AUTO");
    // new metric allows to query the optimization capabilities
    std::vector<std::string> device_cap = exeNetwork.GetMetric(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
    //! [part4]
    return 0;
}
