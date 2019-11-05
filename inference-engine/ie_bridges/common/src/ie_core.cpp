#include "ie_core.h"
#include "helpers.h"

InferenceEngineBridge::IECore::IECore(const std::string &xmlConfigFile) {
    actual = InferenceEngine::Core(xmlConfigFile);
}

std::map<std::string, InferenceEngine::Version>
InferenceEngineBridge::IECore::getVersions(const std::string &deviceName) {
    return actual.GetVersions(deviceName);
}
/*
std::unique_ptr<InferenceEngineBridge::IEExecNetwork>
InferenceEngineBridge::IECore::loadNetwork(InferenceEngineBridge::IENetwork network,
                                           const std::string &deviceName,
                                           const std::map<std::string, std::string> &config,
                                           int &num_requests) {

    InferenceEngine::ResponseDesc response;
    auto exec_network = InferenceEngineBridge::make_unique<InferenceEngineBridge::IEExecNetwork>(network.name,
                                                                               num_requests);
    exec_network->exec_network_ptr = actual.LoadNetwork(network.actual, deviceName, config);

    if (0 == num_requests) {
        num_requests = getOptimalNumberOfRequests(exec_network->exec_network_ptr);
        exec_network->infer_requests.resize(num_requests);
    }

    for (size_t i = 0; i < num_requests; ++i) {
        InferRequestWrap &infer_request = exec_network->infer_requests[i];
        IE_CHECK_CALL(exec_network->exec_network_ptr->CreateInferRequest(infer_request.request_ptr, &response))
    }

    return exec_network;
}*/

std::map<std::string, std::string> InferenceEngineBridge::IECore::queryNetwork(InferenceEngineBridge::IENetwork network,
                                                                               const std::string &deviceName,
                                                                               const std::map<std::string, std::string> &config) {
    auto res = actual.QueryNetwork(network.actual, deviceName, config);
    return res.supportedLayersMap;
}

void InferenceEngineBridge::IECore::setConfig(const std::map<std::string, std::string> &config,
                                              const std::string &deviceName) {
    actual.SetConfig(config, deviceName);
}

void InferenceEngineBridge::IECore::registerPlugin(const std::string &pluginName, const std::string &deviceName) {
    actual.RegisterPlugin(pluginName, deviceName);
}

void InferenceEngineBridge::IECore::unregisterPlugin(const std::string &deviceName) {
    actual.UnregisterPlugin(deviceName);
}

void InferenceEngineBridge::IECore::registerPlugins(const std::string &xmlConfigFile) {
    actual.RegisterPlugins(xmlConfigFile);
}

void InferenceEngineBridge::IECore::addExtension(const std::string &ext_lib_path, const std::string &deviceName) {
    auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(ext_lib_path);
    auto extension = std::dynamic_pointer_cast<InferenceEngine::IExtension>(extension_ptr);
    actual.AddExtension(extension, deviceName);
}

std::vector<std::string> InferenceEngineBridge::IECore::getAvailableDevices() {
    return actual.GetAvailableDevices();
}
