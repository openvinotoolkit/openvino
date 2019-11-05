#include "ie_plugin.h"
#include "helpers.h"

InferenceEngineBridge::IEPlugin::IEPlugin(const std::string &device, const std::vector<std::string> &plugin_dirs) {
    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::PluginDispatcher dispatcher{plugin_dirs};
    actual = dispatcher.getPluginByDevice(device);
    IE_SUPPRESS_DEPRECATED_END
    auto pluginVersion = actual.GetVersion();
    version = std::to_string(pluginVersion->apiVersion.major) + ".";
    version += std::to_string(pluginVersion->apiVersion.minor) + ".";
    version += pluginVersion->buildNumber;
    device_name = device;
}

void InferenceEngineBridge::IEPlugin::setInitialAffinity(const InferenceEngineBridge::IENetwork &net) {
    InferenceEngine::InferenceEnginePluginPtr hetero_plugin(actual);
    InferenceEngine::QueryNetworkResult queryRes;
    auto &network = net.actual;

    hetero_plugin->QueryNetwork(network, {}, queryRes);

    if (queryRes.rc != InferenceEngine::StatusCode::OK) {
        THROW_IE_EXCEPTION << queryRes.resp.msg;
    }

    for (auto &&layer : queryRes.supportedLayersMap) {
        network.getLayerByName(layer.first.c_str())->affinity = layer.second;
    }
}

std::set<std::string> InferenceEngineBridge::IEPlugin::queryNetwork(const InferenceEngineBridge::IENetwork &net) {
    const InferenceEngine::CNNNetwork &network = net.actual;
    InferenceEngine::QueryNetworkResult queryRes;
    actual.QueryNetwork(network, {}, queryRes);

    std::set<std::string> supportedLayers;
    for (auto &&layer : queryRes.supportedLayersMap) {
        supportedLayers.insert(layer.first);
    }

    return supportedLayers;
}

std::unique_ptr<InferenceEngineBridge::IEExecNetwork>
InferenceEngineBridge::IEPlugin::load(const InferenceEngineBridge::IENetwork &net,
                                      int num_requests,
                                      const std::map<std::string, std::string> &config) {
    InferenceEngine::ResponseDesc response;
    auto exec_network = InferenceEngineBridge::make_unique<InferenceEngineBridge::IEExecNetwork>(net.name,
                                                                                                 num_requests);
    exec_network->exec_network_ptr = actual.LoadNetwork(net.actual, config);

    if (0 == num_requests) {
        num_requests = getOptimalNumberOfRequests(exec_network->exec_network_ptr);
        exec_network->infer_requests.resize(num_requests);
    }

    for (size_t i = 0; i < num_requests; ++i) {
        InferenceEngineBridge::InferRequestWrap &infer_request = exec_network->infer_requests[i];
        IE_CHECK_CALL(exec_network->exec_network_ptr->CreateInferRequest(infer_request.request_ptr, &response))
    }

    return exec_network;
}

void InferenceEngineBridge::IEPlugin::setConfig(const std::map<std::string, std::string> &config) {
    actual.SetConfig(config);
}

void InferenceEngineBridge::IEPlugin::addCpuExtension(const std::string &extension_path) {
    auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(extension_path);
    auto extension = std::dynamic_pointer_cast<InferenceEngine::IExtension>(extension_ptr);
    actual.AddExtension(extension);
}