// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <utility>
#include <map>
#include <unordered_map>

#include "ie_metric_helpers.hpp"
#include <ie_api.h>
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include <cpp_interfaces/base/ie_infer_async_request_base.hpp>
#include <multi-device/multi_device_config.hpp>
#include "multi_device.hpp"

namespace MultiDevicePlugin {
    using namespace InferenceEngine;
// ------------------------------MultiDeviceInferRequest----------------------------
MultiDeviceInferRequest::MultiDeviceInferRequest(const InputsDataMap&   networkInputs,
                                                 const OutputsDataMap&  networkOutputs)
        : InferRequestInternal(networkInputs, networkOutputs) {
    // Allocate all input blobs
    for (const auto &it : networkInputs) {
        Layout l = it.second->getLayout();
        Precision p = it.second->getPrecision();
        SizeVector dims = it.second->getTensorDesc().getDims();

        TensorDesc desc = TensorDesc(p, dims, l);
        _inputs[it.first] = make_blob_with_precision(desc);
        _inputs[it.first]->allocate();
    }
    // Allocate all output blobs
    for (const auto &it : networkOutputs) {
        Layout l = it.second->getLayout();
        Precision p = it.second->getPrecision();
        SizeVector dims = it.second->getTensorDesc().getDims();

        TensorDesc desc = TensorDesc(p, dims, l);
        _outputs[it.first] = make_blob_with_precision(desc);
        _outputs[it.first]->allocate();
    }
}

void MultiDeviceInferRequest::SetBlobsToAnotherRequest(InferRequest& req) {
    for (const auto &it : _networkInputs) {
        Blob::Ptr blob;
        auto &name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        GetBlob(name.c_str(), blob);
        req.SetBlob(name.c_str(), blob);
    }
    for (const auto &it : _networkOutputs) {
        Blob::Ptr blob;
        auto &name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        GetBlob(name.c_str(), blob);
        req.SetBlob(name.c_str(), blob);
    }
}

MultiDeviceAsyncInferRequest::MultiDeviceAsyncInferRequest(
    const MultiDeviceInferRequest::Ptr&         inferRequest,
    const bool                                  needPerfCounters,
    const MultiDeviceExecutableNetwork::Ptr&    multiDeviceExecutableNetwork,
    const ITaskExecutor::Ptr&                   callbackExecutor) :
    AsyncInferRequestThreadSafeDefault(inferRequest, nullptr, callbackExecutor),
    _multiDeviceExecutableNetwork{multiDeviceExecutableNetwork},
    _inferRequest{inferRequest},
    _needPerfCounters{needPerfCounters} {
    struct ThisRequestExecutor : public ITaskExecutor {
        explicit ThisRequestExecutor(MultiDeviceAsyncInferRequest* _this_) : _this{_this_} {}
        void run(Task task) override {
            auto workerInferRequest = _this->_workerInferRequest;
            workerInferRequest->_task = std::move(task);
            workerInferRequest->_inferRequest.StartAsync();
        };
        MultiDeviceAsyncInferRequest* _this = nullptr;
    };
    _pipeline = {
        {_multiDeviceExecutableNetwork, [this] {
            _workerInferRequest = MultiDeviceExecutableNetwork::_thisWorkerInferRequest;
            _inferRequest->SetBlobsToAnotherRequest(_workerInferRequest->_inferRequest);
        }},
        {std::make_shared<ThisRequestExecutor>(this), [this] {
            auto status = _workerInferRequest->_status;
            if (InferenceEngine::StatusCode::OK != status) {
                if (nullptr != InferenceEngine::CurrentException()) {
                    std::rethrow_exception(InferenceEngine::CurrentException());
                } else {
                    THROW_IE_EXCEPTION << InferenceEngine::details::as_status << status;
                }
            }
            if (_needPerfCounters) {
                _perfMap = _workerInferRequest->_inferRequest.GetPerformanceCounts();
            }
        }}
    };
}

void MultiDeviceAsyncInferRequest::Infer_ThreadUnsafe() {
    InferUsingAsync();
}

void MultiDeviceAsyncInferRequest::GetPerformanceCounts_ThreadUnsafe(std::map<std::string, InferenceEngineProfileInfo> &perfMap) const {
    perfMap = std::move(_perfMap);
}

MultiDeviceAsyncInferRequest::~MultiDeviceAsyncInferRequest() {
    StopAndWait();
}

// ------------------------------MultiDeviceExecutableNetwork----------------------------

thread_local MultiDeviceExecutableNetwork::WorkerInferRequest* MultiDeviceExecutableNetwork::_thisWorkerInferRequest = nullptr;

struct IdleGuard {
    explicit IdleGuard(MultiDeviceExecutableNetwork::WorkerInferRequest* workerInferRequestPtr,
                       MultiDeviceExecutableNetwork::NotBusyWorkerRequests& notBusyWorkerRequests) :
        _workerInferRequestPtr{workerInferRequestPtr},
        _notBusyWorkerRequests{&notBusyWorkerRequests} {
    }
    ~IdleGuard() {
        if (nullptr != _notBusyWorkerRequests) {
            _notBusyWorkerRequests->push(_workerInferRequestPtr);
        }
    }
    MultiDeviceExecutableNetwork::NotBusyWorkerRequests* Release() {
        auto notBusyWorkerRequests = _notBusyWorkerRequests;
        _notBusyWorkerRequests = nullptr;
        return notBusyWorkerRequests;
    }
    MultiDeviceExecutableNetwork::WorkerInferRequest*     _workerInferRequestPtr = nullptr;
    MultiDeviceExecutableNetwork::NotBusyWorkerRequests*  _notBusyWorkerRequests = nullptr;
};

MultiDeviceExecutableNetwork::MultiDeviceExecutableNetwork(const DeviceMap<InferenceEngine::ExecutableNetwork>&                 networksPerDevice,
                                                           const DeviceMap<unsigned int>&                                       numRequestsPerDevices,
                                                           const std::vector<DeviceName>&                                       networkDevices,
                                                           const std::unordered_map<std::string, InferenceEngine::Parameter>&   config,
                                                           const bool                                                           needPerfCounters) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault(nullptr, std::make_shared<InferenceEngine::ImmediateExecutor>()),
    _devicePriorities{networkDevices},
    _networksPerDevice{networksPerDevice},
    _config{config},
    _needPerfCounters{needPerfCounters} {
    _taskExecutor.reset();
    for (auto&& networkValue : _networksPerDevice) {
        auto& device  = networkValue.first;
        auto& network = networkValue.second;

        auto itNumRequests = numRequestsPerDevices.find(device);
        unsigned int optimalNum = 0;
        try {
            optimalNum = network.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
        } catch (const details::InferenceEngineException &iie) {
            THROW_IE_EXCEPTION
                    << "Every device used with the Multi-Device should "
                    << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                    << "Failed to query the metric for the " << device << " with error:" << iie.what();
        }
        const auto numRequests = (numRequestsPerDevices.end() == itNumRequests) ? optimalNum : itNumRequests->second;
        auto& workerRequests = _workerRequests[device];
        auto& idleWorkerRequests = _idleWorkerRequests[device];
        workerRequests.resize(numRequests);
        auto* idleWorkerRequestsPtr = &(idleWorkerRequests);
        for (auto&& workerRequest : workerRequests) {
            workerRequest._inferRequest = network.CreateInferRequest();
            auto* workerRequestPtr = &workerRequest;
            idleWorkerRequests.push(workerRequestPtr);
            workerRequest._inferRequest.SetCompletionCallback<std::function<void(InferRequest, StatusCode)>>(
                [workerRequestPtr, this, device, idleWorkerRequestsPtr] (InferRequest , StatusCode status) mutable {
                    IdleGuard idleGuard{workerRequestPtr, *idleWorkerRequestsPtr};
                    workerRequestPtr->_status = status;
                    {
                        auto capturedTask = std::move(workerRequestPtr->_task);
                        capturedTask();
                    }
                    if (!_terminate) {
                        idleGuard.Release()->push(workerRequestPtr);
                        ScheduleToWorkerInferRequest();
                    }
                });
        }
    }
}

void MultiDeviceExecutableNetwork::ScheduleToWorkerInferRequest() {
    auto devices = [&] {
        std::lock_guard<std::mutex> lock(_mutex);
        return _devicePriorities;
    }();
    for (auto&& device : devices) {
        auto& idleWorkerRequests = _idleWorkerRequests[device];
        WorkerInferRequest* workerRequestPtr = nullptr;
        if (idleWorkerRequests.try_pop(workerRequestPtr)) {
            IdleGuard idleGuard{workerRequestPtr, idleWorkerRequests};
            Task inferPipelineTask;
            if (_inferPipelineTasks.try_pop(inferPipelineTask)) {
                _thisWorkerInferRequest = workerRequestPtr;
                inferPipelineTask();
                idleGuard.Release();
                break;
            }
        }
    }
}

void MultiDeviceExecutableNetwork::run(Task inferPipelineTask) {
    if (!_terminate) {
        _inferPipelineTasks.push(std::move(inferPipelineTask));
        ScheduleToWorkerInferRequest();
    }
}

MultiDeviceExecutableNetwork::~MultiDeviceExecutableNetwork() {
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _devicePriorities.clear();
    }
    _terminate = true;
    /* NOTE: The only threads that use `MultiDeviceExecutableNetwork` Context are thous that are used by Worker infer requests.
     *       But AsyncInferRequest destructor should waits for all asynchronous tasks that are used by the request
     */
    _workerRequests.clear();
}

InferenceEngine::InferRequestInternal::Ptr MultiDeviceExecutableNetwork::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                                                InferenceEngine::OutputsDataMap networkOutputs) {
    return std::make_shared<MultiDeviceInferRequest>(networkInputs, networkOutputs);
}

void MultiDeviceExecutableNetwork::CreateInferRequest(IInferRequest::Ptr& asyncRequest) {
    auto syncRequestImpl = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
    auto asyncTreadSafeImpl = std::make_shared<MultiDeviceAsyncInferRequest>(std::static_pointer_cast<MultiDeviceInferRequest>(syncRequestImpl),
                                                                             _needPerfCounters,
                                                                             std::static_pointer_cast<MultiDeviceExecutableNetwork>(shared_from_this()),
                                                                             _callbackExecutor);
    asyncRequest.reset(new InferRequestBase<MultiDeviceAsyncInferRequest>(asyncTreadSafeImpl), [](IInferRequest *p) { p->Release(); });
    asyncTreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
}

void MultiDeviceExecutableNetwork::SetConfig(const std::map<std::string, InferenceEngine::Parameter> &config,
        InferenceEngine::ResponseDesc * /* resp */) {
    auto res = config.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES);
    if (res == config.end() || config.size() > 1) {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str <<
            "The only config supported for the Network's SetConfig is MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES";
    } else {
        DeviceMap<unsigned int> newNumRequestsPerDevice;
        std::vector<DeviceName> newDevices;
        MultiDeviceInferencePlugin::ParseDevicesAndNumRequests(res->second, newNumRequestsPerDevice, newDevices);

        if (!newNumRequestsPerDevice.empty()) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str << "You can only change device priorities but not number of requests"
                     <<" with the Network's SetConfig(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES!";
        }

        {
            std::lock_guard<std::mutex> lock{_mutex};
            for (auto device : newDevices) {
                if (_devicePriorities.end() == std::find(std::begin(_devicePriorities), std::end(_devicePriorities), device)) {
                    THROW_IE_EXCEPTION << NOT_FOUND_str << "You can only change device priorities but not add new devices with"
                        << " the Network's SetConfig(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES." << device <<
                            " device was not in the original device list!";
                }
            }
            _devicePriorities = newDevices;
        }
    }
}

void MultiDeviceExecutableNetwork::GetConfig(const std::string &name, InferenceEngine::Parameter &result,
        InferenceEngine::ResponseDesc * /* resp */) const {
    auto res = _config.find(name);
    if (res != _config.end()) {
        result =  res->second;
    } else {
        THROW_IE_EXCEPTION << NOT_FOUND_str << name <<" not found in the ExecutableNetwork config";
    }
}

void MultiDeviceExecutableNetwork::GetMetric(const std::string &name, Parameter &result, ResponseDesc *resp) const {
    if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        unsigned int res = 0u;
        for (auto n : _networksPerDevice) {
            try {
                res += n.second.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
            } catch (const details::InferenceEngineException &iie) {
                  THROW_IE_EXCEPTION
                        << "Every device used with the Multi-Device should "
                        << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                        << "Failed to query the metric for the " << n.first << " with error:" << iie.what();
           }
        }
        result = IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, res);
    } else if (name == METRIC_KEY(NETWORK_NAME)) {
        auto it = _networksPerDevice.begin();
        IE_ASSERT(it != _networksPerDevice.end());
        result = IE_SET_METRIC(NETWORK_NAME, it->second.GetMetric(
            METRIC_KEY(NETWORK_NAME)).as<std::string>());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        result = IE_SET_METRIC(SUPPORTED_METRICS, {
            METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(NETWORK_NAME),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS)
        });
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys = { MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES };
        result = IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, configKeys);
    } else {
        THROW_IE_EXCEPTION << "Unsupported Network metric: " << name;
    }
}

// ------------------------------MultiDeviceInferencePlugin----------------------------
void MultiDeviceInferencePlugin::ParseDevicesAndNumRequests(const std::string&          priorities,
                                                            DeviceMap<unsigned int>&    requests,
                                                            std::vector<DeviceName>&    parsedDevices) {
    // parsing the string and splitting to tokens
    std::vector<std::string> devicesWithRequests;
    // parsing the string and splitting the comma-separated tokens
    std::string::size_type i = 0;
    std::string::size_type idelimeter;
    while ((idelimeter = priorities.find(',', i)) != std::string::npos) {
        devicesWithRequests.push_back(priorities.substr(i, idelimeter - i));
        i = idelimeter + 1;
    }
    // last token in the string (which has no comma after that)
    devicesWithRequests.push_back(priorities.substr(i, priorities.length() - i));

    for (auto&& d : devicesWithRequests) {
        auto openingBracket = d.find_first_of('(');
        auto closingBracket = d.find_first_of(')', openingBracket);
        auto device_name = d.substr(0, openingBracket);
        if (closingBracket != std::string::npos && openingBracket < closingBracket)
            requests[device_name] = std::stol(d.substr(openingBracket + 1, closingBracket - 1));
        parsedDevices.push_back(device_name);
    }
}


IE_SUPPRESS_DEPRECATED_START

DeviceMap<InferenceEngine::InferencePlugin> MultiDeviceInferencePlugin::LoadDevices(const std::vector<DeviceName>& parsedDevices) const {
    DeviceMap<InferenceEngine::InferencePlugin> plugins;
    for (auto&& deviceName : parsedDevices) {
        auto itPlugin = _plugins.find(deviceName);
        if (itPlugin == _plugins.end()) {
            InferencePlugin plugin;
            DeviceIDParser device(deviceName);
            if (_core != nullptr) {
                plugin = InferencePlugin(_core->GetPluginByName(device.getDeviceName()));
            } else {
                IE_SUPPRESS_DEPRECATED_START
                // try to create plugin
                PluginDispatcher dispatcher({file_name_t()});
                plugin = dispatcher.getPluginByDevice(device.getDeviceName());
                IE_SUPPRESS_DEPRECATED_END
            }
            plugins.emplace(deviceName, plugin);

            try {
                for (auto e : _extensions)
                    plugin.AddExtension(e);
            } catch (InferenceEngine::details::InferenceEngineException &) { }
        } else {
            plugins.emplace(deviceName, itPlugin->second);
        }
    }
    return plugins;
}

void MultiDeviceInferencePlugin::LoadDevices(const std::vector<DeviceName>& parsedDevices) {
    _plugins = const_cast<const MultiDeviceInferencePlugin*>(this)->LoadDevices(parsedDevices);
}

IE_SUPPRESS_DEPRECATED_START

void MultiDeviceInferencePlugin::SetConfig(const std::map<std::string, std::string> & config) {
    _config.insert(config.begin(), config.end());
    auto priorities = config.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES);
    if (priorities != config.end()) {
        // since this is plugin's SetConfig, the requests data for the devices is stored in the data member
        ParseDevicesAndNumRequests(priorities->second, _numRequestsPerDevices, _devicePriorities);
        LoadDevices(_devicePriorities);
    }
    // rest of configuration
    ParseConfigForDevices({config.begin(), config.end()}, _plugins);
}

void MultiDeviceInferencePlugin::AddExtension(IExtensionPtr extension) {
    _extensions.emplace_back(extension);
    try {
        for (auto&& plugin : _plugins) {
            plugin.second.AddExtension(extension);
        }
    } catch (InferenceEngine::details::InferenceEngineException &) {}
}

INFERENCE_PLUGIN_API(InferenceEngine::StatusCode) CreatePluginEngine(
        InferenceEngine::IInferencePlugin *&plugin,
        InferenceEngine::ResponseDesc *resp) noexcept {
    try {
        plugin = make_ie_compatible_plugin(
                {{2, 1},
                 CI_BUILD_NUMBER,
                 "MultiDevicePlugin"}, std::make_shared<MultiDeviceInferencePlugin>());
        return OK;
    }
    catch (std::exception &ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}

IE_SUPPRESS_DEPRECATED_END

MultiDeviceInferencePlugin::MultiDeviceInferencePlugin() {
    _pluginName = "MULTI";
}

InferenceEngine::Parameter MultiDeviceInferencePlugin::GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter> & options) const {
    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        std::string name = { "MULTI" };
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, name);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys = { MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES    };
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else {
        THROW_IE_EXCEPTION << "Unsupported metric key " << name;
    }
}

namespace  {

IInferencePluginAPI * getInferencePluginAPIInterface(IInferencePlugin * iplugin) {
    return dynamic_cast<IInferencePluginAPI *>(iplugin);
}

IInferencePluginAPI * getInferencePluginAPIInterface(InferenceEnginePluginPtr iplugin) {
    return getInferencePluginAPIInterface(static_cast<IInferencePlugin *>(iplugin.operator->()));
}

IE_SUPPRESS_DEPRECATED_START

IInferencePluginAPI * getInferencePluginAPIInterface(InferencePlugin plugin) {
    return getInferencePluginAPIInterface(static_cast<InferenceEnginePluginPtr>(plugin));
}

IE_SUPPRESS_DEPRECATED_END

}  // namespace

DeviceMap<std::map<std::string, std::string>>
MultiDeviceInferencePlugin::ParseConfigForDevices(const std::unordered_map<std::string, std::string>&   fullConfig,
                                                  DeviceMap<InferenceEngine::InferencePlugin>&          plugins) const {
    // preparing local version of configs supported by plugins
    DeviceMap<std::map<std::string, std::string>> pluginsConfig;

    // MULTI device is created using Core API
    if (GetCore()) {
        auto unsupportedConfig = fullConfig;

        for (auto p : plugins) {
            DeviceIDParser device(p.first);
            if (!device.getDeviceID().empty()) {  // if device name came with the deviceID, insert the corresponding config
                pluginsConfig[p.first].emplace(PluginConfigParams::KEY_DEVICE_ID, device.getDeviceID());
            }
            try {
                InferenceEnginePluginPtr pluginPublicAPI = GetCore()->GetPluginByName(device.getDeviceName());
                auto pluginPrivateAPI = getInferencePluginAPIInterface(pluginPublicAPI);
                std::vector<std::string> supportedConfigKeys = pluginPrivateAPI->GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), { });

                auto removeKey = [&unsupportedConfig](const std::string & key) {
                    auto it = unsupportedConfig.find(key);
                    if (it != unsupportedConfig.end()) {
                        unsupportedConfig.erase(it);
                    }
                };

                for (auto&& c : fullConfig) {
                    // skip Multi-Device specific setting(s)
                    if (MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES == c.first) {
                        removeKey(c.first);
                        continue;
                    }
                    for (auto && configKey : supportedConfigKeys) {
                        if (c.first.compare(configKey) == 0) {
                            pluginsConfig[p.first][c.first] = c.second;
                            removeKey(c.first);
                        }
                    }
                }
            } catch (const details::InferenceEngineException &ie) {
                THROW_IE_EXCEPTION << "Failed to query metric for " << p.first << " with error:" << ie.what();
            }
        }

        if (!unsupportedConfig.empty()) {
            auto unsupportedValue = unsupportedConfig.begin();
            THROW_IE_EXCEPTION << NOT_FOUND_str << "None plugin loaded to the Multi-Device"
                                                   " reported the following property as supported: " << unsupportedValue->second <<
                               " for the " << unsupportedValue->first << ".";
        }
    } else {
        // every device will just ignore unsupported config values
        for (auto&& c : fullConfig) {
            size_t num_refused_plugins = 0;
            // skip Multi-Device specific setting(s)
            if (MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES == c.first)
                continue;
            for (auto p : plugins) {
                std::map<std::string, std::string> oneconfig;
                oneconfig[c.first] = c.second;
                try {
                    p.second.SetConfig(oneconfig);
                    // if no exception is thrown, then the config was accepted, thus let's save it
                    pluginsConfig[p.first].insert(oneconfig.begin(), oneconfig.end());
                } catch (details::InferenceEngineException&) {
                    num_refused_plugins++;
                }
            }

            // if any plugin is loaded, and none plugin has recognized the specific config, then it is exception to report
            if (plugins.size() && plugins.size() == num_refused_plugins) {
                THROW_IE_EXCEPTION << NOT_FOUND_str << "None plugin loaded to the Multi-Device"
                                                       " reported the following property as supported: " << c.second <<
                                   " for the " << c.first << ".";
            }
        }
    }

    return pluginsConfig;
}

ExecutableNetworkInternal::Ptr MultiDeviceInferencePlugin::LoadExeNetworkImpl(const ICore * core, const ICNNNetwork &network,
                                                                              const std::map<std::string, std::string>& config) {
    auto clonedNetwork = cloneNet(network);

    // check if user has set any priorities/num_requests different from the plugin's default (e.g. via SetConfig)
    auto priorities = config.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES);
    DeviceMap<unsigned int> numRequestsPerDevices;
    std::vector<DeviceName> devicePriorities;
    if (priorities != config.end()) {
        // since this is per network's config, the #requests data, priorities, etc are stored just locally (only for the network)
        ParseDevicesAndNumRequests(priorities->second, numRequestsPerDevices, devicePriorities);
        LoadDevices(devicePriorities);
    }
    if (_plugins.empty())
        THROW_IE_EXCEPTION << NOT_FOUND_str << "No plugin is loaded to the Multi-Device. Failed to load the network";

    // configuration that has just arrived (which has the priority and thus is inserted first) + original plugin config
    std::unordered_map<std::string, std::string> fullConfig;
    fullConfig.insert(config.begin(), config.end());
    fullConfig.insert(_config.begin(), _config.end());

    // collect the settings that are applicable to the devices we are loading the network to
    std::unordered_map<std::string, InferenceEngine::Parameter> multiNetworkConfig;

    // preparing local version of configs supported by plugins
    auto pluginsConfig = ParseConfigForDevices(fullConfig, _plugins);
    DeviceMap<ExecutableNetwork> executableNetworkPerDevice;
    for (auto& p : _plugins) {
        if (devicePriorities.empty() /*all devices that were loaded so far*/ ||
            devicePriorities.end() != std::find(devicePriorities.begin(), devicePriorities.end(), p.first)) {
            // need a local copy, as the device id shouldn't be stored as a part of the multiNetworkConfig
            auto configWithDeviceId = pluginsConfig[p.first];
            DeviceIDParser device(p.first);
            if (!device.getDeviceID().empty()) {  // if device name came with the deviceID, insert the corresponding config
                configWithDeviceId.insert({PluginConfigParams::KEY_DEVICE_ID, device.getDeviceID()});
            }
            executableNetworkPerDevice.insert({p.first, p.second.LoadNetwork(*clonedNetwork, configWithDeviceId)});
            multiNetworkConfig.insert(pluginsConfig[p.first].begin(), pluginsConfig[p.first].end());
        }
    }
    // finally, the multi-device settings that were used
    multiNetworkConfig.insert(priorities != config.end() ? *priorities /*local (network) settings*/
                                : *_config.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES) /* plugin config*/);
    if (executableNetworkPerDevice.empty())
        THROW_IE_EXCEPTION << NOT_FOUND_str << "Failed to load Executable network to any device "
                                            <<  "that the multi-device plugin is initialized to work with";

    auto perfConfig = fullConfig.find(PluginConfigParams::KEY_PERF_COUNT);
    bool enablePerfCounters = (fullConfig.end() != perfConfig) && (perfConfig->second == PluginConfigParams::YES);

    return std::make_shared<MultiDeviceExecutableNetwork>(executableNetworkPerDevice,
                                                          numRequestsPerDevices.empty() ? _numRequestsPerDevices : numRequestsPerDevices,
                                                          devicePriorities.empty() ? _devicePriorities : devicePriorities,
                                                          multiNetworkConfig,
                                                          enablePerfCounters);
}

std::map<std::string, std::string> GetSupportedConfig(const std::map<std::string, std::string>& config,
                                                      const InferenceEngine::InferencePlugin& plugin) {
    auto pluginApi = getInferencePluginAPIInterface(plugin);
    std::vector<std::string> supportedConfigKeys = pluginApi->GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), {});
    std::map<std::string, std::string> supportedConfig;
    for (auto&& key : supportedConfigKeys) {
        auto itKey = config.find(key);
        if (config.end() != itKey) {
            supportedConfig[key] = itKey->second;
        }
    }
    return supportedConfig;
}

void MultiDeviceInferencePlugin::QueryNetwork(const ICNNNetwork&                        network,
                                              const std::map<std::string, std::string>& config,
                                              QueryNetworkResult&                       queryResult) const {
    queryResult.rc = StatusCode::OK;
    // check if user has set any priorities/num_requests different from the plugin's default (e.g. via SetConfig)
    auto priorities = config.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES);
    DeviceMap<unsigned int> numRequestsPerDevices;
    std::vector<DeviceName> devicePriorities;
    DeviceMap<InferenceEngine::InferencePlugin> plugins;
    if (priorities != config.end()) {
        // since this is per network's config, the #requests data, priorities, etc are stored just locally (only for the network)
        ParseDevicesAndNumRequests(priorities->second, numRequestsPerDevices, devicePriorities);
        plugins = LoadDevices(devicePriorities);
    }
    if (plugins.empty()) {
        THROW_IE_EXCEPTION << NOT_FOUND_str << "No plugin is loaded to the Multi-Device. Failed to Query network";
    }

    std::unordered_map<std::string, std::string> fullConfig;
    fullConfig.insert(config.begin(), config.end());
    fullConfig.insert(_config.begin(), _config.end());

    // preparing local version of configs supported by plugins
    auto pluginsConfig = ParseConfigForDevices(fullConfig, plugins);

    std::map<std::string, QueryNetworkResult> queryResults;

    for (auto&& value : plugins) {
        auto& device = value.first;
        auto& plugin = value.second;
        auto configWithDeviceId = pluginsConfig[device];
        DeviceIDParser deviceIDParser(device);
        if (!deviceIDParser.getDeviceID().empty()) {  // if device name came with the deviceID, insert the corresponding config
            configWithDeviceId.insert({PluginConfigParams::KEY_DEVICE_ID, deviceIDParser.getDeviceID()});
        }
        IE_SUPPRESS_DEPRECATED_START
        plugin.QueryNetwork(network, configWithDeviceId, queryResults[device]);
        IE_SUPPRESS_DEPRECATED_END
    }

    details::CNNNetworkIterator i(&network);
    while (i != details::CNNNetworkIterator()) {
        CNNLayer::Ptr layer = *i;
        bool layerIsInQueryResultsForAllDevices = std::all_of(std::begin(queryResults), std::end(queryResults),
                                                                [&](const std::map<std::string, QueryNetworkResult>::value_type& qr) {
                                                                    return qr.second.supportedLayersMap.end()
                                                                        != qr.second.supportedLayersMap.find(layer->name);});
        if (layerIsInQueryResultsForAllDevices) {
            queryResult.supportedLayersMap[layer->name] = GetName();
        }
        i++;
    }
}
}  // namespace MultiDevicePlugin
