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
#include <ie_plugin_config.hpp>
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
                                                           const DeviceMap<DeviceInformation>&                                  networkDevices,
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

        auto itNumRequests = _devicePriorities.find(device);
        unsigned int optimalNum = 0;
        try {
            optimalNum = network.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
        } catch (const details::InferenceEngineException &iie) {
            THROW_IE_EXCEPTION
                    << "Every device used with the Multi-Device should "
                    << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                    << "Failed to query the metric for the " << device << " with error:" << iie.what();
        }
        const auto numRequests = (_devicePriorities.end() == itNumRequests ||
            itNumRequests->second.numRequestsPerDevices == -1) ? optimalNum : itNumRequests->second.numRequestsPerDevices;
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
        auto& idleWorkerRequests = _idleWorkerRequests[device.first];
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
    /* NOTE: The only threads that use `MultiDeviceExecutableNetwork` Context are those that are used by Worker infer requests.
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
    auto priorities = config.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES);
    if (priorities == config.end() || config.size() > 1) {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str <<
            "The only config supported for the Network's SetConfig is MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES";
    } else {
        auto multiPlugin = std::dynamic_pointer_cast<MultiDeviceInferencePlugin>(this->_plugin);
        assert(multiPlugin != nullptr);
        auto metaDevices = multiPlugin->ParseMetaDevices(priorities->second, {});

        if (std::any_of(metaDevices.begin(), metaDevices.end(), [](const std::pair<DeviceName, DeviceInformation> & kvp) {
                return kvp.second.numRequestsPerDevices != -1;
            })) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str << "You can only change device priorities but not number of requests"
                     <<" with the Network's SetConfig(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES!";
        }

        {
            std::lock_guard<std::mutex> lock{_mutex};
            for (auto && device : metaDevices) {
                if (_devicePriorities.find(device.first) == _devicePriorities.end()) {
                    THROW_IE_EXCEPTION << NOT_FOUND_str << "You can only change device priorities but not add new devices with"
                        << " the Network's SetConfig(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES." << device.first <<
                            " device was not in the original device list!";
                }
            }
            _devicePriorities = metaDevices;

            // update value in config
            _config[MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] = priorities->second;
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

namespace {

std::map<std::string, std::string> mergeConfigs(std::map<std::string, std::string> config,
                                                const std::map<std::string, std::string> & local) {
    for (auto && kvp : local) {
        config[kvp.first] = kvp.second;
    }
    return config;
}

}  // namespace

std::map<std::string, std::string> MultiDeviceInferencePlugin::GetSupportedConfig(
    const std::map<std::string, std::string> & config, const std::string & deviceName) const {
    std::vector<std::string> supportedConfigKeys = GetCore()->GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
    std::map<std::string, std::string> supportedConfig;
    for (auto&& key : supportedConfigKeys) {
        auto itKey = config.find(key);
        if (config.end() != itKey) {
            supportedConfig[key] = itKey->second;
        }
    }
    return supportedConfig;
}

DeviceMap<DeviceInformation> MultiDeviceInferencePlugin::ParseMetaDevices(const std::string& priorities,
                                                                          const std::map<std::string, std::string> & config) const {
    DeviceMap<DeviceInformation> metaDevices;

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

    auto getDeviceConfig = [&] (const DeviceName & deviceWithID) {
        DeviceIDParser deviceParser(deviceWithID);
        std::string deviceName = deviceParser.getDeviceName();
        std::map<std::string, std::string> tconfig = mergeConfigs(_config, config);

        // set device ID if any
        std::string deviceIDLocal = deviceParser.getDeviceID();
        if (!deviceIDLocal.empty()) {
            tconfig[PluginConfigParams::KEY_DEVICE_ID] = deviceIDLocal;
        }

        return GetSupportedConfig(tconfig, deviceName);
    };

    for (auto && d : devicesWithRequests) {
        auto openingBracket = d.find_first_of('(');
        auto closingBracket = d.find_first_of(')', openingBracket);
        auto device_name = d.substr(0, openingBracket);

        int numRequests = -1;
        if (closingBracket != std::string::npos && openingBracket < closingBracket) {
            numRequests = std::stol(d.substr(openingBracket + 1, closingBracket - 1));

            if (numRequests <= 0) {
                THROW_IE_EXCEPTION << "Priority value for '" << device_name << "' must be > 0, while " << numRequests
                    << "is passed";
            }
        }

        // create meta device
        metaDevices[device_name] = { getDeviceConfig(device_name), numRequests };
    }

    return metaDevices;
}

Parameter MultiDeviceInferencePlugin::GetConfig(const std::string& name,
        const std::map<std::string, Parameter> & options) const {
    if (name == MULTI_CONFIG_KEY(DEVICE_PRIORITIES)) {
        auto it = _config.find(MULTI_CONFIG_KEY(DEVICE_PRIORITIES));
        if (it == _config.end()) {
            THROW_IE_EXCEPTION << "Value for KEY_MULTI_DEVICE_PRIORITIES is not set";
        } else {
            return { it->second };
        }
    } else {
        THROW_IE_EXCEPTION << "Unsupported config key: " << name;
    }
}

void MultiDeviceInferencePlugin::SetConfig(const std::map<std::string, std::string> & config) {
    for (auto && kvp : config) {
        _config[kvp.first] = kvp.second;
    }
}

IE_SUPPRESS_DEPRECATED_START

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
        std::vector<std::string> configKeys = { MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES };
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else {
        THROW_IE_EXCEPTION << "Unsupported metric key " << name;
    }
}

ExecutableNetworkInternal::Ptr MultiDeviceInferencePlugin::LoadExeNetworkImpl(const ICNNNetwork &network,
                                                                              const std::map<std::string, std::string>& config) {
    if (GetCore() == nullptr) {
        THROW_IE_EXCEPTION << "Please, work with MULTI device via InferencEngine::Core object";
    }

    // TODO: do we really need a clone?
    ICNNNetwork::Ptr clonedNetwork = cloneNet(network);

    auto fullConfig = mergeConfigs(_config, config);
    auto priorities = fullConfig.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES);
    if (priorities == fullConfig.end()) {
        THROW_IE_EXCEPTION << "KEY_MULTI_DEVICE_PRIORITIES key is not set for MULTI device";
    }

    DeviceMap<DeviceInformation> metaDevices = ParseMetaDevices(priorities->second, fullConfig);

    // collect the settings that are applicable to the devices we are loading the network to
    std::unordered_map<std::string, InferenceEngine::Parameter> multiNetworkConfig;
    multiNetworkConfig.insert(*priorities);

    DeviceMap<ExecutableNetwork> executableNetworkPerDevice;
    for (auto& p : metaDevices) {
        auto & deviceName = p.first;
        auto & metaDevice = p.second;
        auto & deviceConfig = metaDevice.config;
        executableNetworkPerDevice.insert({ deviceName, GetCore()->LoadNetwork(CNNNetwork{clonedNetwork}, deviceName, deviceConfig) });
        multiNetworkConfig.insert(deviceConfig.begin(), deviceConfig.end());
    }
    if (executableNetworkPerDevice.empty())
        THROW_IE_EXCEPTION << NOT_FOUND_str << "Failed to load Executable network to any device "
                                            <<  "that the MULTI device is initialized to work with";

    auto perfConfig = fullConfig.find(PluginConfigParams::KEY_PERF_COUNT);
    bool enablePerfCounters = (fullConfig.end() != perfConfig) && (perfConfig->second == PluginConfigParams::YES);

    return std::make_shared<MultiDeviceExecutableNetwork>(executableNetworkPerDevice,
                                                          metaDevices,
                                                          multiNetworkConfig,
                                                          enablePerfCounters);
}

void MultiDeviceInferencePlugin::QueryNetwork(const ICNNNetwork&                        network,
                                              const std::map<std::string, std::string>& config,
                                              QueryNetworkResult&                       queryResult) const {
    if (GetCore() == nullptr) {
        THROW_IE_EXCEPTION << "Please, work with MULTI device via InferencEngine::Core object";
    }

    queryResult.rc = StatusCode::OK;
    queryResult.supportedLayersMap.clear();

    auto fullConfig = mergeConfigs(_config, config);
    auto priorities = fullConfig.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES);
    if (priorities == fullConfig.end()) {
        THROW_IE_EXCEPTION << "KEY_MULTI_DEVICE_PRIORITIES key is not set for MULTI device";
    }

    DeviceMap<DeviceInformation> metaDevices = ParseMetaDevices(priorities->second, fullConfig);
    std::map<std::string, QueryNetworkResult> queryResults;

    for (auto&& value : metaDevices) {
        auto& deviceName = value.first;
        auto& metaDevice = value.second;
        queryResults[deviceName] = GetCore()->QueryNetwork(network, deviceName, metaDevice.config);
    }

    details::CNNNetworkIterator i(&network);
    while (i != details::CNNNetworkIterator()) {
        CNNLayer::Ptr layer = *i;
        bool layerIsInQueryResultsForAllDevices = std::all_of(std::begin(queryResults), std::end(queryResults),
            [&](const std::map<std::string, QueryNetworkResult>::value_type& qr) {
                return qr.second.supportedLayersMap.end() != qr.second.supportedLayersMap.find(layer->name);});
        if (layerIsInQueryResultsForAllDevices) {
            queryResult.supportedLayersMap[layer->name] = GetName();
        }
        i++;
    }
}
}  // namespace MultiDevicePlugin
