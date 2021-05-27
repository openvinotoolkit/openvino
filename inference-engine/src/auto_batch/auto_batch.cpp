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
#include <unordered_set>

#include "ie_metric_helpers.hpp"
#include <cpp_interfaces/base/ie_infer_async_request_base.hpp>
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include <legacy/ie_util_internal.hpp>
#include <ie_plugin_config.hpp>
#include "auto_batch.hpp"

namespace AutoBatchPlugin {
    using namespace InferenceEngine;

    template <Precision::ePrecision precision>
    Blob::Ptr create_shared_blob_on_top_of_batched_blob(Blob::Ptr batched_blob, size_t batch_id, size_t batch_num) {
        typedef typename PrecisionTrait<precision>::value_type TYPE;
        typedef typename std::add_pointer<TYPE>::type TYPEPTR;
        auto ptr = batched_blob->buffer().as<TYPEPTR>();
        auto sizePerBatch = batched_blob->size() / batch_num;
        auto layout = batched_blob->getTensorDesc().getLayout();
        SizeVector dims = batched_blob->getTensorDesc().getDims();

        if (layout == InferenceEngine::Layout::NC || layout == InferenceEngine::Layout::NCDHW
            || layout == InferenceEngine::Layout::NCHW || layout == InferenceEngine::Layout::NHWC
            || layout == InferenceEngine::Layout::NDHWC) {
            dims[0] = 1;
            assert(batched_blob->getTensorDesc().getPrecision() == precision);
            return make_shared_blob<TYPE>({precision, dims, batched_blob->getTensorDesc().getLayout()},
                                          ptr + sizePerBatch * batch_id, sizePerBatch);
        } else {
            // same blob for all requests (e.g. constants)
            return make_shared_blob<TYPE>({precision, dims, batched_blob->getTensorDesc().getLayout()},
                                          ptr);
        }
    }

// ------------------------------AutoBatchInferRequest----------------------------
AutoBatchInferRequest::AutoBatchInferRequest(const InputsDataMap&   networkInputs,
                                             const OutputsDataMap&  networkOutputs,
                                             AutoBatchExecutableNetwork::WorkerInferRequest* workerRequestPtr,
                                             int batch_id, int num_batch,
                                             bool needPerfCounters)
        : InferRequestInternal(networkInputs, networkOutputs), _workerInferRequest(workerRequestPtr),
        _needPerfCounters(needPerfCounters) {
    // Allocate all input blobs
    for (const auto &it : networkInputs) {
        auto blob = workerRequestPtr->_inferRequest.GetBlob(it.first);
        Blob::Ptr res;
        switch (it.second->getTensorDesc().getPrecision()) {
            case InferenceEngine::Precision::FP32:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::FP32>
                        (workerRequestPtr->_inferRequest.GetBlob(it.first), batch_id, num_batch);
                break;
            case InferenceEngine::Precision::I32:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I32>
                        (workerRequestPtr->_inferRequest.GetBlob(it.first), batch_id, num_batch);
                break;
            case InferenceEngine::Precision::I8:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I8>
                        (workerRequestPtr->_inferRequest.GetBlob(it.first), batch_id, num_batch);
                break;
            case InferenceEngine::Precision::U16:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U16>
                        (workerRequestPtr->_inferRequest.GetBlob(it.first), batch_id, num_batch);
                break;

            case InferenceEngine::Precision::I16:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I16>
                        (workerRequestPtr->_inferRequest.GetBlob(it.first), batch_id, num_batch);

                break;
            case InferenceEngine::Precision::U8:
            case InferenceEngine::Precision::BOOL:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U8>
                        (workerRequestPtr->_inferRequest.GetBlob(it.first), batch_id, num_batch);
                break;
            default:
                THROW_IE_EXCEPTION << "Unsupported input precision " << it.second->getTensorDesc().getPrecision();
        }
        _inputs[it.first] = res;
    }
    // Allocate all output blobs
    for (const auto &it : networkOutputs) {
        auto blob = workerRequestPtr->_inferRequest.GetBlob(it.first);
        Blob::Ptr res;
        switch (it.second->getTensorDesc().getPrecision()) {
            case InferenceEngine::Precision::FP32:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::FP32>
                        (workerRequestPtr->_inferRequest.GetBlob(it.first), batch_id, num_batch);
                break;
            case InferenceEngine::Precision::I32:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I32>
                        (workerRequestPtr->_inferRequest.GetBlob(it.first), batch_id, num_batch);
                break;
            case InferenceEngine::Precision::I8:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I8>
                        (workerRequestPtr->_inferRequest.GetBlob(it.first), batch_id, num_batch);
                break;
            case InferenceEngine::Precision::U16:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U16>
                        (workerRequestPtr->_inferRequest.GetBlob(it.first), batch_id, num_batch);
                break;

            case InferenceEngine::Precision::I16:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::I16>
                        (workerRequestPtr->_inferRequest.GetBlob(it.first), batch_id, num_batch);

                break;
            case InferenceEngine::Precision::U8:
            case InferenceEngine::Precision::BOOL:
                res = create_shared_blob_on_top_of_batched_blob<InferenceEngine::Precision::U8>
                        (workerRequestPtr->_inferRequest.GetBlob(it.first), batch_id, num_batch);
                break;
            default:
                THROW_IE_EXCEPTION << "Unsupported input precision " << it.second->getTensorDesc().getPrecision();
        }
        _outputs[it.first] = res;
    }
}

void AutoBatchInferRequest::SetBlobsToAnotherRequest(InferRequest& req) {
    // todo call Set for REMOTE BLOB
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> AutoBatchInferRequest::GetPerformanceCounts() const {
    return _perfMap;
}

void AutoBatchInferRequest::InferImpl() {
    auto _event = _workerInferRequest->_event;
    auto numReady = ++_workerInferRequest->_numRequestsReady;
    if (numReady == _workerInferRequest->_batchSize) {
        _workerInferRequest->_numRequestsReady = 0;
        _workerInferRequest->_inferRequest.StartAsync();
    }
    _event.get();
    if (_needPerfCounters) {
        _perfMap = _workerInferRequest->_inferRequest.GetPerformanceCounts();
    }
}

AutoBatchAsyncInferRequest::AutoBatchAsyncInferRequest(
    const AutoBatchInferRequest::Ptr&           inferRequest,
    const bool                                  needPerfCounters,
    const AutoBatchExecutableNetwork::Ptr&      autoBatchExecutableNetwork,
    const ITaskExecutor::Ptr&                   callbackExecutor) :
    AsyncInferRequestThreadSafeDefault(inferRequest,
            std::make_shared<CPUStreamsExecutor>(
                    IStreamsExecutor::Config{"AutoBatch", 1, 1,
                                             IStreamsExecutor::ThreadBindingType::NONE, 1, 0, 1}),
            callbackExecutor),
    _AutoBatchExecutableNetwork{autoBatchExecutableNetwork},
    _inferRequest{inferRequest} {
  }

void AutoBatchAsyncInferRequest::Infer_ThreadUnsafe() {
    InferUsingAsync();
}

AutoBatchAsyncInferRequest::~AutoBatchAsyncInferRequest() {
    StopAndWait();
}

// ------------------------------AutoBatchExecutableNetwork----------------------------
AutoBatchExecutableNetwork::AutoBatchExecutableNetwork(const InferenceEngine::ExecutableNetwork&    networkForDevice,
                                                           const DeviceInformation&                 networkDevice,
                                                           const std::unordered_map<std::string, InferenceEngine::Parameter>&   config,
                                                           const bool                                                           needPerfCounters) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault(
            nullptr,
            std::make_shared<InferenceEngine::ImmediateExecutor>()),
    _device{networkDevice},
    _network{networkForDevice},
    _config{config},
    _needPerfCounters{needPerfCounters} {
}

AutoBatchExecutableNetwork::~AutoBatchExecutableNetwork() {
//    {
//        std::lock_guard<std::mutex> lock(_mutex);
//        _device = {};
//    }
    _terminate = true;
    /* NOTE: The only threads that use `AutoBatchExecutableNetwork` Context are those that are used by Worker infer requests.
     *       But AsyncInferRequest destructor should waits for all asynchronous tasks that are used by the request
     */
    _workerRequests.clear();
}

InferenceEngine::InferRequestInternal::Ptr AutoBatchExecutableNetwork::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                                                InferenceEngine::OutputsDataMap networkOutputs) {
        // todo : guard request creation from another thread/on-the-fly
        auto num = _numRequestsCreated++;
        auto batch_id = num % _device.batchForDevice;
        if (!batch_id) {  //need new request
            _workerRequests.push_back(std::make_shared<WorkerInferRequest>());
            auto workerRequestPtr = _workerRequests.back();
            workerRequestPtr->_inferRequest = _network.CreateInferRequest();
            workerRequestPtr->_batchSize = _device.batchForDevice;
            workerRequestPtr->_cond = std::promise<void>();
            workerRequestPtr->_event = workerRequestPtr->_cond.get_future().share();
            // _idleWorkerRequests.push(workerRequestPtr);
            workerRequestPtr->_inferRequest.SetCompletionCallback<std::function<void(InferRequest, StatusCode)>>(
                [workerRequestPtr, this] (InferRequest , StatusCode status) mutable {
                    workerRequestPtr->_status = status;
                    auto signal = std::move(workerRequestPtr->_cond);
                    // reset the promise/future for next use
                    workerRequestPtr->_cond = std::promise<void>();
                    workerRequestPtr->_event = workerRequestPtr->_cond.get_future().share();
                    signal.set_value();
                });
       }
    return std::make_shared<AutoBatchInferRequest>(networkInputs, networkOutputs, _workerRequests.back().get(),
            batch_id, _device.batchForDevice, _needPerfCounters);
}

InferenceEngine::IInferRequest::Ptr AutoBatchExecutableNetwork::CreateInferRequest() {
    auto syncRequestImpl = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
    auto asyncTreadSafeImpl = std::make_shared<AutoBatchAsyncInferRequest>(std::static_pointer_cast<AutoBatchInferRequest>(syncRequestImpl),
                                                                             _needPerfCounters,
                                                                             std::static_pointer_cast<AutoBatchExecutableNetwork>(shared_from_this()),
                                                                             _callbackExecutor);
    IInferRequest::Ptr asyncRequest;
    asyncRequest.reset(new InferRequestBase(asyncTreadSafeImpl), [](IInferRequest* p) { p->Release(); });
    asyncTreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
    return asyncRequest;
}

void AutoBatchExecutableNetwork::SetConfig(const std::map<std::string, InferenceEngine::Parameter> &config) {
    // TODO
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}

InferenceEngine::Parameter AutoBatchExecutableNetwork::GetConfig(const std::string &name) const {
    auto res = _config.find(name);
    if (res != _config.end()) {
        return res->second;
    } else {
        THROW_IE_EXCEPTION << NOT_FOUND_str << name <<" not found in the ExecutableNetwork config";
    }
}

InferenceEngine::Parameter AutoBatchExecutableNetwork::GetMetric(const std::string &name) const {
    if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        unsigned int res = 0u;
        try {
            res = _network.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
        } catch (const details::InferenceEngineException &iie) {
            THROW_IE_EXCEPTION
                    << "Every device used with the Auto-Batching should "
                    << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                    << "Failed to query the metric for the "
                    << _network.GetMetric(METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>()
                    << " with error:" << iie.what();
        }
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, res * _device.batchForDevice);
    } else if (name == METRIC_KEY(NETWORK_NAME)) {
        IE_SET_METRIC_RETURN(NETWORK_NAME, _network.GetMetric(
                METRIC_KEY(NETWORK_NAME)).as<std::string>());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, {
            METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(NETWORK_NAME),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS)
        });
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys = { CONFIG_KEY(AUTO_BATCH) };
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else {
        THROW_IE_EXCEPTION << "Unsupported Network metric: " << name;
    }
}

// ------------------------------AutoBatchInferencePlugin----------------------------

namespace {

std::map<std::string, std::string> mergeConfigs(std::map<std::string, std::string> config,
                                                const std::map<std::string, std::string> & local) {
    for (auto && kvp : local) {
        config[kvp.first] = kvp.second;
    }
    return config;
}

}  // namespace

std::map<std::string, std::string> AutoBatchInferencePlugin::GetSupportedConfig(
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

DeviceInformation AutoBatchInferencePlugin::ParseMetaDevice(const std::string& devicesBatchCfg,
                                                                          const std::map<std::string, std::string> & config) const {
    DeviceInformation metaDevice;
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

    auto && d = devicesBatchCfg;
    {
        auto openingBracket = d.find_first_of('(');
        auto closingBracket = d.find_first_of(')', openingBracket);
        auto deviceName = d.substr(0, openingBracket);

        int batch = -1;
        if (closingBracket != std::string::npos && openingBracket < closingBracket) {
            batch = std::stol(d.substr(openingBracket + 1, closingBracket - 1));

            if (batch <= 0) {
                THROW_IE_EXCEPTION << "Batch value for '" << deviceName << "' must be > 0, while " << batch
                    << "is passed";
            }
        }

        // create meta device
        auto cfg = getDeviceConfig(deviceName);
        std::vector<std::string> supportedConfigKeys = GetCore()->GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        if (std::find(std::begin(supportedConfigKeys), std::end(supportedConfigKeys), CONFIG_KEY_INTERNAL(AGGREGATED_PLUGIN))
            != std::end(supportedConfigKeys)) {
            cfg.emplace(CONFIG_KEY_INTERNAL(AGGREGATED_PLUGIN), "");
        }
        metaDevice = { deviceName, cfg, batch };
    }

    return metaDevice;
}

Parameter AutoBatchInferencePlugin::GetConfig(const std::string& name,
        const std::map<std::string, Parameter> & options) const {
    if (name == CONFIG_KEY(AUTO_BATCH)) {
        auto it = _config.find(CONFIG_KEY(AUTO_BATCH));
        if (it == _config.end()) {
            THROW_IE_EXCEPTION << "Value for KEY_AUTO_BATCH is not set";
        } else {
            return { it->second };
        }
    } else {
        THROW_IE_EXCEPTION << "Unsupported config key: " << name;
    }
}

void AutoBatchInferencePlugin::SetConfig(const std::map<std::string, std::string> & config) {
    for (auto && kvp : config) {
        _config[kvp.first] = kvp.second;
    }
}

static const Version version = {{2, 1}, CI_BUILD_NUMBER, "AutoBatchPlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(AutoBatchInferencePlugin, version)

AutoBatchInferencePlugin::AutoBatchInferencePlugin() {
    _pluginName = "BATCH";
}

InferenceEngine::Parameter AutoBatchInferencePlugin::GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter> & options) const {
    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        std::string name = { "BATCH" };
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, name);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys = {
            CONFIG_KEY_INTERNAL(AGGREGATED_PLUGIN)};
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else {
        THROW_IE_EXCEPTION << "Unsupported metric key " << name;
    }
}

ExecutableNetworkInternal::Ptr AutoBatchInferencePlugin::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork&network,
                                                                              const std::map<std::string, std::string>& config) {
    if (GetCore() == nullptr) {
        THROW_IE_EXCEPTION << "Please, work with MULTI device via InferencEngine::Core object";
    }

    auto fullConfig = mergeConfigs(_config, config);
    auto device_batch = fullConfig.find(CONFIG_KEY(AUTO_BATCH));
    if (device_batch == fullConfig.end()) {
        THROW_IE_EXCEPTION << "KEY_AUTO_BATCH key is not set for BATCH device";
    }

    auto metaDevice = ParseMetaDevice(device_batch->second, fullConfig);

    // collect the settings that are applicable to the devices we are loading the network to
    std::unordered_map<std::string, InferenceEngine::Parameter> networkConfig;
    networkConfig.insert(*device_batch);

    ExecutableNetwork executableNetworkForDevice;
    auto & deviceName = metaDevice.deviceName;
    auto & deviceConfig = metaDevice.config;
    // network.serialize("out_orig.xml", "out_orig.bin");

    CNNNetwork clonedNetwork(InferenceEngine::cloneNetwork(network));
    const InputsDataMap inputInfo = clonedNetwork.getInputsInfo();
    ICNNNetwork::InputShapes shapes = clonedNetwork.getInputShapes();

    for (const InputsDataMap::value_type &item : inputInfo) {
        auto layout = item.second->getTensorDesc().getLayout();
        if (layout == InferenceEngine::Layout::NC || layout == InferenceEngine::Layout::NCDHW
                || layout == InferenceEngine::Layout::NCHW || layout == InferenceEngine::Layout::NHWC
                || layout == InferenceEngine::Layout::NDHWC) {
            shapes[item.first][0] = metaDevice.batchForDevice;
            std::cout << "  reshaping the input " << item.first << " (layout " << layout << ")" << " by the batch" << std::endl;
        }
    }
    std::cout << "Reshaped network by batch to  " << metaDevice.batchForDevice << std::endl;
    clonedNetwork.reshape(shapes);
    // clonedNetwork.serialize("out_batch4.xml", "out_batch4.bin");

    std::map<std::string, std::string> deviceConfig0 = deviceConfig;
    // deviceConfig0["DO_NOT_AUTO_BATCH"] = "TRUE";
    executableNetworkForDevice = GetCore()->LoadNetwork(CNNNetwork{clonedNetwork}, deviceName, deviceConfig0);
    networkConfig.insert(deviceConfig.begin(), deviceConfig.end());
    if ((std::shared_ptr<InferenceEngine::IExecutableNetwork>)executableNetworkForDevice == nullptr)
        THROW_IE_EXCEPTION << NOT_FOUND_str << "Failed to load Executable network the device "
                                            <<  "that the BATCH device is initialized to work with";

    auto perfConfig = fullConfig.find(PluginConfigParams::KEY_PERF_COUNT);
    bool enablePerfCounters = (fullConfig.end() != perfConfig) && (perfConfig->second == PluginConfigParams::YES);

    return std::make_shared<AutoBatchExecutableNetwork>(executableNetworkForDevice,
                                                          metaDevice,
                                                          networkConfig,
                                                          enablePerfCounters);
}

InferenceEngine::QueryNetworkResult AutoBatchInferencePlugin::QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                              const std::map<std::string, std::string>& config) const {
//    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    const std::map<std::string, std::string> cfg;
    return GetCore()->QueryNetwork(network, "CPU", cfg);
}
}  // namespace AutoBatchPlugin
