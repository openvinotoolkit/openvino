// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <atomic>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <map>
#include <vector>
#include <utility>
#include <memory>
#include <string>

#include <cpp/ie_plugin_cpp.hpp>
#include <ie_plugin_dispatcher.hpp>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>
#include "ie_iinfer_request.hpp"
#include "details/ie_exception_conversion.hpp"
#include <ie_parallel.hpp>

#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
#include <tbb/concurrent_queue.h>
#endif

namespace MultiDevicePlugin {

using DeviceName = std::string;
template<typename T>
using DeviceMap = std::unordered_map<DeviceName, T>;


class MultiDeviceInferRequest : public InferenceEngine::InferRequestInternal {
public:
    using Ptr = std::shared_ptr<MultiDeviceInferRequest>;
    explicit MultiDeviceInferRequest(const InferenceEngine::InputsDataMap&  networkInputs,
                                     const InferenceEngine::OutputsDataMap& networkOutputs);
    void GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo>&) const override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }
    void InferImpl() override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }
    // Multi-Device impl specific: sets the data (blobs from the device-less requets to the specific device request)
    void SetBlobsToAnotherRequest(InferenceEngine::InferRequest& req);
};

#if ((IE_THREAD == IE_THREAD_TBB) || (IE_THREAD == IE_THREAD_TBB_AUTO))
template <typename T>
using ThreadSafeQueue = tbb::concurrent_queue<T>;
#else
template <typename T>
class ThreadSafeQueue {
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(_mutex);
        _queue.push(std::move(value));
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_queue.empty()) {
            value = std::move(_queue.front());
            _queue.pop();
            return true;
        } else {
            return false;
        }
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.empty();
    }

protected:
    std::queue<T>   _queue;
    std::mutex      _mutex;
};
#endif

class MultiDeviceExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault,
                                     public ITaskExecutor {
public:
    using Ptr = std::shared_ptr<MultiDeviceExecutableNetwork>;
    struct WorkerInferRequest {
        InferenceEngine::InferRequest   _inferRequest;
        Task                            _task;
        InferenceEngine::StatusCode     _status = InferenceEngine::StatusCode::OK;
    };
    using NotBusyWorkerRequests = ThreadSafeQueue<WorkerInferRequest*>;

    explicit MultiDeviceExecutableNetwork(const DeviceMap<InferenceEngine::ExecutableNetwork>&                  networksPerDevice,
                                          const DeviceMap<unsigned int>&                                        numRequestsPerDevices,
                                          const std::vector<DeviceName>&                                        networkDevices,
                                          const std::unordered_map<std::string, InferenceEngine::Parameter>&    config,
                                          const bool                                                            needPerfCounters = false);

    void SetConfig(const std::map<std::string, InferenceEngine::Parameter> &config, InferenceEngine::ResponseDesc *resp) override;
    void GetConfig(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *resp) const override;
    void GetMetric(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *resp) const override;
    void run(Task inferTask) override;
    void CreateInferRequest(InferenceEngine::IInferRequest::Ptr& asyncRequest) override;
    InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                      InferenceEngine::OutputsDataMap networkOutputs) override;
    ~MultiDeviceExecutableNetwork() override;

    void ScheduleToWorkerInferRequest();

    static thread_local WorkerInferRequest*                     _thisWorkerInferRequest;
    std::atomic_bool                                            _terminate = {false};
    std::mutex                                                  _mutex;
    std::vector<DeviceName>                                     _devicePriorities;
    DeviceMap<InferenceEngine::ExecutableNetwork>               _networksPerDevice;
    ThreadSafeQueue<Task>                                       _inferPipelineTasks;
    DeviceMap<NotBusyWorkerRequests>                            _idleWorkerRequests;
    DeviceMap<std::vector<WorkerInferRequest>>                  _workerRequests;
    std::unordered_map<std::string, InferenceEngine::Parameter> _config;
    bool                                                        _needPerfCounters = false;
};

class MultiDeviceAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<MultiDeviceAsyncInferRequest>;

    explicit MultiDeviceAsyncInferRequest(const MultiDeviceInferRequest::Ptr&           inferRequest,
                                          const bool                                    needPerfCounters,
                                          const MultiDeviceExecutableNetwork::Ptr&      multiDeviceExecutableNetwork,
                                          const InferenceEngine::ITaskExecutor::Ptr&    callbackExecutor);
    void Infer_ThreadUnsafe() override;
    void GetPerformanceCounts_ThreadUnsafe(std::map<std::string, InferenceEngineProfileInfo> &_perfMap) const override;
    ~MultiDeviceAsyncInferRequest() override;

protected:
    MultiDeviceExecutableNetwork::Ptr                                   _multiDeviceExecutableNetwork;
    MultiDeviceInferRequest::Ptr                                        _inferRequest;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>  _perfMap;
    bool                                                                _needPerfCounters = false;
    MultiDeviceExecutableNetwork::WorkerInferRequest*                   _workerInferRequest = nullptr;
};

class MultiDeviceInferencePlugin : public InferenceEngine::InferencePluginInternal {
public:
    MultiDeviceInferencePlugin();
    ~MultiDeviceInferencePlugin() override = default;

    InferenceEngine::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const InferenceEngine::ICore* core,
                                                                       const InferenceEngine::ICNNNetwork& network,
                                                                       const std::map<std::string, std::string>& config) override;

    void SetConfig(const std::map<std::string, std::string>& config) override;
    void AddExtension(InferenceEngine::IExtensionPtr extension) override;
    void QueryNetwork(const InferenceEngine::ICNNNetwork&       network,
                      const std::map<std::string, std::string>& config,
                      InferenceEngine::QueryNetworkResult&      res) const override;
    InferenceEngine::Parameter GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter>& options) const override;
    static void ParseDevicesAndNumRequests(const std::string&       devicesRequestsCfg,
                                           DeviceMap<unsigned int>& perDeviceRequests,
                                           std::vector<DeviceName>& parsedDevices);

protected:
    void LoadDevices(const std::vector<DeviceName>& parsedDevices);
    IE_SUPPRESS_DEPRECATED_START
    DeviceMap<std::map<std::string, std::string>>
    ParseConfigForDevices(const std::unordered_map<std::string, std::string>&   fullConfig,
                          DeviceMap<InferenceEngine::InferencePlugin>&          plugins) const;
    DeviceMap<InferenceEngine::InferencePlugin> LoadDevices(const std::vector<DeviceName>& parsedDevices) const;
    DeviceMap<InferenceEngine::InferencePlugin> _plugins;
    IE_SUPPRESS_DEPRECATED_END
    DeviceMap<unsigned int>                     _numRequestsPerDevices;
    std::vector<DeviceName>                     _devicePriorities;
    std::vector<InferenceEngine::IExtensionPtr> _extensions;
};

}  // namespace MultiDevicePlugin
