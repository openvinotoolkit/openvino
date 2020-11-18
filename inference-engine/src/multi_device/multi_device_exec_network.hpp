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
#include <string>

#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>
#include "ie_iinfer_request.hpp"
#include "details/ie_exception_conversion.hpp"
#include <ie_parallel.hpp>

#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
# include <tbb/concurrent_queue.h>
#endif

namespace MultiDevicePlugin {

using DeviceName = std::string;

struct DeviceInformation {
    DeviceName deviceName;
    std::map<std::string, std::string> config;
    int numRequestsPerDevices;
};

template<typename T>
using DeviceMap = std::unordered_map<DeviceName, T>;

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
                                          const std::vector<DeviceInformation>&                                 networkDevices,
                                          const std::unordered_map<std::string, InferenceEngine::Parameter>&    config,
                                          const bool                                                            needPerfCounters = false);

    void SetConfig(const std::map<std::string, InferenceEngine::Parameter> &config) override;
    InferenceEngine::Parameter GetConfig(const std::string &name) const override;
    InferenceEngine::Parameter GetMetric(const std::string &name) const override;
    void run(Task inferTask) override;
    InferenceEngine::IInferRequest::Ptr CreateInferRequest() override;
    InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                      InferenceEngine::OutputsDataMap networkOutputs) override;
    ~MultiDeviceExecutableNetwork() override;

    void ScheduleToWorkerInferRequest();

    static thread_local WorkerInferRequest*                     _thisWorkerInferRequest;
    std::atomic_bool                                            _terminate = {false};
    std::mutex                                                  _mutex;
    std::vector<DeviceInformation>                              _devicePriorities;
    DeviceMap<InferenceEngine::ExecutableNetwork>               _networksPerDevice;
    ThreadSafeQueue<Task>                                       _inferPipelineTasks;
    DeviceMap<NotBusyWorkerRequests>                            _idleWorkerRequests;
    DeviceMap<std::vector<WorkerInferRequest>>                  _workerRequests;
    std::unordered_map<std::string, InferenceEngine::Parameter> _config;
    bool                                                        _needPerfCounters = false;
};

}  // namespace MultiDevicePlugin
