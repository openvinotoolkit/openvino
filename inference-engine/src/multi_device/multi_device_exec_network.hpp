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

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <ie_parallel.hpp>
#include <threading/ie_itask_executor.hpp>

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
template <typename T>
using ThreadSafeBoundedQueue = tbb::concurrent_bounded_queue<T>;
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
protected:
    std::queue<T>   _queue;
    std::mutex      _mutex;
};
template <typename T>
class ThreadSafeBoundedQueue {
public:
    ThreadSafeBoundedQueue() = default;
    bool try_push(T value) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_capacity) {
            _queue.push(std::move(value));
        }
        return _capacity;
    }
    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_capacity && !_queue.empty()) {
            value = std::move(_queue.front());
            _queue.pop();
            return true;
        } else {
            return false;
        }
    }
    void set_capacity(std::size_t newCapacity) {
        std::lock_guard<std::mutex> lock(_mutex);
        _capacity = newCapacity;
    }

protected:
    std::queue<T>   _queue;
    std::mutex      _mutex;
    bool            _capacity = false;
};
#endif

class MultiDeviceExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault,
                                     public InferenceEngine::ITaskExecutor {
public:
    using Ptr = std::shared_ptr<MultiDeviceExecutableNetwork>;
    struct WorkerInferRequest {
        InferenceEngine::InferRequest   _inferRequest;
        InferenceEngine::Task           _task;
        InferenceEngine::StatusCode     _status = InferenceEngine::StatusCode::OK;
    };
    using NotBusyWorkerRequests = ThreadSafeBoundedQueue<WorkerInferRequest*>;

    explicit MultiDeviceExecutableNetwork(const DeviceMap<InferenceEngine::ExecutableNetwork>&                  networksPerDevice,
                                          const std::vector<DeviceInformation>&                                 networkDevices,
                                          const std::unordered_map<std::string, InferenceEngine::Parameter>&    config,
                                          const bool                                                            needPerfCounters = false);

    void SetConfig(const std::map<std::string, InferenceEngine::Parameter> &config) override;
    InferenceEngine::Parameter GetConfig(const std::string &name) const override;
    InferenceEngine::Parameter GetMetric(const std::string &name) const override;
    void run(InferenceEngine::Task inferTask) override;
    InferenceEngine::IInferRequest::Ptr CreateInferRequest() override;
    InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                      InferenceEngine::OutputsDataMap networkOutputs) override;
    ~MultiDeviceExecutableNetwork() override;

    void ScheduleToWorkerInferRequest(InferenceEngine::Task);

    static thread_local WorkerInferRequest*                     _thisWorkerInferRequest;
    std::atomic_bool                                            _terminate = {false};
    std::mutex                                                  _mutex;
    std::vector<DeviceInformation>                              _devicePriorities;
    const std::vector<DeviceInformation>                        _devicePrioritiesInitial;
    DeviceMap<InferenceEngine::ExecutableNetwork>               _networksPerDevice;
    ThreadSafeQueue<InferenceEngine::Task>                      _inferPipelineTasks;
    DeviceMap<NotBusyWorkerRequests>                            _idleWorkerRequests;
    DeviceMap<std::vector<WorkerInferRequest>>                  _workerRequests;
    std::unordered_map<std::string, InferenceEngine::Parameter> _config;
    bool                                                        _needPerfCounters = false;
    std::atomic_size_t                                          _numRequestsCreated = {0};
};

}  // namespace MultiDevicePlugin
