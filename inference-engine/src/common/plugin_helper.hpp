// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>
#include <queue>

#include <ie_parallel.hpp>
#include <threading/ie_itask_executor.hpp>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>

#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
# include <tbb/concurrent_queue.h>
#endif

namespace PluginHelper {

using DeviceName = std::string;
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
private:
    std::queue<T>   _queue;
    std::mutex      _mutex;
};

template <typename T>
class ThreadSafeBoundedQueue {
public:
    ThreadSafeBoundedQueue() = default;
    bool try_push(T value) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_capacity > _queue.size()) {
            _queue.push(std::move(value));
            return true;
        }
        return false;
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

    size_t size() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.size();
    }

private:
    std::queue<T>         _queue;
    mutable std::mutex    _mutex;
    std::size_t           _capacity { 0 };
};
#endif

struct WorkerInferRequest {
    InferenceEngine::SoIInferRequestInternal  _inferRequest;
    InferenceEngine::Task                     _task;
    std::exception_ptr                        _exceptionPtr { nullptr };
};

using NotBusyWorkerRequests = ThreadSafeBoundedQueue<WorkerInferRequest*>;

struct IdleGuard {
    explicit IdleGuard(PluginHelper::WorkerInferRequest* workerInferRequestPtr,
                       NotBusyWorkerRequests& notBusyWorkerRequests) :
        _workerInferRequestPtr{workerInferRequestPtr},
        _notBusyWorkerRequests{&notBusyWorkerRequests} {
    }
    ~IdleGuard() {
        if (nullptr != _notBusyWorkerRequests) {
            _notBusyWorkerRequests->try_push(_workerInferRequestPtr);
        }
    }
    NotBusyWorkerRequests* Release() {
        auto notBusyWorkerRequests = _notBusyWorkerRequests;
        _notBusyWorkerRequests = nullptr;
        return notBusyWorkerRequests;
    }
    PluginHelper::WorkerInferRequest*   _workerInferRequestPtr = nullptr;
    NotBusyWorkerRequests*              _notBusyWorkerRequests = nullptr;
};

void CreateWorkers(InferenceEngine::SoExecutableNetworkInternal&                        executableNetwork,
                   DeviceMap<std::vector<WorkerInferRequest>>&                          workerRequestsMap,
                   DeviceMap<NotBusyWorkerRequests>&                                    idleWorkerRequestsMap,
                   ThreadSafeQueue<InferenceEngine::Task>&                              inferPipelineTasks,
                   DeviceMap<std::unique_ptr<ThreadSafeQueue<InferenceEngine::Task>>>&  inferPipelineTasksDeviceSpecific,
                   uint32_t                                                             optimalNum,
                   const std::string&                                                   device,
                   std::function<void(InferenceEngine::Task, const DeviceName&)>        scheduleFunc);
}