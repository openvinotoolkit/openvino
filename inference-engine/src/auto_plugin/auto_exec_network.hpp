// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <threading/ie_itask_executor.hpp>
#include <ie_parallel.hpp>

#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
#include <tbb/concurrent_queue.h>
#endif

namespace AutoPlugin {

using DeviceName = std::string;
using NetworkFuture = std::future<InferenceEngine::SoExecutableNetworkInternal>;

#if ((IE_THREAD == IE_THREAD_TBB) || (IE_THREAD == IE_THREAD_TBB_AUTO))
template <typename T>
using ThreadSafeBoundedQueue = tbb::concurrent_bounded_queue<T>;
#else
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

    void push(T value) {
        {
            std::unique_lock<std::mutex> lock(_mutex);
            while (_capacity <= _queue.size()) {
                _notFull.wait(lock);
            }
            _queue.push(std::move(value));
        }
        _notEmpty.notify_one();
    }

    void pop(T& value) {
        {
            std::unique_lock<std::mutex> lock(_mutex);
            while (!_capacity || _queue.empty()) {
                _notEmpty.wait(lock);
            }
            value = std::move(_queue.front());
            _queue.pop();
        }
        _notFull.notify_one();
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

protected:
    std::queue<T>   _queue;
    mutable std::mutex      _mutex;
    std::condition_variable _notFull;
    std::condition_variable _notEmpty;
    std::size_t     _capacity { 0 };
};
#endif

class AutoExecutableNetwork : public InferenceEngine::IExecutableNetworkInternal {
public:
    using Ptr = std::shared_ptr<AutoExecutableNetwork>;

    explicit AutoExecutableNetwork(NetworkFuture cpuTask,
                                   NetworkFuture acceleratorTask,
                                   bool          enablePerfCount);

    void Export(std::ostream& networkModel) override;
    InferenceEngine::RemoteContext::Ptr GetContext() const override;
    InferenceEngine::CNNNetwork GetExecGraphInfo() override;
    InferenceEngine::Parameter GetMetric(const std::string &name) const override;
    void SetConfig(const std::map<std::string, InferenceEngine::Parameter>& config) override;
    InferenceEngine::Parameter GetConfig(const std::string& name) const override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                       InferenceEngine::OutputsDataMap networkOutputs) override;
    bool TryGetActualNetwork(InferenceEngine::SoExecutableNetworkInternal& soExecNetwork);
    void PushRequest() const;
    void PopRequest(int32_t& id);

    ~AutoExecutableNetwork();

private:
    void WaitForActualDevice() const;
    void SetRequestQueue();
    void ResetRequestQueue() const;

private:
    InferenceEngine::SoExecutableNetworkInternal _networkFirstReady;
    mutable InferenceEngine::SoExecutableNetworkInternal _networkActualNeeded;
    NetworkFuture _cpuFuture;
    mutable NetworkFuture _acceleratorFuture;
    bool _enablePerfCount;
    mutable std::atomic<bool> _alreadyActualNetwork = {false};
    std::map<std::string, InferenceEngine::Parameter> _cacheConfig;
    mutable ThreadSafeBoundedQueue<int32_t> _requests;
};

}  // namespace AutoPlugin
