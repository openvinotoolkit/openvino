// Copyright (C) 2018-2021 Intel Corporation
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
#include <threading/ie_executor_manager.hpp>
#include "ie_icore.hpp"

#if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
# include <tbb/concurrent_queue.h>
#endif

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {

class MultiDeviceInferencePlugin;

using DeviceName = std::string;
using NetworkFuture = std::future<InferenceEngine::SoExecutableNetworkInternal>;
using NetworkPromise = std::promise<InferenceEngine::SoExecutableNetworkInternal>;

struct DeviceInformation {
    DeviceName deviceName;
    std::map<std::string, std::string> config;
    int numRequestsPerDevices;
    std::string defaultDeviceID;
    DeviceName uniqueName;
};

struct AutoContext {
    bool           needPerfCounters = {false};
    unsigned int   modelPriority = 0;
};

struct AutoLoadContext {
    std::atomic<bool> isEnabled = {false};
    std::atomic<bool> isAlready = {false};
    std::atomic<bool> isLoadSuccess = {false};
    std::future<void> future;
    std::promise<void> promise;
    InferenceEngine::SoExecutableNetworkInternal executableNetwork;
    DeviceInformation  deviceInfo;
    std::vector<DeviceInformation> metaDevices;
    std::string networkPrecision;
    std::string errMessage;
    InferenceEngine::Task task;
    //ACTUALDEVICE's workName is same as it's deviceName,
    //CPU_HELP's workName is "CPU_HELP", and  deviceName is "CPU"
    //add workName is because of ACTUALDEVICE and CPU maybe all CPU,
    //they can't use the same workerQueue
    std::string workName = "";
};

enum AutoLoadContextIndex {
     CPU = 0,
     ACTUALDEVICE = 1,
     CONTEXTNUM = 2
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
        InferenceEngine::SoIInferRequestInternal  _inferRequest;
        InferenceEngine::Task                     _task;
        std::exception_ptr                        _exceptionPtr = nullptr;
    };
    using NotBusyWorkerRequests = ThreadSafeBoundedQueue<WorkerInferRequest*>;

    explicit MultiDeviceExecutableNetwork(const DeviceMap<InferenceEngine::SoExecutableNetworkInternal>&        networksPerDevice,
                                          const std::vector<DeviceInformation>&                                 networkDevices,
                                          const std::unordered_map<std::string, InferenceEngine::Parameter>&    config,
                                          const bool                                                            needPerfCounters = false);
    MultiDeviceExecutableNetwork(const std::string&                           modelPath,
                                 const InferenceEngine::CNNNetwork&           network,
                                 const std::vector<DeviceInformation>&        metaDevices,
                                 const std::string&                           strDevices,
                                 MultiDeviceInferencePlugin*                  plugin,
                                 const AutoContext&                           context,
                                 const bool                                   needPerfCounters = false);

    void SetConfig(const std::map<std::string, InferenceEngine::Parameter> &config) override;
    InferenceEngine::Parameter GetConfig(const std::string &name) const override;
    InferenceEngine::Parameter GetMetric(const std::string &name) const override;
    void run(InferenceEngine::Task inferTask) override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                       InferenceEngine::OutputsDataMap networkOutputs) override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                                                       const std::vector<std::shared_ptr<const ov::Node>>& outputs) override;
    std::shared_ptr<InferenceEngine::RemoteContext> GetContext() const override;
    std::shared_ptr<InferenceEngine::ICore> GetCore() const;
    ~MultiDeviceExecutableNetwork() override;

    void ScheduleToWorkerInferRequest(InferenceEngine::Task, DeviceName preferred_device = "");

    static thread_local WorkerInferRequest*                     _thisWorkerInferRequest;
    // have to use the const char* ptr rather than std::string due to a bug in old gcc versions,
    // the bug is e.g. manifesting on the old CentOS (and it's 4.8.x gcc) used in our testing
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81880
    static thread_local const char*                             _thisPreferredDeviceName;
    mutable std::mutex                                          _mutex;
    std::vector<DeviceInformation>                              _devicePriorities;
    const std::vector<DeviceInformation>                        _devicePrioritiesInitial;
    DeviceMap<InferenceEngine::SoExecutableNetworkInternal>     _networksPerDevice;
    ThreadSafeQueue<InferenceEngine::Task>                      _inferPipelineTasks;
    DeviceMap<std::unique_ptr<ThreadSafeQueue<InferenceEngine::Task>>> _inferPipelineTasksDeviceSpecific;
    DeviceMap<NotBusyWorkerRequests>                            _idleWorkerRequests;
    DeviceMap<std::vector<WorkerInferRequest>>                  _workerRequests;
    std::unordered_map<std::string, InferenceEngine::Parameter> _config;
    bool                                                        _needPerfCounters = false;
    std::atomic_size_t                                          _numRequestsCreated = {0};

private:
    void GenerateWorkers(const std::string& device, const InferenceEngine::SoExecutableNetworkInternal& executableNetwork);
    void WaitActualNetworkReady() const;
    void WaitFirstNetworkReady();
    static bool RunPipelineTask(InferenceEngine::Task& inferPipelineTask,
                                NotBusyWorkerRequests& idleWorkerRequests,
                                const DeviceName& preferred_device);
    void TryToLoadNetWork(AutoLoadContext& context,
                          const std::string& modelPath,
                          const InferenceEngine::CNNNetwork& network);

private:
    std::shared_ptr<InferenceEngine::ICore>                             _core;
    InferenceEngine::IStreamsExecutor::Ptr                              _executor;
    MultiDeviceInferencePlugin*                                         _multiPlugin;
    AutoContext                                                         _context;
    bool                                                                _workModeIsAUTO = {false};
    mutable std::once_flag                                              _oc;
    std::once_flag                                                      _firstLoadOC;
    std::future<void>                                                   _firstLoadFuture;
    std::promise<void>                                                  _firstLoadPromise;
    mutable AutoLoadContext                                             _loadContext[CONTEXTNUM];
    mutable std::mutex                                                  _confMutex;
};

}  // namespace MultiDevicePlugin
