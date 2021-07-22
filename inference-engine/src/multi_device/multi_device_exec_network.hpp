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

#include "plugin_helper.hpp"
#include "plugin_exec_network.hpp"

namespace MultiDevicePlugin {

using namespace PluginHelper;

using NotBusyWorkerRequests = ThreadSafeBoundedQueue<WorkerInferRequest*>;
using DeviceName = std::string;

template<typename T>
using DeviceMap = std::unordered_map<DeviceName, T>;

class MultiDeviceExecutableNetwork : public PluginExecHelper {
public:
    using Ptr = std::shared_ptr<MultiDeviceExecutableNetwork>;

    explicit MultiDeviceExecutableNetwork(const DeviceMap<InferenceEngine::SoExecutableNetworkInternal>&                  networksPerDevice,
                                          const std::vector<DeviceInformation>&                                 networkDevices,
                                          const std::unordered_map<std::string, InferenceEngine::Parameter>&    config,
                                          const bool                                                            needPerfCounters = false);

    void SetConfig(const std::map<std::string, InferenceEngine::Parameter> &config) override;
    InferenceEngine::Parameter GetConfig(const std::string &name) const override;
    InferenceEngine::Parameter GetMetric(const std::string &name) const override;
    void run(InferenceEngine::Task inferTask) override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                       InferenceEngine::OutputsDataMap networkOutputs) override;
    InferenceEngine::RemoteContext::Ptr GetContext() const override;
    ~MultiDeviceExecutableNetwork() override;

    void ScheduleToWorkerInferRequest(InferenceEngine::Task, DeviceName preferred_device = "");

    mutable std::mutex                                          _mutex;
    std::vector<DeviceInformation>                              _devicePriorities;
    DeviceMap<InferenceEngine::SoExecutableNetworkInternal>     _networksPerDevice;
    ThreadSafeQueue<InferenceEngine::Task>                      _inferPipelineTasks;
    DeviceMap<std::unique_ptr<ThreadSafeQueue<InferenceEngine::Task>>> _inferPipelineTasksDeviceSpecific;
    DeviceMap<NotBusyWorkerRequests>                            _idleWorkerRequests;
    DeviceMap<std::vector<WorkerInferRequest>>                  _workerRequests;
    std::unordered_map<std::string, InferenceEngine::Parameter> _config;
    bool                                                        _needPerfCounters = false;
    std::atomic_size_t                                          _numRequestsCreated = {0};
};

}  // namespace MultiDevicePlugin
