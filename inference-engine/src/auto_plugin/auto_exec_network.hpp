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

#include <ie_parallel.hpp>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <threading/ie_itask_executor.hpp>

#include "plugin_helper.hpp"
#include "plugin_exec_network.hpp"

namespace AutoPlugin {

class AutoInferencePlugin;

using namespace PluginHelper;

using DeviceName = std::string;
using ConfigType = std::map<std::string, std::string>;
using NetworkFuture = std::future<InferenceEngine::SoExecutableNetworkInternal>;
using NotBusyWorkerRequests = ThreadSafeBoundedQueue<WorkerInferRequest*>;

template<typename T>
using DeviceMap = std::unordered_map<DeviceName, T>;


class AutoExecutableNetwork : public PluginExecHelper {
public:
    using Ptr = std::shared_ptr<AutoExecutableNetwork>;

    AutoExecutableNetwork(const std::string&                 modelPath,
                          const InferenceEngine::CNNNetwork& network,
                          const ConfigType&                  config,
                          AutoInferencePlugin*               plugin);

    void Export(std::ostream& networkModel) override;
    InferenceEngine::RemoteContext::Ptr GetContext() const override;
    InferenceEngine::CNNNetwork GetExecGraphInfo() override;
    InferenceEngine::Parameter GetMetric(const std::string &name) const override;
    void SetConfig(const std::map<std::string, InferenceEngine::Parameter>& config) override;
    InferenceEngine::Parameter GetConfig(const std::string& name) const override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                       InferenceEngine::OutputsDataMap networkOutputs) override;
    void run(InferenceEngine::Task inferTask) override;
    bool TryGetActualNetwork(InferenceEngine::SoExecutableNetworkInternal& soExecNetwork);

    ~AutoExecutableNetwork() final;

private:
    void WaitForActualDevice() const;
    void ScheduleToWorkerInferRequest(InferenceEngine::Task, DeviceName preferred_device = "");
    static bool RunPipelineTask(InferenceEngine::Task& inferPipelineTask,
                                NotBusyWorkerRequests& idleWorkerRequests,
                                const DeviceName& preferred_device);

private:
    AutoInferencePlugin*                                                _autoPlugin;
    InferenceEngine::SoExecutableNetworkInternal                        _networkFirstReady;
    mutable InferenceEngine::SoExecutableNetworkInternal                _networkActualNeeded;
    NetworkFuture                                                       _cpuFuture;
    mutable NetworkFuture                                               _acceleratorFuture;
    std::map<std::string, InferenceEngine::Parameter>                   _cacheConfig;
    DeviceMap<std::vector<WorkerInferRequest>>                          _workerRequests;
    DeviceMap<NotBusyWorkerRequests>                                    _idleWorkerRequests;
    ThreadSafeQueue<InferenceEngine::Task>                              _inferPipelineTasks;
    DeviceMap<std::unique_ptr<ThreadSafeQueue<InferenceEngine::Task>>>  _inferPipelineTasksDeviceSpecific;
    std::string                                                         _cpuDeviceName;
    std::string                                                         _acceleratorDeviceName;
    bool                                                                _enablePerfCount;
    mutable std::atomic<bool>                                           _alreadyActualNetwork = {false};
};

}  // namespace AutoPlugin
