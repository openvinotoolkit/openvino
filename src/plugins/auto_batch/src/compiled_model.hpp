// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>

#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"
#include "ie_metric_helpers.hpp"
#include "plugin.hpp"
#include "threading/ie_thread_safe_containers.hpp"

namespace ov {
namespace autobatch_plugin {

class AutoBatchAsyncInferRequest;

class AutoBatchExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<AutoBatchExecutableNetwork>;
    struct WorkerInferRequest {
        using Ptr = std::shared_ptr<WorkerInferRequest>;
        InferenceEngine::SoIInferRequestInternal _inferRequestBatched;
        int _batchSize;
        InferenceEngine::ThreadSafeQueueWithSize<std::pair<AutoBatchAsyncInferRequest*, InferenceEngine::Task>> _tasks;
        std::vector<InferenceEngine::Task> _completionTasks;
        std::thread _thread;
        std::condition_variable _cond;
        std::mutex _mutex;
        std::exception_ptr _exceptionPtr;
    };

    explicit AutoBatchExecutableNetwork(
        const InferenceEngine::SoExecutableNetworkInternal& networkForDevice,
        const InferenceEngine::SoExecutableNetworkInternal& networkForDeviceWithoutBatch,
        const DeviceInformation& networkDevices,
        const std::unordered_map<std::string, InferenceEngine::Parameter>& config,
        const std::set<std::string>& batchedIntputs,
        const std::set<std::string>& batchedOutputs);

    void SetConfig(const std::map<std::string, InferenceEngine::Parameter>& config) override;
    InferenceEngine::Parameter GetConfig(const std::string& name) const override;
    InferenceEngine::Parameter GetMetric(const std::string& name) const override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(
        InferenceEngine::InputsDataMap networkInputs,
        InferenceEngine::OutputsDataMap networkOutputs) override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(
        const std::vector<std::shared_ptr<const ov::Node>>& inputs,
        const std::vector<std::shared_ptr<const ov::Node>>& outputs) override;
    std::shared_ptr<InferenceEngine::RemoteContext> GetContext() const override;
    std::shared_ptr<ngraph::Function> GetExecGraphInfo() override;
    virtual ~AutoBatchExecutableNetwork();

protected:
    static unsigned int ParseTimeoutValue(const std::string&);
    std::atomic_bool _terminate = {false};
    DeviceInformation _device;
    InferenceEngine::SoExecutableNetworkInternal _network;
    InferenceEngine::SoExecutableNetworkInternal _networkWithoutBatch;

    std::pair<WorkerInferRequest&, int> GetWorkerInferRequest();
    std::vector<WorkerInferRequest::Ptr> _workerRequests;
    std::mutex _workerRequestsMutex;

    std::unordered_map<std::string, InferenceEngine::Parameter> _config;
    bool _needPerfCounters = false;
    std::atomic_size_t _numRequestsCreated = {0};
    std::atomic_int _timeOut = {0};  // in ms

    const std::set<std::string> _batchedInputs;
    const std::set<std::string> _batchedOutputs;
};
}  // namespace autobatch_plugin
}  // namespace ov