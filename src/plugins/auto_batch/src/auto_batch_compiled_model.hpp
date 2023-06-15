// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <condition_variable>
#include <thread>

#include "auto_batch_plugin.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "threading/ie_thread_safe_containers.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"

namespace ov {
namespace autobatch_plugin {

class AsyncInferRequest;
/**
 * @class CompiledModel
 * @brief Implementation of compiled model
 */

class CompiledModel : public ov::ICompiledModel {
public:
    struct WorkerInferRequest {
        // using Ptr = std::shared_ptr<WorkerInferRequest>;
        ov::SoPtr<ov::IAsyncInferRequest> _inferRequestBatched;
        int _batchSize;
        InferenceEngine::ThreadSafeQueueWithSize<
            std::pair<ov::autobatch_plugin::AsyncInferRequest*, ov::threading::Task>>
            _tasks;
        std::vector<ov::threading::Task> _completionTasks;
        std::thread _thread;
        std::condition_variable _cond;
        std::mutex _mutex;
        std::exception_ptr _exceptionPtr;
    };

    CompiledModel(const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const ov::AnyMap& config,
                  const DeviceInformation& deviceInfo,
                  const std::set<std::string>& batchedInputs,
                  const std::set<std::string>& batchedOutputs,
                  const ov::SoPtr<ov::ICompiledModel>& compiledModelWithBatch,
                  const ov::SoPtr<ov::ICompiledModel>& compiledModelWithoutBatch);
    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;
    std::shared_ptr<ov::IRemoteContext> get_context() const;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;
    void export_model(std::ostream& model) const override;
    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;
    virtual ~CompiledModel();

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
    ov::AnyMap m_config;
    DeviceInformation m_deviceInfo;
    const std::set<std::string> m_batchedInputs;
    const std::set<std::string> m_batchedOutputs;
    ov::SoPtr<ov::ICompiledModel> m_compiledModelWithBatch;
    ov::SoPtr<ov::ICompiledModel> m_compiledModelWithoutBatch;
    std::atomic_uint32_t m_timeOut = {0};  // in ms
    std::pair<std::shared_ptr<ov::autobatch_plugin::CompiledModel::WorkerInferRequest>, int> GetWorkerInferRequest() const;
    mutable std::vector<std::shared_ptr<WorkerInferRequest>> _workerRequests;
    mutable std::mutex _workerRequestsMutex;
    bool _needPerfCounters = false;
    mutable std::atomic_size_t _numRequestsCreated = {0};
    std::atomic_bool _terminate = {false};
};

}  // namespace autobatch_plugin
}  // namespace ov
