// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <condition_variable>
#include <thread>

#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "plugin.hpp"
#include "threading/ie_thread_safe_containers.hpp"

namespace ov {
namespace autobatch_plugin {

class AsyncInferRequest;

class CompiledModel : public ov::ICompiledModel {
public:
    struct WorkerInferRequest {
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
                  const DeviceInformation& device_info,
                  const std::set<std::string>& batched_inputs,
                  const std::set<std::string>& batched_outputs,
                  const ov::SoPtr<ov::ICompiledModel>& compiled_Model_with_batch,
                  const ov::SoPtr<ov::ICompiledModel>& compiled_Model_Without_batch);

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name) const override;

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

    std::shared_ptr<ov::IRemoteContext> get_context() const;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void export_model(std::ostream& model) const override;

    virtual ~CompiledModel();

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
    static unsigned int ParseTimeoutValue(const std::string&);
    std::atomic_bool m_terminate = {false};
    DeviceInformation m_device_info;
    ov::SoPtr<ov::ICompiledModel> m_compiled_model_with_batch;
    ov::SoPtr<ov::ICompiledModel> m_compiled_model_without_batch;

    std::pair<std::shared_ptr<ov::autobatch_plugin::CompiledModel::WorkerInferRequest>, int> GetWorkerInferRequest()
        const;
    mutable std::vector<std::shared_ptr<WorkerInferRequest>> m_worker_requests;
    mutable std::mutex m_worker_requests_mutex;

    ov::AnyMap m_config;
    mutable std::atomic_size_t m_num_requests_created = {0};
    std::atomic_uint32_t m_timeOut = {0};  // in ms

    const std::set<std::string> m_batched_inputs;
    const std::set<std::string> m_batched_outputs;
};
}  // namespace autobatch_plugin
}  // namespace ov