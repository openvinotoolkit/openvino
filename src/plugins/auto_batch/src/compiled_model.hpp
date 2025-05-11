// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <condition_variable>
#include <thread>

#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/threading/thread_safe_containers.hpp"
#include "plugin.hpp"

namespace ov {
namespace autobatch_plugin {

class AsyncInferRequest;

class CompiledModel : public ov::ICompiledModel {
public:
    struct WorkerInferRequest {
        ov::SoPtr<ov::IAsyncInferRequest> _infer_request_batched;
        int _batch_size;
        ov::threading::ThreadSafeQueueWithSize<std::pair<ov::autobatch_plugin::AsyncInferRequest*, ov::threading::Task>>
            _tasks;
        std::vector<ov::threading::Task> _completion_tasks;
        std::thread _thread;
        std::condition_variable _cond;
        std::mutex _mutex;
        std::exception_ptr _exception_ptr;
        bool _is_wakeup;
    };

    CompiledModel(const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const ov::AnyMap& config,
                  const DeviceInformation& device_info,
                  const std::set<std::size_t>& batched_inputs,
                  const std::set<std::size_t>& batched_outputs,
                  const ov::SoPtr<ov::ICompiledModel>& compiled_model_with_batch,
                  const ov::SoPtr<ov::ICompiledModel>& compiled_model_without_batch,
                  const ov::SoPtr<ov::IRemoteContext>& context);

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name) const override;

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void export_model(std::ostream& model) const override;

    virtual ~CompiledModel();

    const std::vector<ov::Output<const ov::Node>>& outputs() const override;

    const std::vector<ov::Output<const ov::Node>>& inputs() const override;

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
    static unsigned int ParseTimeoutValue(const std::string&);
    std::atomic_bool m_terminate = {false};
    ov::AnyMap m_config;
    DeviceInformation m_device_info;

    std::pair<std::shared_ptr<ov::autobatch_plugin::CompiledModel::WorkerInferRequest>, int> GetWorkerInferRequest()
        const;
    mutable std::vector<std::shared_ptr<WorkerInferRequest>> m_worker_requests;
    mutable std::mutex m_worker_requests_mutex;

    mutable std::atomic_size_t m_num_requests_created = {0};
    std::atomic<std::uint32_t> m_time_out = {0};  // in ms

    const std::set<std::size_t> m_batched_inputs;
    const std::set<std::size_t> m_batched_outputs;

    ov::SoPtr<ov::ICompiledModel> m_compiled_model_with_batch;
    ov::SoPtr<ov::ICompiledModel> m_compiled_model_without_batch;
};
}  // namespace autobatch_plugin
}  // namespace ov
