// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/iasync_infer_request.hpp"

#include <memory>

#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/threading/immediate_executor.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "openvino/runtime/variable_state.hpp"

namespace {

struct ImmediateStreamsExecutor : public ov::threading::ITaskExecutor {
    explicit ImmediateStreamsExecutor(const std::shared_ptr<ov::threading::IStreamsExecutor>& streamsExecutor)
        : _streamsExecutor{streamsExecutor} {}
    void run(ov::threading::Task task) override {
        if (_streamsExecutor->get_streams_num() > 1) {
            std::vector<ov::threading::Task> tasks{std::move(task)};
            _streamsExecutor->run_and_wait(tasks);
        } else {
            _streamsExecutor->execute(std::move(task));
        }
    }
    std::shared_ptr<ov::threading::IStreamsExecutor> _streamsExecutor;
};

}  // namespace

ov::IAsyncInferRequest::~IAsyncInferRequest() {
    if (m_fsm && m_pipeline_process) {
        stop_and_wait();
    }
}

ov::IAsyncInferRequest::IAsyncInferRequest(const std::shared_ptr<IInferRequest>& request,
                                           const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                                           const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : IAsyncInferRequest{request, task_executor, callback_executor, std::make_unique<InferRequestFsm>(), {}} {
    if (m_request_executor && m_sync_request)
        m_pipeline = {{m_request_executor, [this] {
                           m_sync_request->infer();
                       }}};
    if (m_sync_request)
        m_sync_pipeline = {{std::make_shared<ov::threading::ImmediateExecutor>(), [this] {
                                m_sync_request->infer();
                            }}};
    auto streams_executor = std::dynamic_pointer_cast<ov::threading::IStreamsExecutor>(m_request_executor);
    if (streams_executor != nullptr) {
        m_sync_pipeline = {{std::make_shared<ImmediateStreamsExecutor>(std::move(streams_executor)), [this] {
                                m_sync_request->infer();
                            }}};
    }
}

ov::IAsyncInferRequest::IAsyncInferRequest(const std::shared_ptr<IInferRequest>& request,
                                           const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                                           const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor,
                                           std::unique_ptr<InferRequestFsm> fsm,
                                           std::unique_ptr<IPipelineProcess> pipeline_process)
    : m_pipeline{},
      m_sync_pipeline{},
      m_fsm{std::move(fsm)},
      m_pipeline_process{std::move(pipeline_process)},
      m_sync_request{request},
      m_request_executor{task_executor},
      m_callback_executor{callback_executor},
      m_sync_callback_executor{} {
    if (!m_pipeline_process) {
        m_pipeline_process = std::make_unique<PipelineProcess>([this]() {
            m_fsm->on_event(InferRequestFsm::DoneEvent{});
        });
    }
}

void ov::IAsyncInferRequest::wait() {
    m_pipeline_process->wait();
}

bool ov::IAsyncInferRequest::wait_for(const std::chrono::milliseconds& timeout) {
    OPENVINO_ASSERT(timeout >= std::chrono::milliseconds{0}, "Timeout can't be less than 0 for InferRequest::wait().");
    return m_pipeline_process->wait_for(timeout);
}

void ov::IAsyncInferRequest::cancel() {
    m_fsm->on_event(InferRequestFsm::CancelEvent{});
}

void ov::IAsyncInferRequest::set_callback(std::function<void(std::exception_ptr)> callback) {
    check_state();
    m_pipeline_process->set_callback(std::move(callback));
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::IAsyncInferRequest::query_state() const {
    check_state();
    return m_sync_request->query_state();
}

void ov::IAsyncInferRequest::infer_thread_unsafe() {
    InferRequestFsm::StartEvent event{m_sync_pipeline.begin(),
                                      m_sync_pipeline.end(),
                                      m_sync_callback_executor,
                                      m_pipeline_process->sync_pipeline_func()};

    try {
        if (m_fsm->is_ready()) {
            m_pipeline_process->prepare_sync();
        }
        m_fsm->on_event(event);
    } catch (...) {
        m_pipeline_process->set_exception(std::current_exception());
    }
}

void ov::IAsyncInferRequest::start_async_thread_unsafe() {
    InferRequestFsm::StartEvent event{m_pipeline.begin(),
                                      m_pipeline.end(),
                                      m_callback_executor,
                                      m_pipeline_process->async_pipeline_func()};

    try {
        if (m_fsm->is_ready()) {
            m_pipeline_process->prepare_async();
        }
        m_fsm->on_event(event);
    } catch (...) {
        m_pipeline_process->set_exception(std::current_exception());
    }
}

void ov::IAsyncInferRequest::check_state() const {
    if (auto lock = m_fsm->lock(); m_fsm->is_busy()) {
        ov::Busy::create("Infer Request is busy");
    } else if (m_fsm->is_cancelled()) {
        ov::Cancelled::create("Infer Request was canceled");
    }
}

void ov::IAsyncInferRequest::check_cancelled_state() const {
    if (auto lock = m_fsm->lock(); m_fsm->is_cancelled()) {
        ov::Cancelled::create("Infer Request was canceled");
    }
}

std::vector<ov::ProfilingInfo> ov::IAsyncInferRequest::get_profiling_info() const {
    check_state();
    return m_sync_request->get_profiling_info();
}

ov::SoPtr<ov::ITensor> ov::IAsyncInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    check_state();
    return m_sync_request->get_tensor(port);
}

void ov::IAsyncInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
    check_state();
    return m_sync_request->set_tensor(port, tensor);
}

std::vector<ov::SoPtr<ov::ITensor>> ov::IAsyncInferRequest::get_tensors(const ov::Output<const ov::Node>& port) const {
    check_state();
    return m_sync_request->get_tensors(port);
}

void ov::IAsyncInferRequest::set_tensors(const ov::Output<const ov::Node>& port,
                                         const std::vector<ov::SoPtr<ov::ITensor>>& tensors) {
    check_state();
    return m_sync_request->set_tensors(port, tensors);
}

void ov::IAsyncInferRequest::stop_and_wait() {
    m_fsm->on_event(InferRequestFsm::StopEvent{});
    m_pipeline_process->stop();
    wait();
}

void ov::IAsyncInferRequest::start_async() {
    check_tensors();
    start_async_thread_unsafe();
}

void ov::IAsyncInferRequest::infer() {
    check_tensors();
    auto g = m_pipeline_process->disable_callback();
    infer_thread_unsafe();
    wait();
}

void ov::IAsyncInferRequest::check_tensors() const {
    m_sync_request->check_tensors();
}

const std::shared_ptr<const ov::ICompiledModel>& ov::IAsyncInferRequest::get_compiled_model() const {
    return m_sync_request->get_compiled_model();
}

const std::vector<ov::Output<const ov::Node>>& ov::IAsyncInferRequest::get_inputs() const {
    return m_sync_request->get_inputs();
}
const std::vector<ov::Output<const ov::Node>>& ov::IAsyncInferRequest::get_outputs() const {
    return m_sync_request->get_outputs();
}
