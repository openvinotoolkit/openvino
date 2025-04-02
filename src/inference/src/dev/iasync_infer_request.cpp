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
    stop_and_wait();
}

ov::IAsyncInferRequest::IAsyncInferRequest(const std::shared_ptr<IInferRequest>& request,
                                           const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                                           const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : m_sync_request(request),
      m_request_executor(task_executor),
      m_callback_executor(callback_executor) {
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

void ov::IAsyncInferRequest::wait() {
    // Just use the last '_futures' member to wait pipeline completion
    auto future = [this] {
        std::lock_guard<std::mutex> lock{m_mutex};
        return m_futures.empty() ? std::shared_future<void>{} : m_futures.back();
    }();
    if (future.valid()) {
        future.get();
    }
}

bool ov::IAsyncInferRequest::wait_for(const std::chrono::milliseconds& timeout) {
    OPENVINO_ASSERT(timeout >= std::chrono::milliseconds{0}, "Timeout can't be less than 0 for InferRequest::wait().");

    // Just use the last '_futures' member to wait pipeline completion
    auto future = [this] {
        std::lock_guard<std::mutex> lock{m_mutex};
        return m_futures.empty() ? std::shared_future<void>{} : m_futures.back();
    }();

    if (!future.valid()) {
        return false;
    }

    const auto status = future.wait_for(std::chrono::milliseconds{timeout});

    if (std::future_status::ready == status) {
        future.get();
        return true;
    } else {
        return false;
    }
}

void ov::IAsyncInferRequest::cancel() {
    std::lock_guard<std::mutex> lock{m_mutex};
    if (m_state == InferState::BUSY) {
        m_state = InferState::CANCELLED;
    }
}

void ov::IAsyncInferRequest::set_callback(std::function<void(std::exception_ptr)> callback) {
    check_state();
    m_callback = std::move(callback);
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::IAsyncInferRequest::query_state() const {
    check_state();
    return m_sync_request->query_state();
}

void ov::IAsyncInferRequest::infer_thread_unsafe() {
    run_first_stage(m_sync_pipeline.begin(), m_sync_pipeline.end(), m_sync_callback_executor);
}

void ov::IAsyncInferRequest::start_async_thread_unsafe() {
    run_first_stage(m_pipeline.begin(), m_pipeline.end(), m_callback_executor);
}

void ov::IAsyncInferRequest::run_first_stage(const Pipeline::iterator itBeginStage,
                                             const Pipeline::iterator itEndStage,
                                             const std::shared_ptr<ov::threading::ITaskExecutor> callbackExecutor) {
    auto& firstStageExecutor = std::get<Stage_e::EXECUTOR>(*itBeginStage);
    OPENVINO_ASSERT(nullptr != firstStageExecutor);
    firstStageExecutor->run(make_next_stage_task(itBeginStage, itEndStage, std::move(callbackExecutor)));
}

ov::threading::Task ov::IAsyncInferRequest::make_next_stage_task(
    const Pipeline::iterator itStage,
    const Pipeline::iterator itEndStage,
    const std::shared_ptr<ov::threading::ITaskExecutor> callbackExecutor) {
    return std::bind(
        [this, itStage, itEndStage](std::shared_ptr<ov::threading::ITaskExecutor>& callbackExecutor) mutable {
            std::exception_ptr currentException = nullptr;
            auto& thisStage = *itStage;
            auto itNextStage = itStage + 1;
            try {
                auto& stageTask = std::get<Stage_e::TASK>(thisStage);
                OPENVINO_ASSERT(nullptr != stageTask);
                stageTask();
                if (itEndStage != itNextStage) {
                    auto& nextStage = *itNextStage;
                    auto& nextStageExecutor = std::get<Stage_e::EXECUTOR>(nextStage);
                    OPENVINO_ASSERT(nullptr != nextStageExecutor);
                    nextStageExecutor->run(make_next_stage_task(itNextStage, itEndStage, std::move(callbackExecutor)));
                }
            } catch (...) {
                currentException = std::current_exception();
            }

            if ((itEndStage == itNextStage) || (nullptr != currentException)) {
                auto lastStageTask = [this, currentException]() mutable {
                    auto promise = std::move(m_promise);
                    std::function<void(std::exception_ptr)> callback;
                    {
                        std::lock_guard<std::mutex> lock{m_mutex};
                        m_state = InferState::IDLE;
                        std::swap(callback, m_callback);
                    }
                    if (callback) {
                        try {
                            callback(currentException);
                        } catch (...) {
                            currentException = std::current_exception();
                        }
                        std::lock_guard<std::mutex> lock{m_mutex};
                        if (!m_callback) {
                            std::swap(callback, m_callback);
                        }
                    }
                    if (nullptr == currentException) {
                        promise.set_value();
                    } else {
                        promise.set_exception(currentException);
                    }
                };

                if (nullptr == callbackExecutor) {
                    lastStageTask();
                } else {
                    callbackExecutor->run(std::move(lastStageTask));
                }
            }
        },
        std::move(callbackExecutor));
}

void ov::IAsyncInferRequest::start_async() {
    infer_impl([this] {
        start_async_thread_unsafe();
    });
}

void ov::IAsyncInferRequest::check_state() const {
    std::lock_guard<std::mutex> lock{m_mutex};
    switch (m_state) {
    case InferState::BUSY:
        ov::Busy::create("Infer Request is busy");
    case InferState::CANCELLED:
        ov::Cancelled::create("Infer Request was canceled");
    default:
        break;
    }
}

void ov::IAsyncInferRequest::check_cancelled_state() const {
    std::lock_guard<std::mutex> lock{m_mutex};
    if (m_state == InferState::CANCELLED)
        ov::Cancelled::create("Infer Request was canceled");
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
    Futures futures;
    InferState state = InferState::IDLE;
    {
        std::lock_guard<std::mutex> lock{m_mutex};
        state = m_state;
        if (state != InferState::STOP) {
            m_callback = {};
            m_state = InferState::STOP;
            futures = std::move(m_futures);
        }
    }
    if (state != InferState::STOP) {
        for (auto&& future : futures) {
            if (future.valid()) {
                future.wait();
            }
        }
    }
}

void ov::IAsyncInferRequest::infer() {
    DisableCallbackGuard disableCallbackGuard{this};
    infer_impl([this] {
        infer_thread_unsafe();
    });
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
