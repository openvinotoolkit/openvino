// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/iasync_infer_request.hpp"

#include <memory>

#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/variable_state.hpp"
#include "threading/ie_immediate_executor.hpp"
#include "threading/ie_istreams_executor.hpp"

namespace {

struct ImmediateStreamsExecutor : public InferenceEngine::ITaskExecutor {
    explicit ImmediateStreamsExecutor(const InferenceEngine::IStreamsExecutor::Ptr& streamsExecutor)
        : _streamsExecutor{streamsExecutor} {}
    void run(InferenceEngine::Task task) override {
        _streamsExecutor->Execute(std::move(task));
    }
    InferenceEngine::IStreamsExecutor::Ptr _streamsExecutor;
};

}  // namespace

ov::IAsyncInferRequest::~IAsyncInferRequest() {
    stop_and_wait();
}

ov::IAsyncInferRequest::IAsyncInferRequest(const std::shared_ptr<IInferRequest>& request,
                                           const InferenceEngine::ITaskExecutor::Ptr& task_executor,
                                           const InferenceEngine::ITaskExecutor::Ptr& callback_executor)
    : IInferRequest(*request),
      m_pipeline{{m_request_executor,
                  [this] {
                      m_sync_request->infer();
                  }}},
      m_sync_pipeline{{std::make_shared<InferenceEngine::ImmediateExecutor>(),
                       [this] {
                           m_sync_request->infer();
                       }}},
      m_sync_request(request),
      m_request_executor(task_executor),
      m_callback_executor(callback_executor) {
    auto streams_executor = std::dynamic_pointer_cast<InferenceEngine::IStreamsExecutor>(m_request_executor);
    if (streams_executor != nullptr) {
        m_sync_pipeline = {{std::make_shared<ImmediateStreamsExecutor>(std::move(streams_executor)), [this] {
                                m_sync_request->infer();
                            }}};
    }
}

void ov::IAsyncInferRequest::infer_thread_unsafe() {
    run_first_stage(m_sync_pipeline.begin(), m_sync_pipeline.end(), m_sync_callback_executor);
}

void ov::IAsyncInferRequest::start_async_thread_unsafe() {
    run_first_stage(m_pipeline.begin(), m_pipeline.end(), m_callback_executor);
}

void ov::IAsyncInferRequest::run_first_stage(const Pipeline::iterator itBeginStage,
                                             const Pipeline::iterator itEndStage,
                                             const InferenceEngine::ITaskExecutor::Ptr callbackExecutor) {
    auto& firstStageExecutor = std::get<Stage_e::executor>(*itBeginStage);
    IE_ASSERT(nullptr != firstStageExecutor);
    firstStageExecutor->run(make_next_stage_task(itBeginStage, itEndStage, std::move(callbackExecutor)));
}

InferenceEngine::Task ov::IAsyncInferRequest::make_next_stage_task(
    const Pipeline::iterator itStage,
    const Pipeline::iterator itEndStage,
    const InferenceEngine::ITaskExecutor::Ptr callbackExecutor) {
    return std::bind(
        [this, itStage, itEndStage](InferenceEngine::ITaskExecutor::Ptr& callbackExecutor) mutable {
            std::exception_ptr currentException = nullptr;
            auto& thisStage = *itStage;
            auto itNextStage = itStage + 1;
            try {
                auto& stageTask = std::get<Stage_e::task>(thisStage);
                IE_ASSERT(nullptr != stageTask);
                stageTask();
                if (itEndStage != itNextStage) {
                    auto& nextStage = *itNextStage;
                    auto& nextStageExecutor = std::get<Stage_e::executor>(nextStage);
                    IE_ASSERT(nullptr != nextStageExecutor);
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
                        m_state = InferState::Idle;
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

void ov::IAsyncInferRequest::infer() {
    DisableCallbackGuard disableCallbackGuard{this};
    infer_impl([&] {
        infer_thread_unsafe();
    });
    wait();
}
void ov::IAsyncInferRequest::start_async() {
    infer_impl([&] {
        start_async_thread_unsafe();
    });
}

void ov::IAsyncInferRequest::wait() {
    // Just use the last '_futures' member to wait pipeline completion
    auto future = [&] {
        std::lock_guard<std::mutex> lock{m_mutex};
        return m_futures.empty() ? std::shared_future<void>{} : m_futures.back();
    }();

    if (!future.valid()) {
        return;
    }

    future.wait();
}

bool ov::IAsyncInferRequest::wait_for(const std::chrono::milliseconds& timeout) {
    OPENVINO_ASSERT(timeout >= std::chrono::milliseconds{0}, "Timeout can't be less than 0 for InferRequest::wait().");
    auto status = std::future_status::deferred;

    // Just use the last '_futures' member to wait pipeline completion
    auto future = [&] {
        std::lock_guard<std::mutex> lock{m_mutex};
        return m_futures.empty() ? std::shared_future<void>{} : m_futures.back();
    }();

    if (!future.valid()) {
        return false;
    }

    status = future.wait_for(std::chrono::milliseconds{timeout});

    if (std::future_status::ready == status) {
        future.get();
        return true;
    } else {
        return false;
    }
}

void ov::IAsyncInferRequest::cancel() {
    std::lock_guard<std::mutex> lock{m_mutex};
    if (m_state == InferState::Busy) {
        m_state = InferState::Cancelled;
    }
}

void ov::IAsyncInferRequest::check_state() const {
    std::lock_guard<std::mutex> lock{m_mutex};
    switch (m_state) {
    case InferState::Busy:
        IE_THROW(RequestBusy);
    case InferState::Cancelled:
        IE_THROW(InferCancelled);
    default:
        break;
    }
}

std::vector<ov::ProfilingInfo> ov::IAsyncInferRequest::get_profiling_info() const {
    check_state();
    return m_sync_request->get_profiling_info();
}

ov::Tensor ov::IAsyncInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    check_state();
    return m_sync_request->get_tensor(port);
}

void ov::IAsyncInferRequest::set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) {
    check_state();
    return m_sync_request->set_tensor(port, tensor);
}

std::vector<ov::Tensor> ov::IAsyncInferRequest::get_tensors(const ov::Output<const ov::Node>& port) const {
    check_state();
    return m_sync_request->get_tensors(port);
}

void ov::IAsyncInferRequest::set_tensors(const ov::Output<const ov::Node>& port,
                                         const std::vector<ov::Tensor>& tensors) {
    check_state();
    return m_sync_request->set_tensors(port, tensors);
}

std::vector<ov::VariableState> ov::IAsyncInferRequest::query_state() const {
    check_state();
    return m_sync_request->query_state();
}

void ov::IAsyncInferRequest::set_callback(std::function<void(std::exception_ptr)> callback) {
    check_state();
    m_callback = std::move(callback);
}

void ov::IAsyncInferRequest::stop_and_wait() {
    Futures futures;
    InferState state = InferState::Idle;
    {
        std::lock_guard<std::mutex> lock{m_mutex};
        state = m_state;
        if (state != InferState::Stop) {
            m_callback = {};
            m_state = InferState::Stop;
            futures = std::move(m_futures);
        }
    }
    if (state != InferState::Stop) {
        for (auto&& future : futures) {
            if (future.valid()) {
                future.wait();
            }
        }
    }
}

const ov::IAsyncInferRequest::InferState& ov::IAsyncInferRequest::get_state() const {
    return m_state;
}
