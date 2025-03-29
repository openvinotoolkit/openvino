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

namespace ov {
class AsyncInferFsm : public IInferRequestFsm {
public:
    explicit AsyncInferFsm() : IInferRequestFsm{}, m_futures{}, m_promise{} {}

    void on_event(const Idle& state, const StopEvent& event) override {
        IInferRequestFsm::on_event(state, event);
        auto l = lock();
        m_futures.clear();
    }

    void on_event(const Busy& state, const StopEvent& event) override {
        IInferRequestFsm::on_event(state, event);
        Futures futures;
        {
            auto guard = lock();
            futures = std::move(m_futures);
        }
        for (auto&& future : futures) {
            if (future.valid()) {
                future.wait();
            }
        }
    }

    void on_event(const Idle& state, const StartEvent& event) override {
        auto guard = lock();
        m_futures.erase(std::remove_if(std::begin(m_futures),
                                       std::end(m_futures),
                                       [](auto&& future) {
                                           return future.valid() ? (std::future_status::ready ==
                                                                    future.wait_for(std::chrono::milliseconds{0}))
                                                                 : true;
                                       }),
                        m_futures.end());
        m_promise = {};
        m_futures.emplace_back(m_promise.get_future().share());
        guard.unlock();
        IInferRequestFsm::on_event(state, event);
        try {
            start_pipeline(event);
        } catch (...) {
            std::cout << "pipe throws" << std::endl;
            m_promise.set_exception(std::current_exception());
            guard.lock();
            m_state = Idle{};
            throw;
        }
    }

    void wait() override {
        // Just use the last '_futures' member to wait pipeline completion
        if (auto future = get_last_future(); future.valid()) {
            future.get();
        }
    }

    virtual bool wait_for(const std::chrono::milliseconds& timeout) override {
        auto has_result = false;
        // Just use the last '_futures' member to wait pipeline completion
        if (auto future = get_last_future(); future.valid()) {
            if (auto status = future.wait_for(timeout); std::future_status::ready == status) {
                future.get();
                has_result = true;
            }
        }
        return has_result;
    }

private:
    std::shared_future<void> get_last_future() {
        const auto fsm_lock = lock();
        return m_futures.empty() ? std::shared_future<void>{} : m_futures.back();
    }

    void start_pipeline(const StartEvent& event) {
        auto& [first_stage_executor, first_stage_task] = *event.first_stage;
        OPENVINO_ASSERT(nullptr != first_stage_executor);
        first_stage_executor->run(make_next_stage_task(event.first_stage, event.last_stage, event.callback_executor));
    }

    threading::Task make_next_stage_task(const Pipeline::iterator itStage,
                                         const Pipeline::iterator itEndStage,
                                         const threading::ITaskExecutor::Ptr callbackExecutor) {
        return std::bind(
            [this, itStage, itEndStage](auto& callbackExecutor) mutable {
                std::exception_ptr currentException = nullptr;
                auto& thisStage = *itStage;
                auto itNextStage = itStage + 1;
                try {
                    auto& [stage_exe, stageTask] = thisStage;
                    OPENVINO_ASSERT(nullptr != stageTask);
                    stageTask();
                    if (itEndStage != itNextStage) {
                        auto& nextStage = *itNextStage;
                        auto& [nextStageExecutor, _] = nextStage;
                        OPENVINO_ASSERT(nullptr != nextStageExecutor);
                        nextStageExecutor->run(
                            make_next_stage_task(itNextStage, itEndStage, std::move(callbackExecutor)));
                    }
                } catch (...) {
                    currentException = std::current_exception();
                }

                if ((itEndStage == itNextStage) || (nullptr != currentException)) {
                    auto lastStageTask = [this, currentException]() mutable {
                        auto promise = std::move(m_promise);
                        std::function<void(std::exception_ptr)> callback;
                        {
                            auto guard = lock();
                            m_state = Idle{};
                            std::swap(callback, m_callback);
                        }
                        if (callback) {
                            try {
                                callback(currentException);
                            } catch (...) {
                                currentException = std::current_exception();
                            }

                            if (auto guard = lock(); !m_callback) {
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

    using Futures = std::vector<std::shared_future<void>>;
    Futures m_futures;
    std::promise<void> m_promise;
};
}  // namespace ov

ov::IAsyncInferRequest::~IAsyncInferRequest() {
    stop_and_wait();
}

ov::IAsyncInferRequest::IAsyncInferRequest(const std::shared_ptr<IInferRequest>& request,
                                           const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                                           const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : IAsyncInferRequest{request, task_executor, callback_executor, std::make_unique<AsyncInferFsm>()} {
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
                                           std::unique_ptr<IInferRequestFsm> fsm)
    : m_pipeline{},
      m_sync_pipeline{},
      m_request_fsm{std::move(fsm)},
      m_mutex{},
      m_sync_request{request},
      m_request_executor{task_executor},
      m_callback_executor{callback_executor},
      m_sync_callback_executor{} {}

void ov::IAsyncInferRequest::wait() {
    m_request_fsm->wait();
}

bool ov::IAsyncInferRequest::wait_for(const std::chrono::milliseconds& timeout) {
    OPENVINO_ASSERT(timeout >= std::chrono::milliseconds{0}, "Timeout can't be less than 0 for InferRequest::wait().");
    return m_request_fsm->wait_for(timeout);
}

void ov::IAsyncInferRequest::cancel() {
    m_request_fsm->cancel();
}

void ov::IAsyncInferRequest::set_callback(std::function<void(std::exception_ptr)> callback) {
    check_state();
    m_request_fsm->set_callback(std::move(callback));
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::IAsyncInferRequest::query_state() const {
    check_state();
    return m_sync_request->query_state();
}

void ov::IAsyncInferRequest::infer_thread_unsafe() {
    m_request_fsm->start(m_sync_pipeline.begin(), m_sync_pipeline.end(), m_sync_callback_executor);
}

void ov::IAsyncInferRequest::start_async_thread_unsafe() {
    m_request_fsm->start(m_pipeline.begin(), m_pipeline.end(), m_callback_executor);
}

void ov::IAsyncInferRequest::check_state() const {
    if (std::lock_guard<std::mutex> lock{m_mutex}; m_request_fsm->is_busy()) {
        ov::Busy::create("Infer Request is busy");
    } else if (m_request_fsm->is_cancelled()) {
        ov::Cancelled::create("Infer Request was canceled");
    }
}

void ov::IAsyncInferRequest::check_cancelled_state() const {
    if (std::lock_guard<std::mutex> lock{m_mutex}; m_request_fsm->is_cancelled()) {
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
    m_request_fsm->stop();
    wait();
}

void ov::IAsyncInferRequest::start_async() {
    check_tensors();
    start_async_thread_unsafe();
}

void ov::IAsyncInferRequest::infer() {
    check_tensors();
    const auto disable_callback_guard = m_request_fsm->disable_callback();
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
