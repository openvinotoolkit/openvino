
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime AsyncInferRequest interface
 * @file iasync_nfer_request.hpp
 */

#pragma once

#include <future>
#include <memory>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/tensor.hpp"
#include "threading/ie_itask_executor.hpp"

namespace ov {

class IInferRequest;

class OPENVINO_RUNTIME_API IAsyncInferRequest {
public:
    enum InferState { Idle, Busy, Cancelled, Stop };
    IAsyncInferRequest(const std::shared_ptr<ISyncInferRequest>& request,
                       const InferenceEngine::ITaskExecutor::Ptr& task_executor,
                       const InferenceEngine::ITaskExecutor::Ptr& callback_executor);
    ~IAsyncInferRequest();

    virtual void start_async();

    virtual void wait();
    virtual bool wait_for(const std::chrono::milliseconds& timeout);

    virtual void cancel();

    virtual std::vector<ov::ProfilingInfo> get_profiling_info() const;

    virtual ov::Tensor get_tensor(const ov::Output<const ov::Node>& port) const;
    virtual void set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor);

    virtual std::vector<ov::Tensor> get_tensors(const ov::Output<const ov::Node>& port) const;
    virtual void set_tensors(const ov::Output<const ov::Node>& port, const std::vector<ov::Tensor>& tensors);

    virtual std::vector<ov::VariableState> query_state() const;

    virtual void set_callback(std::function<void(std::exception_ptr)> callback);

    const InferState& get_state() const;

protected:
    using Stage = std::pair<InferenceEngine::ITaskExecutor::Ptr, InferenceEngine::Task>;
    /**
     * @brief Pipeline is vector of stages
     */
    using Pipeline = std::vector<Stage>;

    void stop_and_wait();
    void check_state() const;
    /**
     * @brief Performs inference of pipeline in syncronous mode
     * @note Used by Infer which ensures thread-safety and calls this method after.
     */
    virtual void infer_thread_unsafe();
    virtual void start_async_thread_unsafe();

    Pipeline m_pipeline;       //!< Pipeline variable that should be filled by inherited class.
    Pipeline m_sync_pipeline;  //!< Synchronous pipeline variable that should be filled by inherited class.

private:
    using Futures = std::vector<std::shared_future<void>>;
    enum Stage_e : std::uint8_t { executor, task };
    InferState m_state = InferState::Idle;
    Futures m_futures;
    std::promise<void> m_promise;

    friend struct DisableCallbackGuard;
    struct DisableCallbackGuard {
        explicit DisableCallbackGuard(IAsyncInferRequest* this_) : _this{this_} {
            std::lock_guard<std::mutex> lock{_this->m_mutex};
            std::swap(m_callback, _this->m_callback);
        }
        ~DisableCallbackGuard() {
            std::lock_guard<std::mutex> lock{_this->m_mutex};
            _this->m_callback = m_callback;
        }
        IAsyncInferRequest* _this = nullptr;
        std::function<void(std::exception_ptr)> m_callback;
    };

    void run_first_stage(const Pipeline::iterator itBeginStage,
                         const Pipeline::iterator itEndStage,
                         const InferenceEngine::ITaskExecutor::Ptr callbackExecutor = {});

    InferenceEngine::Task make_next_stage_task(const Pipeline::iterator itStage,
                                               const Pipeline::iterator itEndStage,
                                               const InferenceEngine::ITaskExecutor::Ptr callbackExecutor);

    template <typename F>
    void infer_impl(const F& f) {
        m_sync_request->check_tensors();
        InferState state = InferState::Idle;
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            state = m_state;
            switch (m_state) {
            case InferState::Busy:
                IE_THROW(RequestBusy);
            case InferState::Cancelled:
                IE_THROW(InferCancelled);
            case InferState::Idle: {
                m_futures.erase(std::remove_if(std::begin(m_futures),
                                               std::end(m_futures),
                                               [](const std::shared_future<void>& future) {
                                                   if (future.valid()) {
                                                       return (std::future_status::ready ==
                                                               future.wait_for(std::chrono::milliseconds{0}));
                                                   } else {
                                                       return true;
                                                   }
                                               }),
                                m_futures.end());
                m_promise = {};
                m_futures.emplace_back(m_promise.get_future().share());
            } break;
            case InferState::Stop:
                break;
            }
            m_state = InferState::Busy;
        }
        if (state != InferState::Stop) {
            try {
                f();
            } catch (...) {
                m_promise.set_exception(std::current_exception());
                std::lock_guard<std::mutex> lock{m_mutex};
                m_state = InferState::Idle;
                throw;
            }
        }
    }

    std::shared_ptr<ISyncInferRequest> m_sync_request;

    InferenceEngine::ITaskExecutor::Ptr m_request_executor;  //!< Used to run inference CPU tasks.
    InferenceEngine::ITaskExecutor::Ptr
        m_callback_executor;  //!< Used to run post inference callback in asynchronous pipline
    InferenceEngine::ITaskExecutor::Ptr
        m_sync_callback_executor;  //!< Used to run post inference callback in synchronous pipline
    mutable std::mutex m_mutex;
    std::function<void(std::exception_ptr)> m_callback;

    friend ov::IInferRequest;
};

}  // namespace ov
