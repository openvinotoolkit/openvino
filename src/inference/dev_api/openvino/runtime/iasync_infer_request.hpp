// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime AsyncInferRequest interface
 * @file openvino/runtime/iasync_infer_request.hpp
 */

#pragma once

#include <future>
#include <memory>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/exception.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"

namespace ov {

/**
 * @brief Base class with default implementation of asynchronous multi staged inference request.
 *        To customize pipeline stages derived class should change the content
 *        of IAsyncInferRequest::m_pipeline member container.
 *        It consists of pairs of tasks and executors which will run the task.
 *        The class is recommended to be used by plugins as a base class for asynchronous inference request
 * implementation.
 * @note  To synchronize derived context with stages
 *        derived class should call IAsyncInferRequest::stop_and_wait() function in destructor.
 * @par Example
 *        Here is an example of asynchronous inference request implementation for some accelerator device.
 *        It uses 5 different executors to run different stages of a synchronous inference request.
 * @ingroup ov_dev_api_async_infer_request_api
 */
class OPENVINO_RUNTIME_API IAsyncInferRequest : public IInferRequest {
public:
    IAsyncInferRequest(const std::shared_ptr<IInferRequest>& request,
                       const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                       const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor);
    ~IAsyncInferRequest();

    /**
     * @brief Start inference of specified input(s) in asynchronous mode
     * @note The method returns immediately. Inference starts also immediately.
     */
    virtual void start_async();

    /**
     * @brief Waits for the result to become available.
     */
    virtual void wait();
    /**
     * @brief Waits for the result to become available. Blocks until specified timeout has elapsed or the result
     * becomes available, whichever comes first.
     * @param timeout - maximum duration in milliseconds to block for
     * @return A true if results are ready.
     */
    virtual bool wait_for(const std::chrono::milliseconds& timeout);

    /**
     * @brief Cancel current inference request execution
     */
    virtual void cancel();

    /**
     * @brief Set callback function which will be called on success or failure of asynchronous request
     * @param callback - function to be called with the following description:
     */
    virtual void set_callback(std::function<void(std::exception_ptr)> callback);

    /**
     * @brief Infers specified input(s) in synchronous mode
     * @note blocks all method of InferRequest while request is ongoing (running or waiting in queue)
     */
    void infer() override;

    /**
     * @brief Queries performance measures per layer to identify the most time consuming operation.
     * @note Not all plugins provide meaningful data.
     * @return Vector of profiling information for operations in a model.
     */
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    /**
     * @brief Gets an input/output tensor for inference.
     * @note If the tensor with the specified @p port is not found, an exception is thrown.
     * @param port Port of the tensor to get.
     * @return Tensor for the port @p port.
     */
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

    /**
     * @brief Sets an input/output tensor to infer.
     * @param port Port of the input or output tensor.
     * @param tensor Reference to a tensor. The element_type and shape of a tensor must match
     * the model's input/output element_type and size.
     */
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;

    /**
     * @brief Gets a batch of tensors for input data to infer by input port.
     * Model input must have batch dimension, and the number of @p tensors must match the batch size.
     * The current version supports setting tensors to model inputs only. If @p port is associated
     * with output (or any other non-input node), an exception is thrown.
     *
     * @param port Port of the input tensor.
     * @param tensors Input tensors for batched infer request. The type of each tensor must match the model
     * input element type and shape (except batch dimension). Total size of tensors must match the input size.
     * @return vector of tensors
     */
    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>& port) const override;
    /**
     * @brief Sets a batch of tensors for input data to infer by input port.
     * Model input must have batch dimension, and the number of @p tensors must match the batch size.
     * The current version supports setting tensors to model inputs only. If @p port is associated
     * with output (or any other non-input node), an exception is thrown.
     *
     * @param port Port of the input tensor.
     * @param tensors Input tensors for batched infer request. The type of each tensor must match the model
     * input element type and shape (except batch dimension). Total size of tensors must match the input size.
     */
    void set_tensors(const ov::Output<const ov::Node>& port,
                     const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

    /**
     * @brief Gets state control interface for the given infer request.
     *
     * State control essential for recurrent models.
     * @return Vector of Variable State objects.
     */
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

    /**
     * @brief Gets pointer to compiled model (usually synchronous request holds the compiled model)
     *
     * @return Pointer to the compiled model
     */
    const std::shared_ptr<const ov::ICompiledModel>& get_compiled_model() const override;

    /**
     * @brief Gets inputs for infer request
     *
     * @return vector of input ports
     */
    const std::vector<ov::Output<const ov::Node>>& get_inputs() const override;

    /**
     * @brief Gets outputs for infer request
     *
     * @return vector of output ports
     */
    const std::vector<ov::Output<const ov::Node>>& get_outputs() const override;

protected:
    using Stage = std::pair<std::shared_ptr<ov::threading::ITaskExecutor>, ov::threading::Task>;
    /**
     * @brief Pipeline is vector of stages
     */
    using Pipeline = std::vector<Stage>;

    /**
     * @brief Forbids pipeline start and wait for all started pipelines.
     * @note Should be called in derived class destructor to wait for completion of usage of derived context captured by
     * pipeline tasks
     */
    void stop_and_wait();

    /**
     * @brief Throws exception if inference request is busy or canceled
     */
    void check_state() const;
    /**
     * @brief Throws exception if inference request is cancelled
     */
    void check_cancelled_state() const;
    /**
     * @brief Performs inference of pipeline in syncronous mode
     * @note Used by Infer which ensures thread-safety and calls this method after.
     */
    virtual void infer_thread_unsafe();
    /**
     * @brief Starts an asynchronous pipeline thread unsafe.
     * @note Used by start_async which ensures thread-safety and calls this method after.
     */
    virtual void start_async_thread_unsafe();
    /**
     * @brief Check that all tensors are valid. Throws an exception if it's not.
     */
    void check_tensors() const override;

    Pipeline m_pipeline;       //!< Pipeline variable that should be filled by inherited class.
    Pipeline m_sync_pipeline;  //!< Synchronous pipeline variable that should be filled by inherited class.

private:
    enum InferState { IDLE, BUSY, CANCELLED, STOP };
    using Futures = std::vector<std::shared_future<void>>;
    enum Stage_e : std::uint8_t { EXECUTOR, TASK };
    InferState m_state = InferState::IDLE;
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
                         const std::shared_ptr<ov::threading::ITaskExecutor> callbackExecutor = {});

    ov::threading::Task make_next_stage_task(const Pipeline::iterator itStage,
                                             const Pipeline::iterator itEndStage,
                                             const std::shared_ptr<ov::threading::ITaskExecutor> callbackExecutor);

    template <typename F>
    void infer_impl(const F& f) {
        check_tensors();
        InferState state = InferState::IDLE;
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            state = m_state;
            switch (m_state) {
            case InferState::BUSY:
                ov::Busy::create("Infer Request is busy");
            case InferState::CANCELLED:
                ov::Cancelled::create("Infer Request was canceled");
            case InferState::IDLE: {
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
            case InferState::STOP:
                break;
            }
            m_state = InferState::BUSY;
        }
        if (state != InferState::STOP) {
            try {
                f();
            } catch (...) {
                m_promise.set_exception(std::current_exception());
                std::lock_guard<std::mutex> lock{m_mutex};
                m_state = InferState::IDLE;
                throw;
            }
        }
    }

    std::shared_ptr<IInferRequest> m_sync_request;

    std::shared_ptr<ov::threading::ITaskExecutor> m_request_executor;  //!< Used to run inference CPU tasks.
    std::shared_ptr<ov::threading::ITaskExecutor>
        m_callback_executor;  //!< Used to run post inference callback in asynchronous pipline
    std::shared_ptr<ov::threading::ITaskExecutor>
        m_sync_callback_executor;  //!< Used to run post inference callback in synchronous pipline
    mutable std::mutex m_mutex;
    std::function<void(std::exception_ptr)> m_callback;
};

}  // namespace ov
