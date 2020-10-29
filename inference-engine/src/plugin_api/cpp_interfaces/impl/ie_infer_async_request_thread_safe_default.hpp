// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <threading/ie_immediate_executor.hpp>
#include <threading/ie_itask_executor.hpp>

#include <cpp_interfaces/interface/ie_iinfer_async_request_internal.hpp>
#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_internal.hpp>
#include <cpp_interfaces/exception2status.hpp>
#include <ie_system_conf.h>

#include <exception>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace InferenceEngine {

/**
 * @ingroup ie_dev_api_async_infer_request_api
 * @brief Base class with default implementation of asynchronous multi staged inference request.
 *        To customize pipeline stages derived class should change the content
 *        of AsyncInferRequestThreadSafeDefault::_pipeline member container.
 *        It consists of pairs of tasks and executors which will run the task.
 *        The class is recommended to be used by plugins as a base class for asynchronous inference request implementation.
 * @note  To synchronize derived context with stages
 *        derived class should call AsyncInferRequestThreadSafeDefault::StopAndWait() function in destructor.
 * @par Example
 *        Here is an example of asynchronous inference request implementation for some accelerator device.
 *        It uses 5 different executors to run different stages of a synchronous inference request.
 *
 * @snippet example_async_infer_request.cpp async_infer_request:define_pipeline
 */
class AsyncInferRequestThreadSafeDefault : public AsyncInferRequestThreadSafeInternal {
public:
    /**
     * @brief A shared pointer to AsyncInferRequestThreadSafeDefault
     */
    using Ptr = std::shared_ptr<AsyncInferRequestThreadSafeDefault>;

    /**
     * @brief      Wraps a InferRequestInternal::Ptr implementation and constructs a 
     * AsyncInferRequestThreadSafeDefault::_pipeline where `taskExecutor` is used to run InferRequestInternal::Infer
     * asynchronously.
     *
     * @param[in]  request           The synchronous request
     * @param[in]  taskExecutor      The task executor
     * @param[in]  callbackExecutor  The callback executor
     */
    AsyncInferRequestThreadSafeDefault(const InferRequestInternal::Ptr& request,
                                       const ITaskExecutor::Ptr& taskExecutor,
                                       const ITaskExecutor::Ptr& callbackExecutor)
        : _requestExecutor {taskExecutor},
          _callbackExecutor {callbackExecutor},
          _syncRequest {request} {
        _pipeline = {
            { _requestExecutor, [this] { _syncRequest->Infer(); } }
        };
    }

    /**
     * @brief      Destroys the object, stops AsyncInferRequestThreadSafeDefault::_pipeline and waits for a finish.
     */
    ~AsyncInferRequestThreadSafeDefault() {
        StopAndWait();
    }

    /**
     * @brief Waits for completion of all pipeline stages
     *        If the pipeline raises an exception it will be rethrown here
     * @param millis_timeout A timeout is `ms` to wait or special enum value of IInferRequest::WaitMode
     * @return A status code
     */
    StatusCode Wait(int64_t millis_timeout) override {
        if (millis_timeout < IInferRequest::WaitMode::RESULT_READY) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str + "Timeout can't be less "
                               << IInferRequest::WaitMode::RESULT_READY << " for InferRequest::Wait\n";
        }
        auto status = std::future_status::deferred;

        // Just use the last '_futures' member to wait pipeline completion
        auto future = [&] {
            std::lock_guard<std::mutex> lock {_mutex};
            return _futures.empty() ? std::shared_future<void> {} : _futures.back();
        }();

        if (!future.valid()) {
            return StatusCode::INFER_NOT_STARTED;
        }

        switch (millis_timeout) {
        case IInferRequest::WaitMode::RESULT_READY: {
            future.wait();
            status = std::future_status::ready;
        } break;
        case IInferRequest::WaitMode::STATUS_ONLY: {
            status = future.wait_for(std::chrono::milliseconds {0});
        } break;
        default: {
            status = future.wait_for(std::chrono::milliseconds {millis_timeout});
        } break;
        }

        if (std::future_status::ready == status) {
            future.get();
            return StatusCode::OK;
        } else {
            return StatusCode::RESULT_NOT_READY;
        }
    }

    /**
     * @brief Sets the pointer to public interface.
     * @note Needed to correctly handle ownership between objects
     * @param[in]  ptr A shared pointer to a public IInferRequest interface.
     */
    void SetPointerToPublicInterface(InferenceEngine::IInferRequest::Ptr ptr) {
        _publicInterface = std::shared_ptr<IInferRequest>(ptr.get(), [](IInferRequest*) {});
    }

protected:
    /**
     * @brief Each pipeline stage is a @ref Task that is executed by specified ITaskExecutor implementation
     */
    using Stage = std::pair<ITaskExecutor::Ptr, Task>;
    /**
     * @brief Pipeline is vector of stages
     */
    using Pipeline = std::vector<Stage>;

    /**
     * @brief Creates and run the first stage task. If destructor was not called add a new std::future to the
     * AsyncInferRequestThreadSafeDefault::_futures list that would be used to wait
     * AsyncInferRequestThreadSafeDefault::_pipeline finish
     */
    void RunFirstStage() {
        _itStage = _pipeline.begin();
        _promise = {};
        bool stop = [&] {
            std::lock_guard<std::mutex> lock(_mutex);
            if (!_stop) {
                _futures.erase(std::remove_if(std::begin(_futures), std::end(_futures),
                                              [](const std::shared_future<void>& future) {
                                                  if (future.valid()) {
                                                      return (std::future_status::ready ==
                                                              future.wait_for(std::chrono::milliseconds {0}));
                                                  } else {
                                                      return true;
                                                  }
                                              }),
                               _futures.end());

                _futures.emplace_back(_promise.get_future().share());
            }
            return _stop;
        }();

        if (!stop) {
            try {
                auto& firstStageExecutor = std::get<Stage_e::executor>(*_itStage);
                IE_ASSERT(nullptr != firstStageExecutor);
                firstStageExecutor->run(MakeNextStageTask());
            } catch (...) {
                _promise.set_exception(std::current_exception());
                throw;
            }
        }
    }

    /**
     * @brief Forbids pipeline start and wait for all started pipelines.
     * @note Should be called in derived class destructor to wait for completion of usage of derived context captured by
     * pipeline tasks
     */
    void StopAndWait() {
        _callback = nullptr;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (!_stop) {
                _stop = true;
                for (auto&& future : _futures) {
                    if (future.valid()) {
                        future.wait();
                    }
                }
            }
        }
    }

    /**
     * @brief Implements Infer() using StartAsync() and Wait()
     */
    void InferUsingAsync() {
        struct CallbackStorage {
            explicit CallbackStorage(AtomicCallback& callback)
                : _callbackRef(callback), _callback(callback.exchange(nullptr)) {}
            ~CallbackStorage() {
                _callbackRef = _callback;
            }
            AtomicCallback& _callbackRef;
            IInferRequest::CompletionCallback _callback;
        } storage {_callback};
        StartAsync_ThreadUnsafe();
        Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    }

    ITaskExecutor::Ptr _requestExecutor;  //!< Used to run inference CPU tasks
    ITaskExecutor::Ptr _callbackExecutor;  //!< Used to run post inference callback
    Pipeline _pipeline;  //!< Pipeline variable that should be filled by inherited class.

    void StartAsync_ThreadUnsafe() override {
        _syncRequest->checkBlobs();
        RunFirstStage();
    }

    void Infer_ThreadUnsafe() override {
        _syncRequest->checkBlobs();
        _syncRequest->InferImpl();
    }

    void GetPerformanceCounts_ThreadUnsafe(std::map<std::string, InferenceEngineProfileInfo>& perfMap) const override {
        _syncRequest->GetPerformanceCounts(perfMap);
    }

    void SetBlob_ThreadUnsafe(const char* name, const Blob::Ptr& data) override {
        _syncRequest->SetBlob(name, data);
    }

    void SetBlob_ThreadUnsafe(const char* name, const Blob::Ptr& data, const PreProcessInfo& info) override {
        _syncRequest->SetBlob(name, data, info);
    }

    void GetBlob_ThreadUnsafe(const char* name, Blob::Ptr& data) override {
        _syncRequest->GetBlob(name, data);
    }

    void GetPreProcess_ThreadUnsafe(const char* name, const PreProcessInfo** info) const override {
        _syncRequest->GetPreProcess(name, info);
    }

    void SetCompletionCallback_ThreadUnsafe(IInferRequest::CompletionCallback callback) override {
        _callback = callback;
    }

    void GetUserData_ThreadUnsafe(void** data) override {
        if (data == nullptr) THROW_IE_EXCEPTION << NOT_ALLOCATED_str;
        *data = _userData;
    }

    void SetUserData_ThreadUnsafe(void* data) override {
        _userData = data;
    }

    void SetBatch_ThreadUnsafe(int batch) override {
        _syncRequest->SetBatch(batch);
    }

private:
    using AtomicCallback = std::atomic<IInferRequest::CompletionCallback>;
    using Futures = std::vector<std::shared_future<void>>;
    using Promise = std::shared_ptr<std::promise<void>>;
    enum Stage_e : std::uint8_t { executor, task };

    /**
     * @brief Create a task with next pipeline stage.
     *        Each call to MakeNextStageTask() generates `InferenceEngine::Task` objects for each stage.
     *        When stage task is called it increments
     *        `_stage` counter, call `_pipeline` task for this stage and generates next stage task using
     * MakeNextStageTask() and pass it to executor. On last stage or if the exception is raised from `_pipeline` task
     * the last stage task is called or passed to callback executor if it is presented. The last stage task call the
     * callback, if it is presented, capture the `_promise` member and use it to forward completion or exception to the
     * one of `_futures` member
     * @return A next stage task
     */
    Task MakeNextStageTask() {
        return [this]() mutable {
            StatusCode requestStatus = StatusCode::OK;
            std::exception_ptr localCurrentException = nullptr;
            auto& thisStage = *_itStage;
            auto copyItStage = ++_itStage;

            try {
                auto& stageTask = std::get<Stage_e::task>(thisStage);
                IE_ASSERT(nullptr != stageTask);
                stageTask();
                if (_pipeline.end() != _itStage) {
                    auto nextStage = *_itStage;
                    auto& nextStageExecutor = std::get<Stage_e::executor>(nextStage);
                    IE_ASSERT(nullptr != nextStageExecutor);
                    nextStageExecutor->run(MakeNextStageTask());
                }
            } catch (InferenceEngine::details::InferenceEngineException& ie_ex) {
                requestStatus = ie_ex.hasStatus() ? ie_ex.getStatus() : StatusCode::GENERAL_ERROR;
                localCurrentException = std::make_exception_ptr(ie_ex);
            } catch (...) {
                requestStatus = StatusCode::GENERAL_ERROR;
                localCurrentException = std::current_exception();
            }

            if ((_pipeline.end() == copyItStage) || (nullptr != localCurrentException)) {
                auto lastStageTask = [this, requestStatus, localCurrentException]() mutable {
                    auto promise = std::move(_promise);
                    auto callback = _callback.load();
                    if (setIsRequestBusy(false)) {
                        if (nullptr != callback) {
                            InferenceEngine::CurrentException() = localCurrentException;
                            try {
                                callback(_publicInterface, requestStatus);
                            } catch (...) {
                                localCurrentException = std::current_exception();
                            }
                            InferenceEngine::CurrentException() = nullptr;
                        }
                        if (nullptr == localCurrentException) {
                            promise.set_value();
                        } else {
                            promise.set_exception(localCurrentException);
                        }
                    }
                };

                if (nullptr == _callbackExecutor) {
                    lastStageTask();
                } else {
                    _callbackExecutor->run(std::move(lastStageTask));
                }
            }
        };
    }

    InferRequestInternal::Ptr _syncRequest;
    void* _userData = nullptr;
    AtomicCallback _callback = {nullptr};
    IInferRequest::Ptr _publicInterface;
    Pipeline::iterator _itStage;
    std::promise<void> _promise;
    mutable std::mutex _mutex;
    Futures _futures;
    bool _stop = false;
};
}  // namespace InferenceEngine
