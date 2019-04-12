// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <map>
#include <list>
#include <string>
#include <mutex>
#include <exception>
#include <cpp_interfaces/interface/ie_iinfer_async_request_internal.hpp>
#include <cpp_interfaces/ie_task_with_stages.hpp>
#include <cpp_interfaces/ie_task_executor.hpp>
#include <cpp_interfaces/exception2status.hpp>
#include "ie_infer_async_request_thread_safe_internal.hpp"

namespace InferenceEngine {

/**
 * @class CallbackManager for wrapping calling of callback
 */
class CallbackManager {
    std::exception_ptr _requestException = nullptr;
    StatusCode _requestStatus = OK;
    IInferRequest::CompletionCallback _callback = nullptr;
    bool _enabled = false;
    IInferRequest::WeakPtr _publicInterface;
    ITaskExecutor::Ptr _callbackExecutor;

public:
    using Ptr = std::shared_ptr<CallbackManager>;

    explicit CallbackManager(const ITaskExecutor::Ptr &callbackExecutor) : _callbackExecutor(callbackExecutor) {}

    void enableCallback() {
        _enabled = true;
    }

    void disableCallback() {
        _enabled = false;
    }

    bool isCallbackEnabled() { return _enabled && _callback != nullptr; }

    void startTask(Task::Ptr task) { _callbackExecutor->startTask(task); }

    void reset() {
        _requestException = nullptr;
        _requestStatus = OK;
    }

    void runCallback() {
        if (isCallbackEnabled()) {
            auto requestPtr = _publicInterface.lock();
            if (!requestPtr) {
                THROW_IE_EXCEPTION << "Failed to run callback: can't get pointer to request";
            }
            _callback(requestPtr, _requestStatus);
            if (_requestException) std::rethrow_exception(_requestException);
        }
    }

    void set_requestException(const std::exception_ptr &requestException) {
        _requestException = requestException;
    }

    void set_requestStatus(StatusCode requestStatus) {
        _requestStatus = requestStatus;
    }

    void set_callback(IInferRequest::CompletionCallback callback) {
        enableCallback();
        _callback = callback;
    }

    void set_publicInterface(IInferRequest::Ptr publicInterface) {
        _publicInterface = publicInterface;
    }
};

class AsyncInferRequestThreadSafeDefault : public AsyncInferRequestThreadSafeInternal {
public:
    typedef std::shared_ptr<AsyncInferRequestThreadSafeDefault> Ptr;

    explicit AsyncInferRequestThreadSafeDefault(InferRequestInternal::Ptr request,
                                                const ITaskExecutor::Ptr &taskExecutor,
                                                const TaskSynchronizer::Ptr &taskSynchronizer,
                                                const ITaskExecutor::Ptr &callbackExecutor)
            : _syncRequest(request),
              _requestExecutor(taskExecutor),
              _requestSynchronizer(taskSynchronizer),
              _callbackManager(callbackExecutor) {
        _syncTask = std::make_shared<Task>([this]() { _syncRequest->Infer(); });
        _currentTask = _syncTask;
    }

    virtual ~AsyncInferRequestThreadSafeDefault() {
        waitAllAsyncTasks();
    }

    void waitAllAsyncTasks() {
        try {
            while (!_listAsyncTasks.empty()) {
                _listAsyncTasks.remove_if([this](StagedTask::Ptr task) -> bool {
                    auto sts = task->getStatus();
                    return !task->isOnWait() && (Task::Status::TS_DONE == sts || Task::Status::TS_ERROR == sts ||
                                                 Task::Status::TS_INITIAL == sts);
                });
                auto findIter = std::find_if(_listAsyncTasks.begin(), _listAsyncTasks.end(),
                                             [this](StagedTask::Ptr task) { return !task->isOnWait(); });
                if (findIter != _listAsyncTasks.end()) {
                    try {
                        (*findIter)->wait(-1);
                    } catch (...) {}
                }
            }
        } catch (...) {}
    }

    virtual void initNextAsyncTask() {
        IE_PROFILING_AUTO_SCOPE(initNextAsyncTask)
        // Most probably was called from callback (or when callback was started) or it was a sync task before, so new task is required
        if (_currentTask->getStatus() == Task::Status::TS_POSTPONED || _currentTask == _syncTask) {
            auto findIter = std::find_if(_listAsyncTasks.begin(), _listAsyncTasks.end(),
                                         [this](StagedTask::Ptr task) -> bool {
                                             return (!task->isOnWait()) && (task != _currentTask) &&
                                                    (Task::Status::TS_DONE == task->getStatus() ||
                                                     Task::Status::TS_ERROR == task->getStatus());
                                         });
            if (findIter == _listAsyncTasks.end()) {
                _asyncTask = createAsyncRequestTask();
                _listAsyncTasks.push_back(_asyncTask);
            } else {
                _asyncTask = *findIter;
            }
        }
        _asyncTask->resetStages();
        _currentTask = _asyncTask;
    }

    virtual void startAsyncTask() {
        if (!_requestExecutor->startTask(_currentTask)) THROW_IE_EXCEPTION << REQUEST_BUSY_str;
    }

    void StartAsync_ThreadUnsafe() override {
        _syncRequest->checkBlobs();
        _callbackManager.reset();
        initNextAsyncTask();
        startAsyncTask();
    }

    virtual void processAsyncTaskFailure(StagedTask::Ptr asyncTask) {
        setIsRequestBusy(false);
        auto requestException = std::current_exception();
        // callback was set and hasn't been called, it must be called
        if (_callbackManager.isCallbackEnabled() && asyncTask->getStage() >= 1) {
            // jump to the "callback" stage because of happened error
            while (asyncTask->getStage() != 1) asyncTask->stageDone();
            _callbackManager.set_requestStatus(GENERAL_ERROR);
            _callbackManager.set_requestException(requestException);
            _callbackManager.startTask(asyncTask);
        } else {
            std::rethrow_exception(requestException);
        }
    }

    virtual StagedTask::Ptr createAsyncRequestTask() {
        return std::make_shared<StagedTask>([this]() {
            auto asyncTaskCopy = _asyncTask;
            try {
                switch (asyncTaskCopy->getStage()) {
                    case 2: {
                        _syncRequest->Infer();
                        asyncTaskCopy->stageDone();
                        if (_callbackManager.isCallbackEnabled()) {
                            _callbackManager.startTask(asyncTaskCopy);
                        } else {
                            asyncTaskCopy->stageDone();
                        }
                    }
                        break;
                    case 1: {
                        setIsRequestBusy(false);
                        asyncTaskCopy->stageDone();
                        _callbackManager.runCallback();
                    }
                        break;
                    default:
                        break;
                }
            } catch (...) {
                processAsyncTaskFailure(asyncTaskCopy);
            }
        }, 2);
    }

    StatusCode Wait(int64_t millis_timeout) override {
        auto taskCopy = _currentTask;
        if (millis_timeout < IInferRequest::WaitMode::RESULT_READY) {
            THROW_IE_EXCEPTION << PARAMETER_MISMATCH_str + "Timeout can't be less "
                               << IInferRequest::WaitMode::RESULT_READY
                               << " for InferRequest::Wait\n";
        }
        Task::Status status;
        if (millis_timeout == IInferRequest::WaitMode::STATUS_ONLY) {
            status = taskCopy->getStatus();
        } else {
            status = taskCopy->wait(millis_timeout);
            setIsRequestBusy(false);
        }

        taskCopy->checkException();
        return Task::TaskStatus2StatusCode(status);
    }

    void Infer_ThreadUnsafe() override {
        _currentTask = _syncTask;
        auto status = _currentTask->runWithSynchronizer(_requestSynchronizer);
        if (status == Task::Status::TS_BUSY)
            THROW_IE_EXCEPTION << "Internal error: AsyncInferRequestThreadSafeDefault failed to start sync task";
        _currentTask->checkException();
    }

    void GetPerformanceCounts_ThreadUnsafe(std::map<std::string, InferenceEngineProfileInfo> &perfMap) const override {
        _syncRequest->GetPerformanceCounts(perfMap);
    }

    void SetBlob_ThreadUnsafe(const char *name, const Blob::Ptr &data) override {
        _syncRequest->SetBlob(name, data);
    }

    void GetBlob_ThreadUnsafe(const char *name, Blob::Ptr &data) override {
        _syncRequest->GetBlob(name, data);
    }

    void SetCompletionCallback_ThreadUnsafe(InferenceEngine::IInferRequest::CompletionCallback callback) override {
        _callbackManager.set_callback(callback);
    }

    void GetUserData_ThreadUnsafe(void **data) override {
        if (data == nullptr) THROW_IE_EXCEPTION << NOT_ALLOCATED_str;
        *data = _userData;
    }

    void SetUserData_ThreadUnsafe(void *data) override {
        _userData = data;
    }

    void SetPointerToPublicInterface(InferenceEngine::IInferRequest::Ptr ptr) {
        _callbackManager.set_publicInterface(ptr);
    }

    void SetBatch_ThreadUnsafe(int batch) override {
        _syncRequest->SetBatch(batch);
    }

protected:
    ITaskExecutor::Ptr _requestExecutor;
    TaskSynchronizer::Ptr _requestSynchronizer;
    InferRequestInternal::Ptr _syncRequest;
    Task::Ptr _syncTask;
    StagedTask::Ptr _asyncTask;
    Task::Ptr _currentTask;
    std::list<StagedTask::Ptr> _listAsyncTasks;
    void *_userData;
    CallbackManager _callbackManager;
};

}  // namespace InferenceEngine
