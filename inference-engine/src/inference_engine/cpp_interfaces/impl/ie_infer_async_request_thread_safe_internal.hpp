// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <map>
#include <string>
#include <cpp_interfaces/ie_task.hpp>
#include "cpp_interfaces/interface/ie_iinfer_async_request_internal.hpp"
#include "cpp_interfaces/impl/ie_infer_request_internal.hpp"

namespace InferenceEngine {

/**
 * @brief Wrapper of async request to support thread-safe execution.
 */
class AsyncInferRequestThreadSafeInternal : public IAsyncInferRequestInternal {
    bool _isRequestBusy = false;
    std::mutex _isBusyMutex;

public:
    typedef std::shared_ptr<AsyncInferRequestThreadSafeInternal> Ptr;

    AsyncInferRequestThreadSafeInternal() {
        setIsRequestBusy(false);
    }

protected:
    virtual bool isRequestBusy() const {
        return _isRequestBusy;
    }

    virtual void setIsRequestBusy(bool isBusy) {
        std::unique_lock<std::mutex> lock(_isBusyMutex);
        _isRequestBusy = isBusy;
    }

public:
    void StartAsync() override {
        if (isRequestBusy()) THROW_IE_EXCEPTION << REQUEST_BUSY_str;
        setIsRequestBusy(true);
        try {
            StartAsync_ThreadUnsafe();
        } catch (...) {
            setIsRequestBusy(false);
            std::rethrow_exception(std::current_exception());
        }
    }

    void GetUserData(void **data) override {
        if (isRequestBusy()) THROW_IE_EXCEPTION << REQUEST_BUSY_str;
        GetUserData_ThreadUnsafe(data);
    }

    void SetUserData(void *data) override {
        if (isRequestBusy()) THROW_IE_EXCEPTION << REQUEST_BUSY_str;
        SetUserData_ThreadUnsafe(data);
    }

    void SetCompletionCallback(IInferRequest::CompletionCallback callback) override {
        if (isRequestBusy()) THROW_IE_EXCEPTION << REQUEST_BUSY_str;
        SetCompletionCallback_ThreadUnsafe(callback);
    }

    void Infer() override {
        if (isRequestBusy()) THROW_IE_EXCEPTION << REQUEST_BUSY_str;
        setIsRequestBusy(true);
        try {
            Infer_ThreadUnsafe();
        } catch (...) {
            setIsRequestBusy(false);
            std::rethrow_exception(std::current_exception());
        }
        setIsRequestBusy(false);
    }

    void GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo> &perfMap) const override {
        if (isRequestBusy()) THROW_IE_EXCEPTION << REQUEST_BUSY_str;
        GetPerformanceCounts_ThreadUnsafe(perfMap);
    }

    void SetBlob(const char *name, const Blob::Ptr &data) override {
        if (isRequestBusy()) THROW_IE_EXCEPTION << REQUEST_BUSY_str;
        SetBlob_ThreadUnsafe(name, data);
    }

    void GetBlob(const char *name, Blob::Ptr &data) override {
        if (isRequestBusy()) THROW_IE_EXCEPTION << REQUEST_BUSY_str;
        GetBlob_ThreadUnsafe(name, data);
    }

    void SetBatch(int batch) override {
        if (isRequestBusy()) THROW_IE_EXCEPTION << REQUEST_BUSY_str;
        SetBatch_ThreadUnsafe(batch);
    };

    /**
     * @brief methods with _ThreadUnsafe prefix are to implement in plugins
     * or in default wrapper (e.g. AsyncInferRequestThreadSafeDefault)
     */
    virtual void StartAsync_ThreadUnsafe() = 0;

    virtual void GetUserData_ThreadUnsafe(void **data) = 0;

    virtual void SetUserData_ThreadUnsafe(void *data) = 0;

    virtual void SetCompletionCallback_ThreadUnsafe(IInferRequest::CompletionCallback callback) = 0;

    virtual void Infer_ThreadUnsafe() = 0;

    virtual void
    GetPerformanceCounts_ThreadUnsafe(std::map<std::string, InferenceEngineProfileInfo> &perfMap) const = 0;

    virtual void SetBlob_ThreadUnsafe(const char *name, const Blob::Ptr &data) = 0;

    virtual void GetBlob_ThreadUnsafe(const char *name, Blob::Ptr &data) = 0;

    virtual void SetBatch_ThreadUnsafe(int batch) = 0;
};

}  // namespace InferenceEngine
