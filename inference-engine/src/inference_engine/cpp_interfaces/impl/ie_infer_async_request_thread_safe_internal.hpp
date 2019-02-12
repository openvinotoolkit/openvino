// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <string>

#include "cpp_interfaces/impl/ie_infer_request_internal.hpp"
#include "cpp_interfaces/interface/ie_iinfer_async_request_internal.hpp"

namespace InferenceEngine {

/**
 * @brief Wrapper of async request to support thread-safe execution.
 */
class AsyncInferRequestThreadSafeInternal : public IAsyncInferRequestInternal {
    std::atomic_bool _isRequestBusy = {false};

public:
    typedef std::shared_ptr<AsyncInferRequestThreadSafeInternal> Ptr;

    AsyncInferRequestThreadSafeInternal() {
        setIsRequestBusy(false);
    }

protected:
    virtual bool isRequestBusy() const {
        return _isRequestBusy;
    }

    virtual bool setIsRequestBusy(bool isBusy) {
        return _isRequestBusy.exchange(isBusy);
    }

    [[noreturn]] static void ThrowBusy() {
        THROW_IE_EXCEPTION << InferenceEngine::details::as_status << StatusCode::REQUEST_BUSY << REQUEST_BUSY_str;
    }

    void CheckBusy() const {
        if (isRequestBusy()) ThrowBusy();
    }

public:
    void StartAsync() override {
        if (setIsRequestBusy(true)) ThrowBusy();
        try {
            StartAsync_ThreadUnsafe();
        } catch (...) {
            setIsRequestBusy(false);
            throw;
        }
    }

    void GetUserData(void** data) override {
        CheckBusy();
        GetUserData_ThreadUnsafe(data);
    }

    void SetUserData(void* data) override {
        CheckBusy();
        SetUserData_ThreadUnsafe(data);
    }

    void SetCompletionCallback(IInferRequest::CompletionCallback callback) override {
        CheckBusy();
        SetCompletionCallback_ThreadUnsafe(callback);
    }

    void Infer() override {
        if (setIsRequestBusy(true)) ThrowBusy();
        try {
            Infer_ThreadUnsafe();
        } catch (...) {
            setIsRequestBusy(false);
            throw;
        }
        setIsRequestBusy(false);
    }

    void GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo>& perfMap) const override {
        CheckBusy();
        GetPerformanceCounts_ThreadUnsafe(perfMap);
    }

    void SetBlob(const char* name, const Blob::Ptr& data) override {
	CheckBusy();
        SetBlob_ThreadUnsafe(name, data);
    }

    void SetBlob(const char* name, const Blob::Ptr& data, const PreProcessInfo& info) override {
        CheckBusy();
        SetBlob_ThreadUnsafe(name, data, info);
    }

    void GetBlob(const char* name, Blob::Ptr& data) override {
        CheckBusy();
        GetBlob_ThreadUnsafe(name, data);
    }

    void GetPreProcess(const char* name, const PreProcessInfo** info) const override {
        GetPreProcess_ThreadUnsafe(name, info);
    }

    void SetBatch(int batch) override {
        CheckBusy();
        SetBatch_ThreadUnsafe(batch);
    };

    /**
     * @brief methods with _ThreadUnsafe prefix are to implement in plugins
     * or in default wrapper (e.g. AsyncInferRequestThreadSafeDefault)
     */
    virtual void StartAsync_ThreadUnsafe() = 0;

    virtual void GetUserData_ThreadUnsafe(void** data) = 0;

    virtual void SetUserData_ThreadUnsafe(void* data) = 0;

    virtual void SetCompletionCallback_ThreadUnsafe(IInferRequest::CompletionCallback callback) = 0;

    virtual void Infer_ThreadUnsafe() = 0;

    virtual void GetPerformanceCounts_ThreadUnsafe(
        std::map<std::string, InferenceEngineProfileInfo>& perfMap) const = 0;

    virtual void SetBlob_ThreadUnsafe(const char* name, const Blob::Ptr& data) = 0;

    virtual void SetBlob_ThreadUnsafe(const char* name, const Blob::Ptr& data, const PreProcessInfo& info) = 0;

    virtual void GetBlob_ThreadUnsafe(const char* name, Blob::Ptr& data) = 0;

    virtual void GetPreProcess_ThreadUnsafe(const char* name, const PreProcessInfo** info) const = 0;

    virtual void SetBatch_ThreadUnsafe(int batch) = 0;
};

}  // namespace InferenceEngine
