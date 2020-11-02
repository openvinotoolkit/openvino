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
 * @brief Wrapper of asynchronous inference request to support thread-safe execution.
 * @ingroup ie_dev_api_async_infer_request_api
 */
class AsyncInferRequestThreadSafeInternal : public IAsyncInferRequestInternal {
    std::atomic_bool _isRequestBusy = {false};

protected:
    /**
     * @brief      Determines if request busy.
     * @return     `True` if request busy, `false` otherwise.
     */
    virtual bool isRequestBusy() const {
        return _isRequestBusy;
    }

    /**
     * @brief      Sets the is request busy.
     * @param[in]  isBusy  Indicates if busy
     * @return     `True` is case of success, `false` otherwise.
     */
    virtual bool setIsRequestBusy(bool isBusy) {
        return _isRequestBusy.exchange(isBusy);
    }

    /**
     * @brief Throws an exception that an inference request is busy.
     */
    [[noreturn]] static void ThrowBusy() {
        THROW_IE_EXCEPTION << InferenceEngine::details::as_status << StatusCode::REQUEST_BUSY << REQUEST_BUSY_str;
    }

    /**
     * @brief Checks whether an inference request is busy and calls ThrowBusy if `true`
     */
    void CheckBusy() const {
        if (isRequestBusy()) ThrowBusy();
    }

public:
    /**
     * @brief A shared pointer to a AsyncInferRequestThreadSafeInternal implementation
     */
    typedef std::shared_ptr<AsyncInferRequestThreadSafeInternal> Ptr;

    /**
     * @brief Constructs a new instance.
     */
    AsyncInferRequestThreadSafeInternal() {
        setIsRequestBusy(false);
    }

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

protected:
    /**
     * @brief Starts an asynchronous pipeline thread unsafe.
     * @note Used by AsyncInferRequestThreadSafeInternal::StartAsync which ensures thread-safety
     *       and calls this method after.
     */
    virtual void StartAsync_ThreadUnsafe() = 0;

    /**
     * @brief Gets the user data thread unsafe.
     * @note Used by AsyncInferRequestThreadSafeInternal::GetUserData which ensures thread-safety
     *       and calls this method after.
     * @param data The user data
     */
    virtual void GetUserData_ThreadUnsafe(void** data) = 0;

    /**
     * @brief Sets the user data thread unsafe.
     * @note Used by AsyncInferRequestThreadSafeInternal::SetUserData which ensures thread-safety
     *       and calls this method after.
     * @param data  The user data
     */
    virtual void SetUserData_ThreadUnsafe(void* data) = 0;

    /**
     * @brief Sets the completion callback thread unsafe.
     * @note Used by AsyncInferRequestThreadSafeInternal::SetCompletionCallback which ensures thread-safety
     *       and calls this method after.
     * @param[in]  callback The callback to set
     */
    virtual void SetCompletionCallback_ThreadUnsafe(IInferRequest::CompletionCallback callback) = 0;

    /**
     * @brief Performs inference of pipeline in syncronous mode
     * @note Used by AsyncInferRequestThreadSafeInternal::Infer which ensures thread-safety
     *       and calls this method after.
     */
    virtual void Infer_ThreadUnsafe() = 0;

    /**
     * @brief Gets the performance counts thread unsafe.
     * @note Used by AsyncInferRequestThreadSafeInternal::GetPerformanceCounts which ensures thread-safety
     *       and calls this method after.
     * @param perfMap  The performance map
     */
    virtual void GetPerformanceCounts_ThreadUnsafe(
        std::map<std::string, InferenceEngineProfileInfo>& perfMap) const = 0;

    /**
     * @brief Sets the blob thread unsafe.
     * @note Used by AsyncInferRequestThreadSafeInternal::SetBlob which ensures thread-safety
     *       and calls this method after.
     * @param[in]  name  The name of input / output data to set a blob to
     * @param[in]  data  The blob to set
     */
    virtual void SetBlob_ThreadUnsafe(const char* name, const Blob::Ptr& data) = 0;

    /**
     * @brief Sets the blob with preprocessing information thread unsafe.
     * @note Used by AsyncInferRequestThreadSafeInternal::SetBlob which ensures thread-safety
     *       and calls this method after.
     * @param[in] name  The name of input / output data to set a blob to
     * @param[in] data  The blob to set
     * @param[in] info  The preprocessing information
     */
    virtual void SetBlob_ThreadUnsafe(const char* name, const Blob::Ptr& data, const PreProcessInfo& info) = 0;

    /**
     * @brief Gets the input or output blob thread unsafe.
     * @note Used by AsyncInferRequestThreadSafeInternal::GetBlob which ensures thread-safety
     *       and calls this method after.
     * @param[in] name  The name of input / output data to get a blob for
     * @param     data  The data
     */
    virtual void GetBlob_ThreadUnsafe(const char* name, Blob::Ptr& data) = 0;

    /**
     * @brief Gets the preprocessing information thread unsafe.
     * @note Used by AsyncInferRequestThreadSafeInternal::GetPreProcess which ensures thread-safety
     *       and calls this method after.
     * @param[in] name  The name of input / output data to get a processing information for
     * @param      info  The preprocessing information
     */
    virtual void GetPreProcess_ThreadUnsafe(const char* name, const PreProcessInfo** info) const = 0;

    /**
     * @brief Sets the dynamic batch thread unsafe.
     * @note Used by AsyncInferRequestThreadSafeInternal::SetBatch which ensures thread-safety
     *       and calls this method after.
     * @param[in]  batch  The dynamic batch value
     */
    virtual void SetBatch_ThreadUnsafe(int batch) = 0;
};

}  // namespace InferenceEngine
