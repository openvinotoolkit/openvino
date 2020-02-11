// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides wrapper classes for infer requests and callbacks.
 *
 * @file ie_infer_request.hpp
 */
#pragma once

#include <map>
#include <memory>
#include <string>

#include "details/ie_exception_conversion.hpp"
#include "ie_iinfer_request.hpp"
#include "ie_plugin_ptr.hpp"

namespace InferenceEngine {

namespace details {

class ICompletionCallbackWrapper {
public:
    virtual ~ICompletionCallbackWrapper() = default;

    virtual void call(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code) const noexcept = 0;
};

template <class T>
class CompletionCallbackWrapper : public ICompletionCallbackWrapper {
    T lambda;

public:
    explicit CompletionCallbackWrapper(const T& lambda): lambda(lambda) {}

    void call(InferenceEngine::IInferRequest::Ptr /*request*/, InferenceEngine::StatusCode /*code*/) const
        noexcept override {
        lambda();
    }
};

template <>
class CompletionCallbackWrapper<IInferRequest::CompletionCallback> : public ICompletionCallbackWrapper {
    IInferRequest::CompletionCallback callBack;

public:
    explicit CompletionCallbackWrapper(const IInferRequest::CompletionCallback& callBack): callBack(callBack) {}

    void call(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code) const noexcept override {
        callBack(request, code);
    }
};

}  // namespace details

/**
 * @copybrief IInferRequest
 *
 * Wraps IInferRequest
 * It can throw exceptions safely for the application, where it is properly handled.
 */
class InferRequest {
    IInferRequest::Ptr actual;
    InferenceEnginePluginPtr plg;
    std::shared_ptr<details::ICompletionCallbackWrapper> callback;

    static void callWrapper(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code) {
        details::ICompletionCallbackWrapper* pWrapper = nullptr;
        ResponseDesc dsc;
        request->GetUserData(reinterpret_cast<void**>(&pWrapper), &dsc);
        pWrapper->call(request, code);
    }

public:
    /**
     * @brief Default constructor
     */
    InferRequest() = default;

    /**
     * @brief Destructor
     */
    ~InferRequest() {
        actual = nullptr;
    }

    /**
     * @brief Sets input/output data to infer
     *
     * @note: Memory allocation does not happen
     * @param name Name of input or output blob.
     * @param data Reference to input or output blob. The type of a blob must match the network input precision and
     * size.
     */
    void SetBlob(const std::string& name, const Blob::Ptr& data) {
        CALL_STATUS_FNC(SetBlob, name.c_str(), data);
    }

    /**
     * @copybrief  IInferRequest::GetBlob
     *
     * Wraps IInferRequest::GetBlob
     */
    Blob::Ptr GetBlob(const std::string& name) {
        Blob::Ptr data;
        CALL_STATUS_FNC(GetBlob, name.c_str(), data);
        std::string error = "Internal error: blob with name `" + name + "` is not allocated!";
        auto blobPtr = data.get();
        if (blobPtr == nullptr) THROW_IE_EXCEPTION << error;
        if (blobPtr->buffer() == nullptr) THROW_IE_EXCEPTION << error;
        return data;
    }

    /**
     * @brief Sets pre-process for input data
     * @note: Will return an error in case if data blob is output
     * @param name Name of input blob.
     * @param data - a reference to input. The type of Blob must correspond to the network input precision and size.
     * @param info Preprocess info for blob.
     */
    void SetBlob(const std::string &name, const Blob::Ptr &data, const PreProcessInfo& info) {
        CALL_STATUS_FNC(SetBlob, name.c_str(), data, info);
    }

    /**
     * @brief Gets pre-process for input data
     * @param name Name of input blob.
     * @return pointer to pre-process info of blob with name
     */
    const PreProcessInfo& GetPreProcess(const std::string& name) const {
        const PreProcessInfo* info = nullptr;
        CALL_STATUS_FNC(GetPreProcess, name.c_str(), &info);
        return *info;
    }

    /**
     * @copybrief IInferRequest::Infer
     *
     * Wraps IInferRequest::Infer
     */
    void Infer() {
        CALL_STATUS_FNC_NO_ARGS(Infer);
    }

    /**
     * @copybrief IInferRequest::GetPerformanceCounts
     *
     * Wraps IInferRequest::GetPerformanceCounts
     */
    std::map<std::string, InferenceEngineProfileInfo> GetPerformanceCounts() const {
        std::map<std::string, InferenceEngineProfileInfo> perfMap;
        CALL_STATUS_FNC(GetPerformanceCounts, perfMap);
        return perfMap;
    }

    /**
     * @brief Sets input data to infer
     *
     * @note: Memory allocation doesn't happen
     * @param inputs - a reference to a map of input blobs accessed by input names.
     *        The type of Blob must correspond to the network input precision and size.
     */
    void SetInput(const BlobMap& inputs) {
        for (auto&& input : inputs) {
            CALL_STATUS_FNC(SetBlob, input.first.c_str(), input.second);
        }
    }

    /**
     * @brief Sets data that will contain result of the inference
     *
     * @note: Memory allocation doesn't happen
     * @param results - a reference to a map of result blobs accessed by output names.
     *        The type of Blob must correspond to the network output precision and size.
     */
    void SetOutput(const BlobMap& results) {
        for (auto&& result : results) {
            CALL_STATUS_FNC(SetBlob, result.first.c_str(), result.second);
        }
    }

    /**
     * @brief Sets new batch size when dynamic batching is enabled in executable network that created this request.
     *
     * @param batch new batch size to be used by all the following inference calls for this request.
     */
    void SetBatch(const int batch) {
        CALL_STATUS_FNC(SetBatch, batch);
    }

    /**
     * constructs InferRequest from the initialized shared_pointer
     * @param request Initialized shared pointer
     * @param plg Plugin to use
     */
    explicit InferRequest(IInferRequest::Ptr request, InferenceEnginePluginPtr plg = {}): actual(request), plg(plg) {}

    /**
     * @brief Start inference of specified input(s) in asynchronous mode
     *
     * @note: It returns immediately. Inference starts also immediately.
     */
    void StartAsync() {
        CALL_STATUS_FNC_NO_ARGS(StartAsync);
    }

    /**
     * @copybrief IInferRequest::Wait
     *
     * Wraps IInferRequest::Wait
     */
    StatusCode Wait(int64_t millis_timeout) {
        ResponseDesc resp;
        auto res = actual->Wait(millis_timeout, &resp);
        if (res != OK && res != RESULT_NOT_READY && res != INFER_NOT_STARTED) {
            InferenceEngine::details::extract_exception(res, resp.msg);
        }
        return res;
    }

    /**
     * @copybrief IInferRequest::SetCompletionCallback
     *
     * Wraps IInferRequest::SetCompletionCallback
     *
     * @param callbackToSet Lambda callback object which will be called on processing finish.
     */
    template <class T>
    void SetCompletionCallback(const T& callbackToSet) {
        callback.reset(new details::CompletionCallbackWrapper<T>(callbackToSet));
        CALL_STATUS_FNC(SetUserData, callback.get());
        actual->SetCompletionCallback(callWrapper);
    }

    /**
     * @brief  IInferRequest pointer to be used directly in CreateInferRequest functions
     */
    operator IInferRequest::Ptr&() {
        return actual;
    }

    /**
     * @brief Checks if current InferRequest object is not initialized
     *
     * @return true if current InferRequest object is not initialized, false - otherwise
     */
    bool operator!() const noexcept {
        return !actual;
    }

    /**
     * @brief Checks if current InferRequest object is initialized
     *
     * @return true if current InferRequest object is initialized, false - otherwise
     */
    explicit operator bool() const noexcept {
        return !!actual;
    }

    /**
     * @brief A smart pointer to the InferRequest object
     */
    using Ptr = std::shared_ptr<InferRequest>;
};

namespace details {

template <>
class CompletionCallbackWrapper<std::function<void(InferRequest, StatusCode)>> : public ICompletionCallbackWrapper {
    std::function<void(InferRequest, StatusCode)> lambda;

public:
    explicit CompletionCallbackWrapper(const std::function<void(InferRequest, InferenceEngine::StatusCode)>& lambda)
        : lambda(lambda) {}

    void call(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code) const noexcept override {
        lambda(InferRequest(request), code);
    }
};

}  // namespace details
}  // namespace InferenceEngine
