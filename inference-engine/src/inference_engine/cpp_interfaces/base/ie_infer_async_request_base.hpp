// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "cpp_interfaces/exception2status.hpp"
#include "ie_iinfer_request.hpp"
#include "ie_preprocess.hpp"
#include "ie_profiling.hpp"

namespace InferenceEngine {

/**
 * @brief cpp interface for async infer request, to avoid dll boundaries and simplify internal development
 * @tparam T Minimal CPP implementation of IInferRequest (e.g. AsyncInferRequestThreadSafeDefault)
 */
template <class T>
class InferRequestBase : public IInferRequest {
protected:
    std::shared_ptr<T> _impl;

public:
    typedef std::shared_ptr<InferRequestBase<T>> Ptr;

    explicit InferRequestBase(std::shared_ptr<T> impl): _impl(impl) {}

    StatusCode Infer(ResponseDesc* resp) noexcept override {
        IE_PROFILING_AUTO_SCOPE(Infer);
        TO_STATUS(_impl->Infer());
    }

    StatusCode GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo>& perfMap,
                                    ResponseDesc* resp) const noexcept override {
        TO_STATUS(_impl->GetPerformanceCounts(perfMap));
    }

    StatusCode SetBlob(const char* name, const Blob::Ptr& data, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->SetBlob(name, data));
    }

    StatusCode SetBlob(const char* name, const Blob::Ptr& data, const PreProcessInfo& info, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->SetBlob(name, data, info));
    }

    StatusCode GetBlob(const char* name, Blob::Ptr& data, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->GetBlob(name, data));
    }

    StatusCode GetPreProcess(const char* name, const PreProcessInfo** info, ResponseDesc *resp) const noexcept override {
        TO_STATUS(_impl->GetPreProcess(name, info));
    }

    StatusCode StartAsync(ResponseDesc* resp) noexcept override {
        IE_PROFILING_AUTO_SCOPE(StartAsync);
        TO_STATUS(_impl->StartAsync());
    }

    StatusCode Wait(int64_t millis_timeout, ResponseDesc* resp) noexcept override {
        IE_PROFILING_AUTO_SCOPE(Wait);
        NO_EXCEPT_CALL_RETURN_STATUS(_impl->Wait(millis_timeout));
    }

    StatusCode SetCompletionCallback(CompletionCallback callback) noexcept override {
        TO_STATUS_NO_RESP(_impl->SetCompletionCallback(callback));
    }

    StatusCode GetUserData(void** data, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->GetUserData(data));
    }

    StatusCode SetUserData(void* data, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->SetUserData(data));
    }

    void Release() noexcept override {
        delete this;
    }

    StatusCode SetBatch(int batch_size, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->SetBatch(batch_size));
    }

protected:
    ~InferRequestBase() = default;
};

}  // namespace InferenceEngine
