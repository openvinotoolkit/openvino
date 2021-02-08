// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "cpp_interfaces/exception2status.hpp"
#include "cpp_interfaces/plugin_itt.hpp"
#include <cpp_interfaces/base/ie_variable_state_base.hpp>
#include <cpp_interfaces/interface/ie_iinfer_async_request_internal.hpp>
#include "ie_iinfer_request.hpp"
#include "ie_preprocess.hpp"

namespace InferenceEngine {

/**
 * @brief Inference request `noexcept` wrapper which accepts IAsyncInferRequestInternal derived instance which can throw exceptions
 * @ingroup ie_dev_api_async_infer_request_api
 */
class InferRequestBase : public IInferRequest {
    std::shared_ptr<IAsyncInferRequestInternal> _impl;

public:
    /**
     * @brief Constructor with actual underlying implementation.
     * @param impl Underlying implementation of type IAsyncInferRequestInternal
     */
    explicit InferRequestBase(std::shared_ptr<IAsyncInferRequestInternal> impl): _impl(impl) {}

    StatusCode Infer(ResponseDesc* resp) noexcept override {
        OV_ITT_SCOPED_TASK(itt::domains::Plugin, "Infer");
        TO_STATUS(_impl->Infer());
    }

    StatusCode Cancel(ResponseDesc* resp) noexcept override {
        OV_ITT_SCOPED_TASK(itt::domains::Plugin, "Cancel");
        NO_EXCEPT_CALL_RETURN_STATUS(_impl->Cancel());
    }

    StatusCode GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo>& perfMap,
                                    ResponseDesc* resp) const noexcept override {
        TO_STATUS(perfMap = _impl->GetPerformanceCounts());
    }

    StatusCode SetBlob(const char* name, const Blob::Ptr& data, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->SetBlob(name, data));
    }

    StatusCode SetBlob(const char* name, const Blob::Ptr& data, const PreProcessInfo& info, ResponseDesc* resp) noexcept override {
        TO_STATUS(_impl->SetBlob(name, data, info));
    }

    StatusCode GetBlob(const char* name, Blob::Ptr& data, ResponseDesc* resp) noexcept override {
        TO_STATUS(data = _impl->GetBlob(name));
    }

    StatusCode GetPreProcess(const char* name, const PreProcessInfo** info, ResponseDesc *resp) const noexcept override {
        TO_STATUS(*info = &(_impl->GetPreProcess(name)));
    }

    StatusCode StartAsync(ResponseDesc* resp) noexcept override {
        OV_ITT_SCOPED_TASK(itt::domains::Plugin, "StartAsync");
        TO_STATUS(_impl->StartAsync());
    }

    StatusCode Wait(int64_t millis_timeout, ResponseDesc* resp) noexcept override {
        OV_ITT_SCOPED_TASK(itt::domains::Plugin, "Wait");
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

    IE_SUPPRESS_DEPRECATED_START
    StatusCode QueryState(IVariableState::Ptr& pState, size_t idx, ResponseDesc* resp) noexcept override {
        try {
            auto v = _impl->QueryState();
            if (idx >= v.size()) {
                return OUT_OF_BOUNDS;
            }
            pState = std::make_shared<VariableStateBase>(v[idx]);
            return OK;
        } catch (const std::exception& ex) {
            return InferenceEngine::DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
        } catch (...) {
            return InferenceEngine::DescriptionBuffer(UNEXPECTED);
        }
    }
    IE_SUPPRESS_DEPRECATED_END

private:
    ~InferRequestBase() = default;
};

}  // namespace InferenceEngine
