// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp"
#include "infer_request.hpp"

namespace HeteroPlugin {

class HeteroAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<HeteroAsyncInferRequest>;
    HeteroAsyncInferRequest(const InferenceEngine::IInferRequestInternal::Ptr& request,
                            const InferenceEngine::ITaskExecutor::Ptr& taskExecutor,
                            const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor);
    ~HeteroAsyncInferRequest();
    InferenceEngine::StatusCode Wait(int64_t millis_timeout) override;

private:
    HeteroInferRequest::Ptr _heteroInferRequest;
};

}  // namespace HeteroPlugin
