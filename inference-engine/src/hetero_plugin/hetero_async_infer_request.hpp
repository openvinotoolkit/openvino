// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include "cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp"
#include "hetero_infer_request.hpp"

namespace HeteroPlugin {

class HeteroAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<HeteroAsyncInferRequest>;
    HeteroAsyncInferRequest(const InferenceEngine::IInferRequestInternal::Ptr& request,
                            const InferenceEngine::ITaskExecutor::Ptr&        taskExecutor,
                            const InferenceEngine::ITaskExecutor::Ptr&        callbackExecutor);
    ~HeteroAsyncInferRequest();
    void StartAsync_ThreadUnsafe() override;
    InferenceEngine::StatusCode Wait(int64_t millis_timeout) override;

private:
    HeteroInferRequest::Ptr                     _heteroInferRequest;
};

}  // namespace HeteroPlugin

