// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>
#include "cldnn_infer_request.h"

namespace CLDNNPlugin {

class CLDNNAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Parent = InferenceEngine::AsyncInferRequestThreadSafeDefault;
    CLDNNAsyncInferRequest(const CLDNNInferRequest::Ptr &inferRequest,
                           const InferenceEngine::ITaskExecutor::Ptr& taskExecutor,
                           const InferenceEngine::ITaskExecutor::Ptr& waitExecutor,
                           const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor);

    ~CLDNNAsyncInferRequest();

    void Infer_ThreadUnsafe() override;
    void StartAsync_ThreadUnsafe() override;

private:
    CLDNNInferRequest::Ptr _inferRequest;
    InferenceEngine::ITaskExecutor::Ptr _waitExecutor;
};

}  // namespace CLDNNPlugin
