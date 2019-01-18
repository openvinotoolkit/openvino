// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file for IInferRequest interface
 * @file ie_iinfer_request.hpp
 */

#pragma once

#include <unordered_set>
#include <utility>
#include <string>
#include <map>
#include <memory>

#include "cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp"
#include "hetero_infer_request.h"

namespace HeteroPlugin {

class HeteroAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    typedef std::shared_ptr<HeteroAsyncInferRequest> Ptr;

    HeteroAsyncInferRequest(HeteroInferRequest::Ptr request,
                            const InferenceEngine::ITaskExecutor::Ptr &taskExecutor,
                            const InferenceEngine::TaskSynchronizer::Ptr &taskSynchronizer,
                            const InferenceEngine::ITaskExecutor::Ptr &callbackExecutor);

    void StartAsync() override;

    InferenceEngine::StatusCode Wait(int64_t millis_timeout) override;

    void SetCompletionCallback(InferenceEngine::IInferRequest::CompletionCallback callback) override;

private:
    HeteroInferRequest::Ptr _heteroInferRequest;
};

}  // namespace HeteroPlugin

