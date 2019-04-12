//
// Copyright 2017-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
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

