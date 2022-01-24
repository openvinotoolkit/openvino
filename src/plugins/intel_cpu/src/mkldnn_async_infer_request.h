// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>
#include "mkldnn_infer_request.h"

namespace MKLDNNPlugin {

class MKLDNNAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    MKLDNNAsyncInferRequest(const InferenceEngine::IInferRequestInternal::Ptr &inferRequest,
                            const InferenceEngine::ITaskExecutor::Ptr &taskExecutor,
                            const InferenceEngine::ITaskExecutor::Ptr &callbackExecutor);
    ~MKLDNNAsyncInferRequest();
};

}  // namespace MKLDNNPlugin
