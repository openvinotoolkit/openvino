// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>
#include "infer_request.h"

namespace ov {
namespace intel_cpu {

class AsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    AsyncInferRequest(const InferenceEngine::IInferRequestInternal::Ptr &inferRequest,
                      const InferenceEngine::ITaskExecutor::Ptr &taskExecutor,
                      const InferenceEngine::ITaskExecutor::Ptr &callbackExecutor);
    ~AsyncInferRequest();
};

}   // namespace intel_cpu
}   // namespace ov

