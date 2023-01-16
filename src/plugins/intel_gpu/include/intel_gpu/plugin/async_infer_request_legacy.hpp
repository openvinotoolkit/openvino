// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>
#include "intel_gpu/plugin/infer_request_legacy.hpp"

namespace ov {
namespace intel_gpu {

class AsyncInferRequestLegacy : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Parent = InferenceEngine::AsyncInferRequestThreadSafeDefault;
    AsyncInferRequestLegacy(const InferRequestLegacy::Ptr &inferRequest,
                      const InferenceEngine::ITaskExecutor::Ptr& taskExecutor,
                      const InferenceEngine::ITaskExecutor::Ptr& waitExecutor,
                      const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor);

    ~AsyncInferRequestLegacy();

    void StartAsync_ThreadUnsafe() override;

private:
    InferRequestLegacy::Ptr _inferRequest;
    InferenceEngine::ITaskExecutor::Ptr _waitExecutor;
};

}  // namespace intel_gpu
}  // namespace ov
