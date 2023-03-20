// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock.h>
#include "ie_icore.hpp"
#include "plugin.hpp"
#include <iostream>

using namespace MockMultiDevicePlugin;
namespace MockMultiDevice {

class mockAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Parent = InferenceEngine::AsyncInferRequestThreadSafeDefault;
    mockAsyncInferRequest(const InferenceEngine::IInferRequestInternal::Ptr &inferRequest,
                      const ImmediateExecutor::Ptr& taskExecutor,
                      const ImmediateExecutor::Ptr& callbackExecutor,
                      bool ifThrow);

    ~mockAsyncInferRequest() override = default;
private:
    bool _throw;
};

mockAsyncInferRequest::mockAsyncInferRequest(const InferenceEngine::IInferRequestInternal::Ptr &inferRequest,
                                     const ImmediateExecutor::Ptr& taskExecutor,
                                     const ImmediateExecutor::Ptr& callbackExecutor,
                                     bool ifThrow)
    : InferenceEngine::AsyncInferRequestThreadSafeDefault(inferRequest, taskExecutor, callbackExecutor), _throw(ifThrow) {
    _pipeline = {};

    _pipeline.push_back({taskExecutor,
                [this] {
                    if (_throw)
                        IE_THROW();
    } });
}
} // namespace MockMultiDevice