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

struct mockRequestExecutor : public IE::ITaskExecutor {
public:
    using Ptr = std::shared_ptr<mockRequestExecutor>;
    ~mockRequestExecutor() override = default;

    void run(IE::Task task) override {
        task();
    }
};

class mockAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Parent = InferenceEngine::AsyncInferRequestThreadSafeDefault;
    mockAsyncInferRequest(const InferenceEngine::IInferRequestInternal::Ptr &inferRequest,
                      const mockRequestExecutor::Ptr& taskExecutor,
                      const mockRequestExecutor::Ptr& callbackExecutor,
                      bool threw);

    ~mockAsyncInferRequest() override = default;
private:
    bool _throw;
};

mockAsyncInferRequest::mockAsyncInferRequest(const InferenceEngine::IInferRequestInternal::Ptr &inferRequest,
                                     const mockRequestExecutor::Ptr& taskExecutor,
                                     const mockRequestExecutor::Ptr& callbackExecutor,
                                     bool threw)
    : InferenceEngine::AsyncInferRequestThreadSafeDefault(inferRequest, taskExecutor, callbackExecutor), _throw(threw) {
    _pipeline = {};

    _pipeline.push_back({taskExecutor,
                [this] {
                    if (_throw)
                        IE_THROW();
    } });
}
} // namespace MockMultiDevice