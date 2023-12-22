// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <string>
#include <vector>

#include "cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp"

using namespace ov::threading;
using namespace InferenceEngine;

class MockAsyncInferRequestDefault : public AsyncInferRequestThreadSafeDefault {
public:
    MockAsyncInferRequestDefault(IInferRequestInternal::Ptr request,
                                 const ITaskExecutor::Ptr& taskExecutor,
                                 const ITaskExecutor::Ptr& callbackExecutor)
        : AsyncInferRequestThreadSafeDefault(request, taskExecutor, callbackExecutor) {}

    MOCK_METHOD0(CheckBlob, void());
};
