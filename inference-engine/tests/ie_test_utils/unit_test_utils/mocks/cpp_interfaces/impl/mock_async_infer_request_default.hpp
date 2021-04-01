// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <ie_iinfer_request.hpp>

#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>

#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_infer_request_internal.hpp"

using namespace InferenceEngine;

class MockAsyncInferRequestDefault : public AsyncInferRequestThreadSafeDefault {
public:
    MockAsyncInferRequestDefault(InferRequestInternal::Ptr request,
                                 const ITaskExecutor::Ptr &taskExecutor,
                                 const ITaskExecutor::Ptr &callbackExecutor)
            : AsyncInferRequestThreadSafeDefault(request, taskExecutor, callbackExecutor) {}

    MOCK_METHOD0(CheckBlob, void());
};
