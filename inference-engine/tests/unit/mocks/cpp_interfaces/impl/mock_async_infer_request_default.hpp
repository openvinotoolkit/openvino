// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "cpp_interfaces/impl/mock_infer_request_internal.hpp"

#include <ie_iinfer_request.hpp>
#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>

using namespace InferenceEngine;

class MockAsyncInferRequestDefault : public AsyncInferRequestThreadSafeDefault {
public:
    MockAsyncInferRequestDefault(InferRequestInternal::Ptr request,
                                 const ITaskExecutor::Ptr &taskExecutor,
                                 const TaskSynchronizer::Ptr &taskSynchronizer,
                                 const ITaskExecutor::Ptr &callbackExecutor)
            : AsyncInferRequestThreadSafeDefault(request, taskExecutor, taskSynchronizer, callbackExecutor) {}

    MOCK_METHOD0(startAsyncTask, void());

    MOCK_METHOD0(initNextAsyncTask, void());
};
