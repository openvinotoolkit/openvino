// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>
#include <cpp_interfaces/ie_itask_executor.hpp>

class MockTaskExecutor : public InferenceEngine::ITaskExecutor {
public:
    typedef std::shared_ptr<MockTaskExecutor> Ptr;

    MOCK_METHOD1(startTask, bool(InferenceEngine::Task::Ptr));
};
