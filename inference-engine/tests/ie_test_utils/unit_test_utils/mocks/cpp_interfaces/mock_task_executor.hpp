// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <gmock/gmock.h>
#include <threading/ie_itask_executor.hpp>

class MockTaskExecutor : public InferenceEngine::ITaskExecutor {
public:
    typedef std::shared_ptr<MockTaskExecutor> Ptr;

    MOCK_METHOD1(run, void(InferenceEngine::Task));
};
