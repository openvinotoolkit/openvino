// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <memory>

#include "openvino/runtime/threading/itask_executor.hpp"

class MockTaskExecutor : public ov::threading::ITaskExecutor {
public:
    typedef std::shared_ptr<MockTaskExecutor> Ptr;

    MOCK_METHOD1(run, void(ov::threading::Task));
};
