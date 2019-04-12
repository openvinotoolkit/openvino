// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_plugin.hpp"
#include <cpp_interfaces/ie_task_synchronizer.hpp>
#include <gmock/gmock.h>
#include <string>
#include <vector>
#include <memory>

class MockTaskSynchronizer : public InferenceEngine::TaskSynchronizer {
public:
    typedef std::shared_ptr<MockTaskSynchronizer> Ptr;
    MOCK_METHOD0(lock, void());
    MOCK_METHOD0(unlock, void());
};

class MockTaskSynchronizerPrivate : public InferenceEngine::TaskSynchronizer {
public:
    typedef std::shared_ptr<MockTaskSynchronizerPrivate> Ptr;
    MOCK_METHOD0(_addTaskToQueue, unsigned int());
    MOCK_METHOD1(_waitInQueue, void(unsigned int));
};
