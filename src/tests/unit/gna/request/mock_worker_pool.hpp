// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gmock/gmock.h"
#include "request/worker_pool.hpp"

namespace GNAPluginNS {
namespace request {

class MockWorkerPool : public WorkerPool {
public:
    MOCK_METHOD(void, addModelWorker, (std::shared_ptr<Worker>), (override));
    MOCK_METHOD(size_t, size, (), (const, override));
    MOCK_METHOD(size_t, empty, (), (const, override));
    MOCK_METHOD(Worker&, worker, (uint32_t), (override));
    MOCK_METHOD(const Worker&, worker, (uint32_t), (const, override));
    MOCK_METHOD(Worker&, firstWorker, (uint32_t), (override));
    MOCK_METHOD(const Worker&, firstWorker, (uint32_t), (const, override));
    MOCK_METHOD(Worker&, lastWorker, (uint32_t), (override));
    MOCK_METHOD(const Worker&, lastWorker, (uint32_t), (const, override));
    MOCK_METHOD(std::shared_ptr<Worker>, findFreeModelWorker, (), (override));
};

}  // namespace request
}  // namespace GNAPluginNS
