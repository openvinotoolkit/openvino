// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "mock_worker.hpp"
#include "request/worker_pool_impl.hpp"

using namespace GNAPluginNS;
using namespace request;
using namespace testing;

class GNA_Request_WorkerPoolImplTest : public ::testing::Test {
public:
    static const constexpr uint32_t kExpectedIndex = 0;
};
const constexpr uint32_t GNA_Request_WorkerPoolImplTest::kExpectedIndex;

TEST_F(GNA_Request_WorkerPoolImplTest, initDeinit) {
    ASSERT_NO_THROW(GNAPluginNS::request::WorkerPoolImpl());
}

TEST_F(GNA_Request_WorkerPoolImplTest, addModelWorker_nullptr) {
    std::shared_ptr<WorkerPoolImpl> workerPool;
    ASSERT_NO_THROW(workerPool = std::make_shared<WorkerPoolImpl>());
    ASSERT_THROW(workerPool->addModelWorker(nullptr), std::exception);
}

TEST_F(GNA_Request_WorkerPoolImplTest, addModelWorker_proper) {
    WorkerPoolImpl workerPool;
    auto workerMock = std::make_shared<MockWorker>();

    EXPECT_CALL(*workerMock.get(), setRepresentingIndex(kExpectedIndex)).Times(1);
    ASSERT_NO_THROW(workerPool.addModelWorker(workerMock));
}

TEST_F(GNA_Request_WorkerPoolImplTest, size) {
    WorkerPoolImpl workerPool;
    ASSERT_EQ(workerPool.size(), 0);
    auto workerMock = std::make_shared<MockWorker>();
    EXPECT_CALL(*workerMock.get(), setRepresentingIndex(kExpectedIndex)).Times(1);
    ASSERT_NO_THROW(workerPool.addModelWorker(workerMock));
    ASSERT_EQ(workerPool.size(), 1);
}

TEST_F(GNA_Request_WorkerPoolImplTest, worker) {
    WorkerPoolImpl workerPool;
    ASSERT_THROW(workerPool.worker(0), std::exception);
    auto workerMock = std::make_shared<MockWorker>();
    EXPECT_CALL(*workerMock.get(), setRepresentingIndex(kExpectedIndex)).Times(1);
    ASSERT_NO_THROW(workerPool.addModelWorker(workerMock));
    ASSERT_NO_THROW(workerPool.worker(0));
}

TEST_F(GNA_Request_WorkerPoolImplTest, worker_const) {
    WorkerPoolImpl workerPool;
    const WorkerPoolImpl& constWorkerPool = workerPool;
    ASSERT_THROW(constWorkerPool.worker(0), std::exception);
    auto workerMock = std::make_shared<MockWorker>();
    EXPECT_CALL(*workerMock.get(), setRepresentingIndex(kExpectedIndex)).Times(1);
    ASSERT_NO_THROW(workerPool.addModelWorker(workerMock));
    ASSERT_NO_THROW(constWorkerPool.worker(0));
}

TEST_F(GNA_Request_WorkerPoolImplTest, firstWorker) {
    WorkerPoolImpl workerPool;
    ASSERT_THROW(workerPool.firstWorker(), std::exception);
    auto workerMock = std::make_shared<MockWorker>();
    EXPECT_CALL(*workerMock.get(), setRepresentingIndex(kExpectedIndex)).Times(1);
    ASSERT_NO_THROW(workerPool.addModelWorker(workerMock));
    ASSERT_NO_THROW(workerPool.firstWorker());
}

TEST_F(GNA_Request_WorkerPoolImplTest, firstWorker_const) {
    WorkerPoolImpl workerPool;
    const WorkerPoolImpl& constWorkerPool = workerPool;
    ASSERT_THROW(constWorkerPool.firstWorker(), std::exception);
    auto workerMock = std::make_shared<MockWorker>();
    EXPECT_CALL(*workerMock.get(), setRepresentingIndex(kExpectedIndex)).Times(1);
    ASSERT_NO_THROW(workerPool.addModelWorker(workerMock));
    ASSERT_NO_THROW(constWorkerPool.firstWorker());
}

TEST_F(GNA_Request_WorkerPoolImplTest, lastWorker) {
    WorkerPoolImpl workerPool;
    ASSERT_THROW(workerPool.lastWorker(), std::exception);
    auto workerMock = std::make_shared<MockWorker>();
    EXPECT_CALL(*workerMock.get(), setRepresentingIndex(kExpectedIndex)).Times(1);
    ASSERT_NO_THROW(workerPool.addModelWorker(workerMock));
    ASSERT_NO_THROW(workerPool.lastWorker());
}

TEST_F(GNA_Request_WorkerPoolImplTest, lastWorker_const) {
    WorkerPoolImpl workerPool;
    const WorkerPoolImpl& constWorkerPool = workerPool;
    ASSERT_THROW(constWorkerPool.lastWorker(), std::exception);
    auto workerMock = std::make_shared<MockWorker>();
    EXPECT_CALL(*workerMock.get(), setRepresentingIndex(kExpectedIndex)).Times(1);
    ASSERT_NO_THROW(workerPool.addModelWorker(workerMock));
    ASSERT_NO_THROW(constWorkerPool.lastWorker());
}

TEST_F(GNA_Request_WorkerPoolImplTest, findFreeModelWorker) {
    WorkerPoolImpl workerPool;
    auto workerMock = std::make_shared<MockWorker>();

    std::shared_ptr<Worker> foundWorker;
    // return nullptr if no worker added
    ASSERT_NO_THROW(foundWorker = workerPool.findFreeModelWorker());
    ASSERT_EQ(foundWorker.get(), nullptr);
    EXPECT_CALL(*workerMock.get(), setRepresentingIndex(kExpectedIndex)).Times(1);
    ASSERT_NO_THROW(workerPool.addModelWorker(workerMock));

    // return nullptr if all workers busy
    EXPECT_CALL(*workerMock.get(), isFree()).Times(1).WillOnce(Return(false));
    ASSERT_NO_THROW(foundWorker = workerPool.findFreeModelWorker());
    ASSERT_EQ(foundWorker.get(), nullptr);

    // return valid pointer if at least one worker free
    EXPECT_CALL(*workerMock.get(), isFree()).Times(1).WillOnce(Return(true));
    ASSERT_NO_THROW(foundWorker = workerPool.findFreeModelWorker());
    ASSERT_NE(foundWorker.get(), nullptr);
    auto worker = std::static_pointer_cast<Worker>(workerMock);
    ASSERT_EQ(foundWorker.get(), worker.get());
}