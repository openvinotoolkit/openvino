// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "mock_subrequest.hpp"
#include "request/model_wrapper_factory.hpp"
#include "request/worker_impl.hpp"

using namespace GNAPluginNS;
using namespace request;
using namespace testing;

class GNA_Request_WorkerImplTest : public ::testing::Test {};

TEST_F(GNA_Request_WorkerImplTest, initDeinit) {
    ASSERT_THROW(WorkerImpl(nullptr, {}), std::exception);

    std::vector<std::shared_ptr<Subrequest>> subrequests;
    ASSERT_THROW(GNAPluginNS::request::WorkerImpl(nullptr, subrequests), std::exception);
    auto wrapper = ModelWrapperFactory::createTrivial();
    ASSERT_THROW(WorkerImpl(wrapper, {}), std::exception);

    subrequests.push_back(nullptr);
    ASSERT_THROW(WorkerImpl(wrapper, subrequests), std::exception);
    subrequests.clear();

    subrequests.push_back(std::make_shared<MockSubrequest>());
    ASSERT_NO_THROW(WorkerImpl(wrapper, subrequests));
}

TEST_F(GNA_Request_WorkerImplTest, model) {
    auto wrapper = ModelWrapperFactory::createTrivial();
    auto expectedModelPtr = &wrapper->object();
    std::vector<std::shared_ptr<Subrequest>> subrequests;
    subrequests.push_back(std::make_shared<MockSubrequest>());

    WorkerImpl worker(wrapper, subrequests);
    EXPECT_EQ(expectedModelPtr, worker.model());
}

TEST_F(GNA_Request_WorkerImplTest, model_const) {
    auto wrapper = ModelWrapperFactory::createTrivial();
    auto expectedModelPtr = &wrapper->object();
    std::vector<std::shared_ptr<Subrequest>> subrequests;
    subrequests.push_back(std::make_shared<MockSubrequest>());

    const WorkerImpl worker(wrapper, subrequests);
    EXPECT_EQ(expectedModelPtr, worker.model());
}

TEST_F(GNA_Request_WorkerImplTest, enqueueRequest) {
    auto wrapper = ModelWrapperFactory::createTrivial();
    std::vector<std::shared_ptr<Subrequest>> subrequests;

    auto subrequestMock1 = std::make_shared<MockSubrequest>();
    subrequests.push_back(subrequestMock1);
    auto subrequestMock2 = std::make_shared<MockSubrequest>();
    subrequests.push_back(subrequestMock2);
    WorkerImpl worker(wrapper, subrequests);

    // check if exception will be thrown if worker will be busy - at least one subrequest is pending.
    EXPECT_CALL(*subrequestMock1.get(), isPending()).Times(1).WillOnce(Return(false));
    EXPECT_CALL(*subrequestMock2.get(), isPending()).Times(1).WillOnce(Return(true));
    EXPECT_THROW(worker.enqueueRequest(), std::exception);

    EXPECT_CALL(*subrequestMock1.get(), isPending()).Times(1).WillOnce(Return(false));
    EXPECT_CALL(*subrequestMock2.get(), isPending()).Times(1).WillOnce(Return(false));
    EXPECT_CALL(*subrequestMock1.get(), enqueue()).Times(1);
    EXPECT_CALL(*subrequestMock2.get(), enqueue()).Times(1);
    EXPECT_NO_THROW(worker.enqueueRequest());
}

TEST_F(GNA_Request_WorkerImplTest, wait) {
    auto wrapper = ModelWrapperFactory::createTrivial();
    std::vector<std::shared_ptr<Subrequest>> subrequests;
    int64_t referenceTimeout = 1;

    auto subrequestMock1 = std::make_shared<MockSubrequest>();
    subrequests.push_back(subrequestMock1);
    WorkerImpl worker(wrapper, subrequests);

    // subrequest enuqued and completed on wait
    EXPECT_CALL(*subrequestMock1.get(), isPending()).Times(1).WillOnce(Return(true));
    EXPECT_CALL(*subrequestMock1.get(), isAborted()).Times(1).WillOnce(Return(false));
    EXPECT_CALL(*subrequestMock1.get(), wait(referenceTimeout)).Times(1).WillOnce(Return(RequestStatus::kCompleted));
    EXPECT_EQ(RequestStatus::kCompleted, worker.wait(referenceTimeout));

    // subrequest enuqued and aborted on wait
    EXPECT_CALL(*subrequestMock1.get(), isPending()).Times(1).WillOnce(Return(true));
    EXPECT_CALL(*subrequestMock1.get(), isAborted()).Times(1).WillOnce(Return(true));
    EXPECT_CALL(*subrequestMock1.get(), wait(referenceTimeout)).Times(1).WillOnce(Return(RequestStatus::kAborted));
    EXPECT_EQ(RequestStatus::kAborted, worker.wait(referenceTimeout));

    // subrequest enuqued and panding on wait
    EXPECT_CALL(*subrequestMock1.get(), isPending()).Times(1).WillOnce(Return(true));
    EXPECT_CALL(*subrequestMock1.get(), wait(referenceTimeout)).Times(1).WillOnce(Return(RequestStatus::kPending));
    EXPECT_EQ(RequestStatus::kPending, worker.wait(referenceTimeout));

    // subrequest not enuqued and not aborted
    EXPECT_CALL(*subrequestMock1.get(), isPending()).Times(1).WillOnce(Return(false));
    EXPECT_CALL(*subrequestMock1.get(), isAborted()).Times(1).WillOnce(Return(false));
    EXPECT_EQ(RequestStatus::kCompleted, worker.wait(referenceTimeout));

    // subrequest not enuqued and aborted
    EXPECT_CALL(*subrequestMock1.get(), isPending()).Times(1).WillOnce(Return(false));
    EXPECT_CALL(*subrequestMock1.get(), isAborted()).Times(1).WillOnce(Return(true));
    EXPECT_EQ(RequestStatus::kAborted, worker.wait(referenceTimeout));
}

TEST_F(GNA_Request_WorkerImplTest, isFree) {
    auto wrapper = ModelWrapperFactory::createTrivial();
    std::vector<std::shared_ptr<Subrequest>> subrequests;

    auto subrequestMock1 = std::make_shared<MockSubrequest>();
    subrequests.push_back(subrequestMock1);

    WorkerImpl worker(wrapper, subrequests);
    EXPECT_CALL(*subrequestMock1.get(), isPending()).Times(1).WillOnce(Return(true));
    EXPECT_EQ(false, worker.isFree());

    EXPECT_CALL(*subrequestMock1.get(), isPending()).Times(1).WillOnce(Return(false));
    EXPECT_EQ(true, worker.isFree());
}

TEST_F(GNA_Request_WorkerImplTest, result) {
    auto wrapper = ModelWrapperFactory::createTrivial();
    std::vector<std::shared_ptr<Subrequest>> subrequests;

    subrequests.push_back(std::make_shared<MockSubrequest>());

    InferenceEngine::BlobMap referenceBlobMap;
    const std::string referenceBlobKey("test1");
    referenceBlobMap[referenceBlobKey] = nullptr;
    WorkerImpl worker(wrapper, subrequests);
    worker.setResult(referenceBlobMap);
    EXPECT_EQ(referenceBlobMap, worker.result());

    // for rvalue reference
    auto referenceBlobMapCopy = referenceBlobMap;
    worker.setResult(std::move(referenceBlobMapCopy));
    EXPECT_EQ(referenceBlobMap, worker.result());
}
