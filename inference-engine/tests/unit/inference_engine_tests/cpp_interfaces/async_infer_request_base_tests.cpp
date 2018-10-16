// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include <cpp_interfaces/impl/mock_inference_plugin_internal.hpp>
#include <cpp_interfaces/interface/mock_iasync_infer_request_internal.hpp>

#include <ie_version.hpp>
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include <cpp_interfaces/base/ie_infer_async_request_base.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class InferRequestBaseTests : public ::testing::Test {
protected:
    std::shared_ptr<MockIAsyncInferRequestInternal> mock_impl;
    shared_ptr<IInferRequest> request;
    ResponseDesc dsc;

    virtual void TearDown() {
    }

    virtual void SetUp() {
        mock_impl.reset(new MockIAsyncInferRequestInternal());
        request = details::shared_from_irelease(new InferRequestBase<MockIAsyncInferRequestInternal>(mock_impl));
    }
};

// StartAsync
TEST_F(InferRequestBaseTests, canForwardStartAsync) {
    EXPECT_CALL(*mock_impl.get(), StartAsync()).Times(1);
    ASSERT_EQ(OK, request->StartAsync(&dsc));
}

TEST_F(InferRequestBaseTests, canReportErrorInStartAsync) {
    EXPECT_CALL(*mock_impl.get(), StartAsync()).WillOnce(Throw(std::runtime_error("compare")));
    ASSERT_NE(request->StartAsync(&dsc), OK);
    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(InferRequestBaseTests, canCatchUnknownErrorInStartAsync) {
    EXPECT_CALL(*mock_impl.get(), StartAsync()).WillOnce(Throw(5));
    ASSERT_EQ(UNEXPECTED, request->StartAsync(nullptr));
}

// GetUserData
TEST_F(InferRequestBaseTests, canForwardGetUserData) {
    void **data = nullptr;
    EXPECT_CALL(*mock_impl.get(), GetUserData(data)).Times(1);
    ASSERT_EQ(OK, request->GetUserData(data, &dsc));
}

TEST_F(InferRequestBaseTests, canReportErrorInGetUserData) {
    EXPECT_CALL(*mock_impl.get(), GetUserData(_)).WillOnce(Throw(std::runtime_error("compare")));
    ASSERT_NE(request->GetUserData(nullptr, &dsc), OK);
    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(InferRequestBaseTests, canCatchUnknownErrorInGetUserData) {
    EXPECT_CALL(*mock_impl.get(), GetUserData(_)).WillOnce(Throw(5));
    ASSERT_EQ(UNEXPECTED, request->GetUserData(nullptr, nullptr));
}

// SetUserData
TEST_F(InferRequestBaseTests, canForwardSetUserData) {
    void *data = nullptr;
    EXPECT_CALL(*mock_impl.get(), SetUserData(data)).Times(1);
    ASSERT_EQ(OK, request->SetUserData(data, &dsc));
}

TEST_F(InferRequestBaseTests, canReportErrorInSetUserData) {
    EXPECT_CALL(*mock_impl.get(), SetUserData(_)).WillOnce(Throw(std::runtime_error("compare")));
    ASSERT_NE(request->SetUserData(nullptr, &dsc), OK);
    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(InferRequestBaseTests, canCatchUnknownErrorInSetUserData) {
    EXPECT_CALL(*mock_impl.get(), SetUserData(_)).WillOnce(Throw(5));
    ASSERT_EQ(UNEXPECTED, request->SetUserData(nullptr, nullptr));
}

// Wait
TEST_F(InferRequestBaseTests, canForwardWait) {
    int64_t ms = 0;
    EXPECT_CALL(*mock_impl.get(), Wait(ms)).WillOnce(Return(StatusCode::OK));
    ASSERT_EQ(OK, request->Wait(ms, &dsc)) << dsc.msg;
}

TEST_F(InferRequestBaseTests, canReportErrorInWait) {
    EXPECT_CALL(*mock_impl.get(), Wait(_)).WillOnce(Throw(std::runtime_error("compare")));
    int64_t ms = 0;
    ASSERT_NE(request->Wait(ms, &dsc), OK);
    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(InferRequestBaseTests, canCatchUnknownErrorInWait) {
    EXPECT_CALL(*mock_impl.get(), Wait(_)).WillOnce(Throw(5));
    int64_t ms = 0;
    ASSERT_EQ(UNEXPECTED, request->Wait(ms, nullptr));
}

// Infer
TEST_F(InferRequestBaseTests, canForwardInfer) {
    EXPECT_CALL(*mock_impl.get(), Infer()).Times(1);
    ASSERT_EQ(OK, request->Infer(&dsc));
}

TEST_F(InferRequestBaseTests, canReportErrorInInfer) {
    EXPECT_CALL(*mock_impl.get(), Infer()).WillOnce(Throw(std::runtime_error("compare")));
    ASSERT_NE(request->Infer(&dsc), OK);
    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(InferRequestBaseTests, canCatchUnknownErrorInInfer) {
    EXPECT_CALL(*mock_impl.get(), Infer()).WillOnce(Throw(5));
    ASSERT_EQ(UNEXPECTED, request->Infer(nullptr));
}

// GetPerformanceCounts
TEST_F(InferRequestBaseTests, canForwardGetPerformanceCounts) {
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> info;
    EXPECT_CALL(*mock_impl.get(), GetPerformanceCounts(Ref(info))).Times(1);
    ASSERT_EQ(OK, request->GetPerformanceCounts(info, &dsc));
}

TEST_F(InferRequestBaseTests, canReportErrorInGetPerformanceCounts) {
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> info;
    EXPECT_CALL(*mock_impl.get(), GetPerformanceCounts(_)).WillOnce(Throw(std::runtime_error("compare")));
    ASSERT_NE(request->GetPerformanceCounts(info, &dsc), OK);
    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(InferRequestBaseTests, canCatchUnknownErrorInGetPerformanceCounts) {
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> info;
    EXPECT_CALL(*mock_impl.get(), GetPerformanceCounts(_)).WillOnce(Throw(5));
    ASSERT_EQ(UNEXPECTED, request->GetPerformanceCounts(info, nullptr));
}

// GetBlob
TEST_F(InferRequestBaseTests, canForwardGetBlob) {
    Blob::Ptr data;
    const char *name = "";
    EXPECT_CALL(*mock_impl.get(), GetBlob(name, Ref(data))).Times(1);
    ASSERT_EQ(OK, request->GetBlob(name, data, &dsc));
}

TEST_F(InferRequestBaseTests, canReportErrorInGetBlob) {
    EXPECT_CALL(*mock_impl.get(), GetBlob(_, _)).WillOnce(Throw(std::runtime_error("compare")));
    Blob::Ptr data;
    ASSERT_NE(request->GetBlob(nullptr, data, &dsc), OK);
    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(InferRequestBaseTests, canCatchUnknownErrorInGetBlob) {
    Blob::Ptr data;
    EXPECT_CALL(*mock_impl.get(), GetBlob(_, _)).WillOnce(Throw(5));
    ASSERT_EQ(UNEXPECTED, request->GetBlob(nullptr, data, nullptr));
}

// SetBlob
TEST_F(InferRequestBaseTests, canForwardSetBlob) {
    Blob::Ptr data;
    const char *name = "";
    EXPECT_CALL(*mock_impl.get(), SetBlob(name, Ref(data))).Times(1);
    ASSERT_EQ(OK, request->SetBlob(name, data, &dsc));
}

TEST_F(InferRequestBaseTests, canReportErrorInSetBlob) {
    EXPECT_CALL(*mock_impl.get(), SetBlob(_, _)).WillOnce(Throw(std::runtime_error("compare")));
    Blob::Ptr data;
    ASSERT_NE(request->SetBlob(nullptr, data, &dsc), OK);
    ASSERT_STREQ(dsc.msg, "compare");
}

TEST_F(InferRequestBaseTests, canCatchUnknownErrorInSetBlob) {
    Blob::Ptr data;
    EXPECT_CALL(*mock_impl.get(), SetBlob(_, _)).WillOnce(Throw(5));
    ASSERT_EQ(UNEXPECTED, request->SetBlob(nullptr, data, nullptr));
}

// SetCompletionCallback
TEST_F(InferRequestBaseTests, canForwardSetCompletionCallback) {
    InferenceEngine::IInferRequest::CompletionCallback callback = nullptr;
    EXPECT_CALL(*mock_impl.get(), SetCompletionCallback(callback)).Times(1);
    ASSERT_NO_THROW(request->SetCompletionCallback(callback));
}

TEST_F(InferRequestBaseTests, canReportErrorInSetCompletionCallback) {
    EXPECT_CALL(*mock_impl.get(), SetCompletionCallback(_)).WillOnce(Throw(std::runtime_error("compare")));
    ASSERT_NO_THROW(request->SetCompletionCallback(nullptr));
}
