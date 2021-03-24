// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cpp/ie_infer_request.hpp>
#include <cpp_interfaces/exception2status.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;


TEST(InferRequestCPPTests, throwsOnInitWithNull) {
    IInferRequest::Ptr nlptr = nullptr;
    ASSERT_THROW(InferRequest req(nlptr), InferenceEngine::Exception);
}

TEST(InferRequestCPPTests, throwsOnUninitializedSetBlob) {
    InferRequest req;
    ASSERT_THROW(req.SetBlob({}, {}), InferenceEngine::Exception);
}

TEST(InferRequestCPPTests, throwsOnUninitializedGetBlob) {
    InferRequest req;
    ASSERT_THROW(req.GetBlob({}), InferenceEngine::Exception);
}

TEST(InferRequestCPPTests, throwsOnUninitializedSetBlobPreproc) {
    InferRequest req;
    ASSERT_THROW(req.SetBlob({}, {}, {}), InferenceEngine::Exception);
}

TEST(InferRequestCPPTests, throwsOnUninitializedGetPreProcess) {
    InferRequest req;
    ASSERT_THROW(req.GetPreProcess({}), InferenceEngine::Exception);
}

TEST(InferRequestCPPTests, throwsOnUninitializedInfer) {
    InferRequest req;
    ASSERT_THROW(req.Infer(), InferenceEngine::Exception);
}

TEST(InferRequestCPPTests, throwsOnUninitializedGetPerformanceCounts) {
    InferRequest req;
    ASSERT_THROW(req.GetPerformanceCounts(), InferenceEngine::Exception);
}

TEST(InferRequestCPPTests, throwsOnUninitializedSetInput) {
    InferRequest req;
    ASSERT_THROW(req.SetInput({{}}), InferenceEngine::Exception);
}

TEST(InferRequestCPPTests, throwsOnUninitializedSetOutput) {
    InferRequest req;
    ASSERT_THROW(req.SetOutput({{}}), InferenceEngine::Exception);
}

TEST(InferRequestCPPTests, throwsOnUninitializedSetBatch) {
    InferRequest req;
    ASSERT_THROW(req.SetBatch({}), InferenceEngine::Exception);
}

TEST(InferRequestCPPTests, throwsOnUninitializedStartAsync) {
    InferRequest req;
    ASSERT_THROW(req.StartAsync(), InferenceEngine::Exception);
}

TEST(InferRequestCPPTests, throwsOnUninitializedWait) {
    InferRequest req;
    ASSERT_THROW(req.Wait({}), InferenceEngine::Exception);
}

TEST(InferRequestCPPTests, throwsOnUninitializedSetCompletionCallback) {
    InferRequest req;
    std::function<void(InferRequest, StatusCode)> f;
    ASSERT_THROW(req.SetCompletionCallback(f), InferenceEngine::Exception);
}

TEST(InferRequestCPPTests, throwsOnUninitializedCast) {
    InferRequest req;
    ASSERT_THROW((void)static_cast<IInferRequest::Ptr &>(req), InferenceEngine::Exception);
}

TEST(InferRequestCPPTests, throwsOnUninitializedQueryState) {
    InferRequest req;
    ASSERT_THROW(req.QueryState(), InferenceEngine::Exception);
}
