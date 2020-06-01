// Copyright (C) 2018-2020 Intel Corporation
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
    ASSERT_THROW(InferRequest req(nlptr), InferenceEngine::details::InferenceEngineException);
}

TEST(InferRequestCPPTests, throwsOnUninitializedSetBlob) {
    InferRequest req;
    ASSERT_THROW(req.SetBlob({}, {}), InferenceEngine::details::InferenceEngineException);
}

TEST(InferRequestCPPTests, throwsOnUninitializedGetBlob) {
    InferRequest req;
    ASSERT_THROW(req.GetBlob({}), InferenceEngine::details::InferenceEngineException);
}

TEST(InferRequestCPPTests, throwsOnUninitializedSetBlobPreproc) {
    InferRequest req;
    ASSERT_THROW(req.SetBlob({}, {}, {}), InferenceEngine::details::InferenceEngineException);
}

TEST(InferRequestCPPTests, throwsOnUninitializedGetPreProcess) {
    InferRequest req;
    ASSERT_THROW(req.GetPreProcess({}), InferenceEngine::details::InferenceEngineException);
}

TEST(InferRequestCPPTests, throwsOnUninitializedInfer) {
    InferRequest req;
    ASSERT_THROW(req.Infer(), InferenceEngine::details::InferenceEngineException);
}

TEST(InferRequestCPPTests, throwsOnUninitializedGetPerformanceCounts) {
    InferRequest req;
    ASSERT_THROW(req.GetPerformanceCounts(), InferenceEngine::details::InferenceEngineException);
}

TEST(InferRequestCPPTests, throwsOnUninitializedSetInput) {
    InferRequest req;
    ASSERT_THROW(req.SetInput({{}}), InferenceEngine::details::InferenceEngineException);
}

TEST(InferRequestCPPTests, throwsOnUninitializedSetOutput) {
    InferRequest req;
    ASSERT_THROW(req.SetOutput({{}}), InferenceEngine::details::InferenceEngineException);
}

TEST(InferRequestCPPTests, throwsOnUninitializedSetBatch) {
    InferRequest req;
    ASSERT_THROW(req.SetBatch({}), InferenceEngine::details::InferenceEngineException);
}

TEST(InferRequestCPPTests, throwsOnUninitializedStartAsync) {
    InferRequest req;
    ASSERT_THROW(req.StartAsync(), InferenceEngine::details::InferenceEngineException);
}

TEST(InferRequestCPPTests, throwsOnUninitializedWait) {
    InferRequest req;
    ASSERT_THROW(req.Wait({}), InferenceEngine::details::InferenceEngineException);
}

TEST(InferRequestCPPTests, throwsOnUninitializedSetCompletionCallback) {
    InferRequest req;
    std::function<void(InferRequest, StatusCode)> f;
    ASSERT_THROW(req.SetCompletionCallback(f), InferenceEngine::details::InferenceEngineException);
}

TEST(InferRequestCPPTests, throwsOnUninitializedCast) {
    InferRequest req;
    ASSERT_THROW(auto &ireq = static_cast<IInferRequest::Ptr &>(req), InferenceEngine::details::InferenceEngineException);
}
