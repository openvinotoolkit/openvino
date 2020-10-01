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


TEST(TransformationTests, smoke_throwsOnInitWithNull) {
    IInferRequest::Ptr nlptr = nullptr;
    ASSERT_THROW(InferRequest req(nlptr), InferenceEngine::details::InferenceEngineException);
}

TEST(TransformationTests, smoke_throwsOnUninitializedSetBlob) {
    InferRequest req;
    ASSERT_THROW(req.SetBlob({}, {}), InferenceEngine::details::InferenceEngineException);
}

TEST(TransformationTests, smoke_throwsOnUninitializedGetBlob) {
    InferRequest req;
    ASSERT_THROW(req.GetBlob({}), InferenceEngine::details::InferenceEngineException);
}

TEST(TransformationTests, smoke_throwsOnUninitializedSetBlobPreproc) {
    InferRequest req;
    ASSERT_THROW(req.SetBlob({}, {}, {}), InferenceEngine::details::InferenceEngineException);
}

TEST(TransformationTests, smoke_throwsOnUninitializedGetPreProcess) {
    InferRequest req;
    ASSERT_THROW(req.GetPreProcess({}), InferenceEngine::details::InferenceEngineException);
}

TEST(TransformationTests, smoke_throwsOnUninitializedInfer) {
    InferRequest req;
    ASSERT_THROW(req.Infer(), InferenceEngine::details::InferenceEngineException);
}

TEST(TransformationTests, smoke_throwsOnUninitializedGetPerformanceCounts) {
    InferRequest req;
    ASSERT_THROW(req.GetPerformanceCounts(), InferenceEngine::details::InferenceEngineException);
}

TEST(TransformationTests, smoke_throwsOnUninitializedSetInput) {
    InferRequest req;
    ASSERT_THROW(req.SetInput({{}}), InferenceEngine::details::InferenceEngineException);
}

TEST(TransformationTests, smoke_throwsOnUninitializedSetOutput) {
    InferRequest req;
    ASSERT_THROW(req.SetOutput({{}}), InferenceEngine::details::InferenceEngineException);
}

TEST(TransformationTests, smoke_throwsOnUninitializedSetBatch) {
    InferRequest req;
    ASSERT_THROW(req.SetBatch({}), InferenceEngine::details::InferenceEngineException);
}

TEST(TransformationTests, smoke_throwsOnUninitializedStartAsync) {
    InferRequest req;
    ASSERT_THROW(req.StartAsync(), InferenceEngine::details::InferenceEngineException);
}

TEST(TransformationTests, smoke_throwsOnUninitializedWait) {
    InferRequest req;
    ASSERT_THROW(req.Wait({}), InferenceEngine::details::InferenceEngineException);
}

TEST(TransformationTests, smoke_throwsOnUninitializedSetCompletionCallback) {
    InferRequest req;
    std::function<void(InferRequest, StatusCode)> f;
    ASSERT_THROW(req.SetCompletionCallback(f), InferenceEngine::details::InferenceEngineException);
}

TEST(TransformationTests, smoke_throwsOnUninitializedCast) {
    InferRequest req;
    ASSERT_THROW(auto &ireq = static_cast<IInferRequest::Ptr &>(req), InferenceEngine::details::InferenceEngineException);
}
