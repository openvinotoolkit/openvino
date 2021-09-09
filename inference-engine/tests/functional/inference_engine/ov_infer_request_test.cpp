// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cpp/ie_infer_request.hpp>
#include <openvino/runtime/infer_request.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;


TEST(InferRequestOVTests, throwsOnUninitializedSetBlob) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.set_blob({}, {}), InferenceEngine::NotAllocated);
}

TEST(InferRequestOVTests, throwsOnUninitializedGetBlob) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.get_blob({}), InferenceEngine::NotAllocated);
}

TEST(InferRequestOVTests, throwsOnUninitializedInfer) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.infer(), InferenceEngine::NotAllocated);
}

TEST(InferRequestOVTests, throwsOnUninitializedGetPerformanceCounts) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.get_profiling_info(), InferenceEngine::NotAllocated);
}

TEST(InferRequestOVTests, throwsOnUninitializedSetInput) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.set_input({{}}), InferenceEngine::NotAllocated);
}

TEST(InferRequestOVTests, throwsOnUninitializedSetOutput) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.set_output({{}}), InferenceEngine::NotAllocated);
}

TEST(InferRequestOVTests, throwsOnUninitializedSetBatch) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.set_batch({}), InferenceEngine::NotAllocated);
}

TEST(InferRequestOVTests, throwsOnUninitializedStartAsync) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.start_async(), InferenceEngine::NotAllocated);
}

TEST(InferRequestOVTests, throwsOnUninitializedWait) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.wait(), InferenceEngine::NotAllocated);
}

TEST(InferRequestOVTests, throwsOnUninitializedWaitFor) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.wait_for({}), InferenceEngine::NotAllocated);
}

TEST(InferRequestOVTests, throwsOnUninitializedSetCompletionCallback) {
    ov::runtime::InferRequest req;
    std::function<void(std::exception_ptr)> f;
    ASSERT_THROW(req.set_callback(f), InferenceEngine::NotAllocated);
}

TEST(InferRequestOVTests, throwsOnUninitializedQueryState) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.query_state(), InferenceEngine::NotAllocated);
}
