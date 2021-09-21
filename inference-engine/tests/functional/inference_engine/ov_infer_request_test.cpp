// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cpp/ie_infer_request.hpp>
#include <openvino/core/except.hpp>
#include <openvino/runtime/infer_request.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;


TEST(InferRequestOVTests, throwsOnUninitializedSetBlob) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.set_blob({}, {}), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedGetBlob) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.get_blob({}), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedInfer) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.infer(), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedGetPerformanceCounts) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.get_profiling_info(), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedSetInput) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.set_input({{}}), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedSetOutput) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.set_output({{}}), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedSetBatch) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.set_batch({}), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedStartAsync) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.start_async(), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedWait) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.wait(), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedWaitFor) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.wait_for({}), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedSetCompletionCallback) {
    ov::runtime::InferRequest req;
    std::function<void(std::exception_ptr)> f;
    ASSERT_THROW(req.set_callback(f), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedQueryState) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.query_state(), ov::Exception);
}
