// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cpp/ie_infer_request.hpp>
#include <openvino/core/except.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/remote_tensor.hpp>
#include <openvino/runtime/compiled_model.hpp>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

TEST(InferRequestOVTests, throwsOnUninitializedSetTensor) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.set_tensor("", {}), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedGetTensor) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.get_tensor(""), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedInfer) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.infer(), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedGetPerformanceCounts) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.get_profiling_info(), ov::Exception);
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

TEST(InferRequestOVTests, throwsOnUninitializedSetRemoteTensorWithName) {
    ov::runtime::InferRequest req;
    ov::runtime::RemoteTensor remote_tensor;
    ASSERT_THROW(req.set_tensor("", remote_tensor), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedSetInputRemoteTensor) {
    ov::runtime::InferRequest req;
    ov::runtime::RemoteTensor remote_tensor;
    ASSERT_THROW(req.set_input_tensor(0, remote_tensor), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedSetOutputRemoteTensor) {
    ov::runtime::InferRequest req;
    ov::runtime::RemoteTensor remote_tensor;
    ASSERT_THROW(req.set_output_tensor(0, remote_tensor), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedSetRemoteTensor) {
    ov::runtime::InferRequest req;
    ov::runtime::RemoteTensor remote_tensor;
    ASSERT_THROW(req.set_tensor(ov::Output<const ov::Node>(), remote_tensor), ov::Exception);
}


TEST(InferRequestOVTests, throwsOnGetCompiledModel) {
    ov::runtime::InferRequest req;
    ASSERT_THROW(req.get_compiled_model(), ov::Exception);
}
