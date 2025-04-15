// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/except.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/remote_tensor.hpp>

using namespace ::testing;
using namespace std;

TEST(InferRequestOVTests, throwsOnUninitializedSetTensor) {
    ov::InferRequest req;
    ASSERT_THROW(req.set_tensor("", {}), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedGetTensor) {
    ov::InferRequest req;
    ASSERT_THROW(req.get_tensor(""), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedInfer) {
    ov::InferRequest req;
    ASSERT_THROW(req.infer(), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedGetPerformanceCounts) {
    ov::InferRequest req;
    ASSERT_THROW(req.get_profiling_info(), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedStartAsync) {
    ov::InferRequest req;
    ASSERT_THROW(req.start_async(), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedWait) {
    ov::InferRequest req;
    ASSERT_THROW(req.wait(), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedWaitFor) {
    ov::InferRequest req;
    ASSERT_THROW(req.wait_for({}), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedSetCompletionCallback) {
    ov::InferRequest req;
    std::function<void(std::exception_ptr)> f;
    ASSERT_THROW(req.set_callback(f), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedQueryState) {
    ov::InferRequest req;
    ASSERT_THROW(req.query_state(), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedSetRemoteTensorWithName) {
    ov::InferRequest req;
    ov::RemoteTensor remote_tensor;
    ASSERT_THROW(req.set_tensor("", remote_tensor), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedSetInputRemoteTensor) {
    ov::InferRequest req;
    ov::RemoteTensor remote_tensor;
    ASSERT_THROW(req.set_input_tensor(0, remote_tensor), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedSetOutputRemoteTensor) {
    ov::InferRequest req;
    ov::RemoteTensor remote_tensor;
    ASSERT_THROW(req.set_output_tensor(0, remote_tensor), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnUninitializedSetRemoteTensor) {
    ov::InferRequest req;
    ov::RemoteTensor remote_tensor;
    ASSERT_THROW(req.set_tensor(ov::Output<const ov::Node>(), remote_tensor), ov::Exception);
}

TEST(InferRequestOVTests, throwsOnGetCompiledModel) {
    ov::InferRequest req;
    ASSERT_THROW(req.get_compiled_model(), ov::Exception);
}
