// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <future>

#include "openvino/runtime/exception.hpp"

#include "overload/overload_test_utils_npu.hpp"

namespace ov {
namespace test {
namespace behavior {
using OVInferRequestCancellationTestsNPU = OVInferRequestTestsNPU;

TEST_P(OVInferRequestCancellationTestsNPU, canCancelAsyncRequest) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.cancel());
    try {
        req.wait();
    } catch (const ov::Cancelled&) {
        SUCCEED();
    }
}

TEST_P(OVInferRequestCancellationTestsNPU, CanResetAfterCancelAsyncRequest) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.cancel());
    try {
        req.wait();
    } catch (const ov::Cancelled&) {
        SUCCEED();
    }
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
}

TEST_P(OVInferRequestCancellationTestsNPU, canCancelBeforeAsyncRequest) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.cancel());
}

TEST_P(OVInferRequestCancellationTestsNPU, canCancelInferRequest) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    auto infer = std::async(std::launch::async, [&req] {
        req.infer();
    });
    while (!req.wait_for({})) {
    }
    OV_ASSERT_NO_THROW(req.cancel());
    try {
        infer.get();
    } catch (const ov::Cancelled&) {
        SUCCEED();
    }
}
}  // namespace behavior
}  // namespace test
}  // namespace ov
