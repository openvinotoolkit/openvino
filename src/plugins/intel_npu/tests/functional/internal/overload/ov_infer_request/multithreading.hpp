// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <future>

#include "overload/overload_test_utils_npu.hpp"

namespace ov {
namespace test {
namespace behavior {
using OVInferRequestMultithreadingTestsNPU = OVInferRequestTestsNPU;

TEST_P(OVInferRequestMultithreadingTestsNPU, canRun3SyncRequestsConsistentlyFromThreads) {
    ov::InferRequest req1, req2, req3;
    OV_ASSERT_NO_THROW(req1 = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req2 = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req3 = execNet.create_infer_request());

    auto f1 = std::async(std::launch::async, [&] {
        req1.infer();
    });
    auto f2 = std::async(std::launch::async, [&] {
        req2.infer();
    });
    auto f3 = std::async(std::launch::async, [&] {
        req3.infer();
    });

    f1.wait();
    f2.wait();
    f3.wait();

    OV_ASSERT_NO_THROW(f1.get());
    OV_ASSERT_NO_THROW(f2.get());
    OV_ASSERT_NO_THROW(f3.get());
}

TEST_P(OVInferRequestMultithreadingTestsNPU, canRun3AsyncRequestsConsistentlyFromThreadsWithoutWait) {
    ov::InferRequest req1, req2, req3;
    OV_ASSERT_NO_THROW(req1 = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req2 = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req3 = execNet.create_infer_request());

    OV_ASSERT_NO_THROW(req1.infer());
    OV_ASSERT_NO_THROW(req2.infer());
    OV_ASSERT_NO_THROW(req3.infer());

    auto f1 = std::async(std::launch::async, [&] {
        req1.start_async();
    });
    auto f2 = std::async(std::launch::async, [&] {
        req2.start_async();
    });
    auto f3 = std::async(std::launch::async, [&] {
        req3.start_async();
    });

    f1.wait();
    f2.wait();
    f3.wait();

    OV_ASSERT_NO_THROW(f1.get());
    OV_ASSERT_NO_THROW(f2.get());
    OV_ASSERT_NO_THROW(f3.get());
}

TEST_P(OVInferRequestMultithreadingTestsNPU, canRun3AsyncRequestsConsistentlyWithWait) {
    ov::InferRequest req1, req2, req3;
    OV_ASSERT_NO_THROW(req1 = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req2 = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req3 = execNet.create_infer_request());

    req1.start_async();
    OV_ASSERT_NO_THROW(req1.wait());

    req2.start_async();
    OV_ASSERT_NO_THROW(req2.wait());

    req3.start_async();
    OV_ASSERT_NO_THROW(req3.wait());
}

TEST_P(OVInferRequestMultithreadingTestsNPU, canRun3AsyncRequestsParallelWithWait) {
    ov::InferRequest req1, req2, req3;
    OV_ASSERT_NO_THROW(req1 = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req2 = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req3 = execNet.create_infer_request());

    req1.start_async();
    req2.start_async();
    req3.start_async();

    OV_ASSERT_NO_THROW(req2.wait());
    OV_ASSERT_NO_THROW(req1.wait());
    OV_ASSERT_NO_THROW(req3.wait());
}
}  // namespace behavior
}  // namespace test
}  // namespace ov
