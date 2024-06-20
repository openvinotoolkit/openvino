// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <future>

#include "overload/overload_test_utils_npu.hpp"

namespace ov {
namespace test {
namespace behavior {
using OVInferRequestCallbackTestsNPU = OVInferRequestTestsNPU;

TEST_P(OVInferRequestCallbackTestsNPU, canCallAsyncWithCompletionCallback) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    bool is_called = false;
    OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
        // HSD_1805940120: Wait on starting callback return HDDL_ERROR_INVAL_TASK_HANDLE
        ASSERT_EQ(exception_ptr, nullptr);
        is_called = true;
    }));
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    ASSERT_TRUE(is_called);
}

TEST_P(OVInferRequestCallbackTestsNPU, syncInferDoesNotCallCompletionCallback) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    bool is_called = false;
    req.set_callback([&](std::exception_ptr exception_ptr) {
        ASSERT_EQ(nullptr, exception_ptr);
        is_called = true;
    });
    req.infer();
    ASSERT_FALSE(is_called);
}

// test that can wait all callbacks on dtor
TEST_P(OVInferRequestCallbackTestsNPU, canStartSeveralAsyncInsideCompletionCallbackWithSafeDtor) {
    const int NUM_ITER = 10;
    struct TestUserData {
        std::atomic<int> numIter = {0};
        std::promise<bool> promise;
    };
    TestUserData data;

    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
        if (exception_ptr) {
            data.promise.set_exception(exception_ptr);
        } else {
            if (data.numIter.fetch_add(1) != NUM_ITER) {
                req.start_async();
            } else {
                data.promise.set_value(true);
            }
        }
    }));
    auto future = data.promise.get_future();
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
    future.wait();
    auto callbackStatus = future.get();
    ASSERT_TRUE(callbackStatus);
    auto dataNumIter = data.numIter - 1;
    ASSERT_EQ(NUM_ITER, dataNumIter);
}

TEST_P(OVInferRequestCallbackTestsNPU, returnGeneralErrorIfCallbackThrowException) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.set_callback([](std::exception_ptr) {
        OPENVINO_THROW("Throw");
    }));
    OV_ASSERT_NO_THROW(req.start_async());
    ASSERT_THROW(req.wait(), ov::Exception);
}

TEST_P(OVInferRequestCallbackTestsNPU, ReturnResultNotReadyFromWaitInAsyncModeForTooSmallTimeout) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    std::promise<std::chrono::system_clock::time_point> callbackTimeStamp;
    auto callbackTimeStampFuture = callbackTimeStamp.get_future();
    // add a callback to the request and capture the timestamp
    OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
        if (exception_ptr) {
            callbackTimeStamp.set_exception(exception_ptr);
        } else {
            callbackTimeStamp.set_value(std::chrono::system_clock::now());
        }
    }));
    OV_ASSERT_NO_THROW(req.start_async());
    bool ready = false;
    OV_ASSERT_NO_THROW(ready = req.wait_for({}));
    // get timestamp taken AFTER return from the wait(STATUS_ONLY)
    const auto afterWaitTimeStamp = std::chrono::system_clock::now();
    // IF the callback timestamp is larger than the afterWaitTimeStamp
    // then we should observe false ready result
    if (afterWaitTimeStamp < callbackTimeStampFuture.get()) {
        ASSERT_FALSE(ready);
    }
    OV_ASSERT_NO_THROW(req.wait());
}

TEST_P(OVInferRequestCallbackTestsNPU, ImplDoesNotCopyCallback) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    {
        auto somePtr = std::make_shared<int>(42);
        OV_ASSERT_NO_THROW(req.set_callback([somePtr](std::exception_ptr exception_ptr) {
            ASSERT_EQ(nullptr, exception_ptr);
            ASSERT_EQ(1, somePtr.use_count());
        }));
    }
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
