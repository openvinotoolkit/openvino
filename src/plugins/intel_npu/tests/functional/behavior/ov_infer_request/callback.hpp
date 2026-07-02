// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <chrono>
#include <future>
#include <memory>

#include "common/utils.hpp"

namespace ov {
namespace test {
namespace behavior {
using OVInferRequestCallbackTestsNPU = OVInferRequestTestsNPU;

TEST_P(OVInferRequestCallbackTestsNPU, callbackCanCallInfer) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());

    std::atomic<size_t> callback_calls{0};
    std::promise<void> done;
    auto done_future = done.get_future();
    std::atomic_bool done_signaled{false};

    OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
        ASSERT_EQ(nullptr, exception_ptr);
        callback_calls.fetch_add(1, std::memory_order_relaxed);
        req.infer();

        if (!done_signaled.exchange(true, std::memory_order_relaxed)) {
            done.set_value();
        }
    }));

    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());

    ASSERT_EQ(done_future.wait_for(std::chrono::seconds(30)), std::future_status::ready);
    ASSERT_EQ(callback_calls.load(std::memory_order_relaxed), 1);
}

TEST_P(OVInferRequestCallbackTestsNPU, callbackCanCallStartAsyncAndWait) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());

    std::atomic<size_t> callback_calls{0};
    std::promise<void> nested_done;
    auto nested_done_future = nested_done.get_future();
    std::atomic_bool nested_done_signaled{false};

    OV_ASSERT_NO_THROW(req.set_callback([&](std::exception_ptr exception_ptr) {
        ASSERT_EQ(nullptr, exception_ptr);
        const auto call_index = callback_calls.fetch_add(1, std::memory_order_relaxed);

        if (call_index == 0) {
            req.start_async();
            req.wait();
        } else if (call_index == 1) {
            if (!nested_done_signaled.exchange(true, std::memory_order_relaxed)) {
                nested_done.set_value();
            }
        }
    }));

    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());

    ASSERT_EQ(nested_done_future.wait_for(std::chrono::seconds(30)), std::future_status::ready);
    ASSERT_GE(callback_calls.load(std::memory_order_relaxed), 2);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
