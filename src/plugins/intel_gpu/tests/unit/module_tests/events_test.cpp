// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "runtime/ocl/ocl_engine.hpp"
#include "runtime/ocl/ocl_stream.hpp"
#include "runtime/ocl/ocl_user_event.hpp"

#include "CL/cl.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>

using namespace cldnn;
using namespace ::tests;

TEST(user_event, can_create_as_complete) {
    auto& stream = get_test_stream();
    auto user_ev = stream.create_user_event(true);

    ASSERT_NE(std::dynamic_pointer_cast<ocl::ocl_user_event>(user_ev), nullptr);
}

TEST(user_event, profiling_reports_device_domain_start_timestamp) {
    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::enable_profiling(true));
    auto stream = engine.create_stream(config);

    cl_ulong device_before = 0, host_before = 0;
    auto cl_device = dynamic_cast<const ocl::ocl_engine&>(engine).get_cl_device().get();
    if (clGetDeviceAndHostTimer(cl_device, &device_before, &host_before) != CL_SUCCESS)
        GTEST_SKIP() << "device does not support clGetDeviceAndHostTimer";

    auto user_ev = stream->create_user_event(true);
    const auto profiling_info = user_ev->get_profiling_info();

    cl_ulong device_after = 0, host_after = 0;
    ASSERT_EQ(clGetDeviceAndHostTimer(cl_device, &device_after, &host_after), CL_SUCCESS);

    ASSERT_EQ(profiling_info.size(), 3);
    EXPECT_EQ(profiling_info[0].stage, instrumentation::profiling_stage::duration);
    EXPECT_EQ(profiling_info[1].stage, instrumentation::profiling_stage::starting);
    EXPECT_EQ(profiling_info[2].stage, instrumentation::profiling_stage::executing);

    EXPECT_TRUE(profiling_info[1].is_valid_start);
    EXPECT_TRUE(profiling_info[2].is_valid_start);

    // The start must land in the device domain, inside the bracketing window.
    const auto start_ns = profiling_info[2].start.count();
    EXPECT_GT(start_ns, 0);
    EXPECT_GE(start_ns, static_cast<int64_t>(device_before) - 1000000);  // 1 ms of slack
    EXPECT_LE(start_ns, static_cast<int64_t>(device_after) + 1000000);
}

TEST(user_event, profiling_start_timestamps_are_ordered) {
    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::enable_profiling(true));
    auto stream = engine.create_stream(config);

    auto first = stream->create_user_event(true);
    // Separate them beyond clock resolution so this tests ordering.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    auto second = stream->create_user_event(true);

    const auto first_info = first->get_profiling_info();
    const auto second_info = second->get_profiling_info();

    ASSERT_EQ(first_info.size(), 3);
    ASSERT_EQ(second_info.size(), 3);
    ASSERT_TRUE(first_info[2].is_valid_start);
    ASSERT_TRUE(second_info[2].is_valid_start);

    EXPECT_LE(first_info[2].start.count(), second_info[2].start.count());
}

TEST(user_event, no_start_timestamp_when_profiling_disabled) {
    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::enable_profiling(false));
    auto stream = engine.create_stream(config);

    auto user_ev = stream->create_user_event(true);
    const auto profiling_info = user_ev->get_profiling_info();

    ASSERT_EQ(profiling_info.size(), 2);
    EXPECT_EQ(profiling_info[0].stage, instrumentation::profiling_stage::duration);
    EXPECT_EQ(profiling_info[1].stage, instrumentation::profiling_stage::executing);
    EXPECT_FALSE(profiling_info[0].is_valid_start);
    EXPECT_FALSE(profiling_info[1].is_valid_start);
}

TEST(user_event, can_create_as_not_complete) {
    auto& stream = get_test_stream();
    auto user_ev = stream.create_user_event(false);

    ASSERT_NE(std::dynamic_pointer_cast<ocl::ocl_user_event>(user_ev), nullptr);
}

TEST(user_event, can_create_as_not_complete_and_set) {
    auto& stream = get_test_stream();
    auto user_ev = stream.create_user_event(false);
    user_ev->set();

    ASSERT_NE(std::dynamic_pointer_cast<ocl::ocl_user_event>(user_ev), nullptr);
}

TEST(user_event, can_create_as_complete_and_wait) {
    auto& stream = get_test_stream();
    auto user_ev = stream.create_user_event(true);
    user_ev->wait();

    ASSERT_NE(std::dynamic_pointer_cast<ocl::ocl_user_event>(user_ev), nullptr);
}

TEST(user_event, can_create_as_not_complete_set_and_wait) {
    auto& stream = get_test_stream();
    auto user_ev = stream.create_user_event(false);
    user_ev->set();
    user_ev->wait();

    ASSERT_NE(std::dynamic_pointer_cast<ocl::ocl_user_event>(user_ev), nullptr);
}

TEST(user_event, fail_on_create_as_not_complete_and_wait) {
    auto& stream = get_test_stream();
    auto user_ev = stream.create_user_event(false);
    ASSERT_THROW(user_ev->wait(), std::runtime_error);
}
