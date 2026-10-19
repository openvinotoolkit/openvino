// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "runtime/ocl/ocl_device_clock.hpp"
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

namespace {
// Synthesized start timestamps need the optional OpenCL 2.1 device/host timer; without
// it device_clock_sync stays invalid and user events fall back to two intervals.
bool has_device_timer(const engine& eng) {
#if defined(CL_VERSION_2_1)
    cl_ulong device_ts = 0, host_ts = 0;
    auto cl_device = dynamic_cast<const ocl::ocl_engine&>(eng).get_cl_device().get();
    return clGetDeviceAndHostTimer(cl_device, &device_ts, &host_ts) == CL_SUCCESS;
#else
    (void)eng;
    return false;
#endif
}
}  // namespace

TEST(user_event, can_create_as_complete) {
    auto& stream = get_test_stream();
    auto user_ev = stream.create_user_event(true);

    ASSERT_NE(std::dynamic_pointer_cast<ocl::ocl_user_event>(user_ev), nullptr);
}

// Needs the 2.1 timer directly to bracket the expected value, so it only exists when
// the plugin is targeting OpenCL 2.1 or newer.
#if defined(CL_VERSION_2_1)
TEST(user_event, profiling_reports_device_domain_start_timestamp) {
    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::enable_profiling(true));
    auto stream = engine.create_stream(config);

    if (!has_device_timer(engine))
        GTEST_SKIP() << "device does not support clGetDeviceAndHostTimer";

    cl_ulong device_before = 0, host_before = 0;
    auto cl_device = dynamic_cast<const ocl::ocl_engine&>(engine).get_cl_device().get();
    ASSERT_EQ(clGetDeviceAndHostTimer(cl_device, &device_before, &host_before), CL_SUCCESS);

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
#endif  // CL_VERSION_2_1

TEST(user_event, profiling_start_timestamps_are_ordered) {
    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::enable_profiling(true));
    auto stream = engine.create_stream(config);

    if (!has_device_timer(engine))
        GTEST_SKIP() << "device does not support clGetDeviceAndHostTimer";

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

namespace {
using ns = std::chrono::nanoseconds;
using anchor = ocl::device_clock_sync::anchor;

// base maps host 1ms -> device 5ms.
constexpr anchor make_base() { return anchor{ns(1000000), ns(5000000), true}; }

std::chrono::nanoseconds map_host(const anchor& late, int64_t host_ns) {
    return ocl::device_clock_sync::interpolate(make_base(), late, ns(host_ns));
}
}  // namespace

TEST(device_clock_sync, interpolate_uses_offset_only_without_second_anchor) {
    EXPECT_EQ(map_host(anchor{}, 2000000), ns(6000000));
}

TEST(device_clock_sync, interpolate_applies_rate_from_second_anchor) {
    // 2 ms of host maps to 2.1 ms of device -> rate 1.05.
    const anchor late{ns(3000000), ns(7100000), true};
    EXPECT_EQ(map_host(late, 2000000), ns(6050000));
}

TEST(device_clock_sync, interpolate_ignores_rate_when_anchors_are_too_close) {
    // 0.5 ms apart is below the minimum span, so the 2x rate must be ignored.
    const anchor late{ns(1500000), ns(6000000), true};
    EXPECT_EQ(map_host(late, 2000000), ns(6000000));
}

TEST(device_clock_sync, interpolate_ignores_implausible_rate) {
    // rate 1.5 is far outside real host/device drift and must be rejected.
    const anchor late{ns(3000000), ns(8000000), true};
    EXPECT_EQ(map_host(late, 2000000), ns(6000000));
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
