// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "runtime/ocl/ocl_device_clock.hpp"
#include "runtime/ocl/ocl_engine.hpp"
#include "runtime/ocl/ocl_event.hpp"
#include "runtime/ocl/ocl_stream.hpp"
#include "runtime/ocl/ocl_user_event.hpp"

#include "CL/cl.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>

using namespace cldnn;
using namespace ::tests;

class cross_queue_event : public ::testing::TestWithParam<bool> {
protected:
    void check_upload_visibility(bool external_barrier, QueueTypes queue_type = QueueTypes::in_order);
};

void cross_queue_event::check_upload_visibility(bool external_barrier, QueueTypes queue_type) {
    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::enable_profiling(GetParam()),
                        ov::intel_gpu::queue_type(queue_type));
    auto transfer = engine.create_stream(config);
    auto other_transfer = external_barrier ? engine.create_stream(config) : nullptr;
    auto compute = engine.create_stream(config);
    auto& transfer_queue = dynamic_cast<ocl::ocl_stream&>(*transfer).get_cl_queue();
    auto& compute_queue = dynamic_cast<ocl::ocl_stream&>(*compute).get_cl_queue();
    const auto& context = dynamic_cast<const ocl::ocl_engine&>(engine).get_cl_context();

    std::array<uint32_t, 256> source{};
    std::array<uint32_t, 256> result{};
    for (size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<uint32_t>(index * 17 + 3);
    }
    cl::Buffer slot(context, CL_MEM_READ_WRITE, sizeof(source));
    cl::Buffer output(context, CL_MEM_READ_WRITE, sizeof(source));
    cl::Program program(context, "__kernel void consume(__global const uint* input, __global uint* output) { size_t index = get_global_id(0); output[index] = input[index]; }");
    program.build();
    cl::Kernel kernel(program, "consume");
    kernel.setArg(0, slot);
    kernel.setArg(1, output);
    cl::UserEvent upload_gate(context);
    cl::UserEvent other_gate(context);
    std::vector<cl::Event> upload_deps{upload_gate};
    cl::Event upload_done;
    auto first_bytes = external_barrier ? sizeof(source) / 2 : sizeof(source);
    transfer_queue.enqueueWriteBuffer(slot, CL_FALSE, 0, first_bytes, source.data(), &upload_deps, &upload_done);

    std::vector<cl::Event> consumer_deps{upload_done};
    if (external_barrier) {
        cl::Event other_done;
        std::vector<cl::Event> other_deps{other_gate};
        auto& other_queue = dynamic_cast<ocl::ocl_stream&>(*other_transfer).get_cl_queue();
        other_queue.enqueueWriteBuffer(slot, CL_FALSE, first_bytes, sizeof(source) - first_bytes,
                                       source.data() + source.size() / 2, &other_deps, &other_done);
        dynamic_cast<ocl::ocl_stream&>(*compute).enqueue_external_barrier(
            {std::make_shared<ocl::ocl_event>(upload_done), std::make_shared<ocl::ocl_event>(other_done)});
    } else {
        compute_queue.enqueueBarrierWithWaitList(&consumer_deps);
    }
    cl::Event consumer_done;
    compute_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(source.size()), cl::NullRange, nullptr, &consumer_done);
    transfer->flush();
    if (other_transfer) {
        other_transfer->flush();
    }
    compute->flush();

    auto consumer_status = consumer_done.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
    upload_gate.setStatus(CL_COMPLETE);
    if (external_barrier) {
        upload_done.wait();
        auto partial_status = consumer_done.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
        other_gate.setStatus(CL_COMPLETE);
        EXPECT_NE(partial_status, CL_COMPLETE);
    }
    consumer_done.wait();
    compute_queue.enqueueReadBuffer(output, CL_TRUE, 0, sizeof(result), result.data());

    EXPECT_NE(consumer_status, CL_COMPLETE);
    EXPECT_EQ(result, source);
}

TEST_P(cross_queue_event, native_barrier_preserves_upload_visibility) {
    check_upload_visibility(false);
}

TEST_P(cross_queue_event, external_barrier_preserves_upload_visibility) {
    check_upload_visibility(true);
}

TEST_P(cross_queue_event, external_barrier_preserves_upload_visibility_out_of_order) {
    check_upload_visibility(true, QueueTypes::out_of_order);
}

TEST_P(cross_queue_event, external_barrier_rejects_missing_native_events) {
    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::enable_profiling(GetParam()));
    auto stream = engine.create_stream(config);
    auto& compute = dynamic_cast<ocl::ocl_stream&>(*stream);
    EXPECT_NO_THROW(compute.enqueue_external_barrier({}));
    EXPECT_THROW(compute.enqueue_external_barrier({nullptr}), ov::Exception);
    EXPECT_THROW(compute.enqueue_external_barrier({stream->create_base_event()}), ov::Exception);
    auto grouped = stream->group_events({stream->enqueue_marker({}), stream->enqueue_marker({})});
    EXPECT_THROW(compute.enqueue_external_barrier({grouped}), ov::Exception);
}

TEST_P(cross_queue_event, external_barrier_rejects_foreign_context_and_failed_events) {
    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::enable_profiling(GetParam()));
    auto stream = engine.create_stream(config);
    auto& compute = dynamic_cast<ocl::ocl_stream&>(*stream);
    const auto& ocl_engine = dynamic_cast<const ocl::ocl_engine&>(engine);
    cl::Context other_context(ocl_engine.get_cl_device());
    cl::UserEvent foreign(other_context);
    foreign.setStatus(CL_COMPLETE);
    EXPECT_THROW(compute.enqueue_external_barrier({std::make_shared<ocl::ocl_event>(foreign)}), ov::Exception);
    cl::UserEvent failed(ocl_engine.get_cl_context());
    failed.setStatus(-1);
    EXPECT_THROW(compute.enqueue_external_barrier({std::make_shared<ocl::ocl_event>(failed)}), ov::Exception);
}

TEST_P(cross_queue_event, external_barrier_propagates_late_dependency_failure) {
    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::enable_profiling(GetParam()),
                        ov::intel_gpu::queue_type(QueueTypes::in_order));
    auto stream = engine.create_stream(config);
    auto& compute = dynamic_cast<ocl::ocl_stream&>(*stream);
    cl::UserEvent gate(dynamic_cast<const ocl::ocl_engine&>(engine).get_cl_context());
    compute.enqueue_external_barrier({std::make_shared<ocl::ocl_event>(gate)});
    auto completion = stream->enqueue_marker({});
    gate.setStatus(-1);
    stream->flush();
    EXPECT_THROW(completion->wait(), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(les_capability,
                         cross_queue_event,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                             return info.param ? "profiling_enabled" : "profiling_disabled";
                         });

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
