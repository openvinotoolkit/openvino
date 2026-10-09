// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#ifdef OV_GPU_WITH_SYCL_RT

#include "sycl_test_context.hpp"

#include "intel_gpu/runtime/kernel.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "intel_gpu/runtime/kernel_builder.hpp"

#include "runtime/sycl/sycl_base_event.hpp"
#include "runtime/sycl/sycl_event.hpp"
#include "runtime/sycl/sycl_user_event.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace cldnn;
using namespace sycl_tests;


/*
USER EVENTS:
*/

TEST(sycl_event, can_create_user_event_as_complete) {
    auto ctx = create_sycl_test_context();
    auto user_ev = ctx.sycl_test_stream->create_user_event(true);

    ASSERT_NE(std::dynamic_pointer_cast<cldnn::sycl::sycl_user_event>(user_ev), nullptr);
    ASSERT_TRUE(user_ev->is_set());
}

TEST(sycl_event, can_create_user_event_as_not_complete) {
    auto ctx = create_sycl_test_context();
    auto user_ev = ctx.sycl_test_stream->create_user_event(false);

    ASSERT_NE(std::dynamic_pointer_cast<cldnn::sycl::sycl_user_event>(user_ev), nullptr);
    ASSERT_FALSE(user_ev->is_set());
}

TEST(sycl_event, can_create_user_event_as_not_complete_and_set) {
    auto ctx = create_sycl_test_context();
    auto user_ev = ctx.sycl_test_stream->create_user_event(false);
    ASSERT_FALSE(user_ev->is_set());
    user_ev->set();

    ASSERT_NE(std::dynamic_pointer_cast<cldnn::sycl::sycl_user_event>(user_ev), nullptr);
    ASSERT_TRUE(user_ev->is_set());
}

TEST(sycl_event, can_create_user_event_as_complete_and_wait) {
    auto ctx = create_sycl_test_context();
    auto user_ev = ctx.sycl_test_stream->create_user_event(true);
    user_ev->wait();

    ASSERT_NE(std::dynamic_pointer_cast<cldnn::sycl::sycl_user_event>(user_ev), nullptr);
    ASSERT_TRUE(user_ev->is_set());
}

TEST(sycl_event, can_create_user_event_as_not_complete_set_and_wait) {
    auto ctx = create_sycl_test_context();
    auto user_ev = ctx.sycl_test_stream->create_user_event(false);
    user_ev->set();
    user_ev->wait();

    ASSERT_NE(std::dynamic_pointer_cast<cldnn::sycl::sycl_user_event>(user_ev), nullptr);
    ASSERT_TRUE(user_ev->is_set());
}

TEST(sycl_event, fail_on_create_as_not_complete_and_wait) {
    auto ctx = create_sycl_test_context();
    auto user_ev = ctx.sycl_test_stream->create_user_event(false);
    ASSERT_THROW(user_ev->wait(), std::runtime_error);
}

// A user event has no native ::sycl::event while it is not set, so get() must reject it.
TEST(sycl_event, fail_on_get_native_event_when_not_set) {
    auto ctx = create_sycl_test_context();
    auto user_ev = ctx.sycl_test_stream->create_user_event(false);

    auto sycl_ev = std::dynamic_pointer_cast<cldnn::sycl::sycl_base_event>(user_ev);
    ASSERT_NE(sycl_ev, nullptr);
    ASSERT_THROW(sycl_ev->get(), ov::Exception);

    user_ev->set();
    ASSERT_NO_THROW(sycl_ev->get());
}

// A user event has no native ::sycl::event while it is not set, and wait_for_events() collects the
// native event of every dependency, so it goes through get() and is rejected.
TEST(sycl_event, fail_on_wait_for_events_when_not_set) {
    auto ctx = create_sycl_test_context();
    auto user_ev = ctx.sycl_test_stream->create_user_event(false);

    ASSERT_THROW(ctx.sycl_test_stream->wait_for_events({ user_ev }), std::runtime_error);

    user_ev->set();
    ASSERT_NO_THROW(ctx.sycl_test_stream->wait_for_events({ user_ev }));
}

// A user event has no native ::sycl::event while it is not set, so it can't be aggregated:
// sycl_events picks one of its dependencies as the native event it waits on, and a user event
// simply has none to offer until set() is called.
TEST(sycl_event, fail_on_group_events_when_not_set) {
    auto ctx = create_sycl_test_context();
    auto user_ev = ctx.sycl_test_stream->create_user_event(false);
    auto base_ev = ctx.sycl_test_stream->enqueue_marker({}, false);

    // group_events() returns deps[0] as is for a single dependency, so a second event is needed to
    // make it build a sycl_events aggregate and go through update_last_sycl_event().
    // Both orders must be rejected: the guard runs before the queue stamp comparison, so detection
    // does not depend on the position of the user event in the list.
    ASSERT_THROW(ctx.sycl_test_stream->group_events({ user_ev, base_ev }), ov::Exception);
    ASSERT_THROW(ctx.sycl_test_stream->group_events({ base_ev, user_ev }), ov::Exception);

    user_ev->set();
    ASSERT_NO_THROW(ctx.sycl_test_stream->group_events({ user_ev, base_ev }));
}

TEST(sycl_event, user_event_does_not_hide_incomplete_deps_in_group) {
    auto ctx = create_sycl_test_context();
    auto user_ev = ctx.sycl_test_stream->create_user_event(true);
    auto base_ev = ctx.sycl_test_stream->enqueue_marker({}, false);

    // The aggregate must rely on the native event of base_ev rather than on the placeholder event of
    // the user event, which is always in the complete state.
    auto grouped_ev = ctx.sycl_test_stream->group_events({ user_ev, base_ev });
    auto grouped_sycl_ev = std::dynamic_pointer_cast<cldnn::sycl::sycl_base_event>(grouped_ev);
    ASSERT_NE(grouped_sycl_ev, nullptr);

    auto base_sycl_ev = std::dynamic_pointer_cast<cldnn::sycl::sycl_base_event>(base_ev);
    ASSERT_NE(base_sycl_ev, nullptr);
    ASSERT_EQ(grouped_sycl_ev->get(), base_sycl_ev->get());

    ASSERT_NO_THROW(grouped_ev->wait());
    ASSERT_TRUE(grouped_ev->is_set());
}

// The time of a user event is spent on the host, so like ocl_user_event it must be reported as a
// "duration" interval, which the profiling report counts as CPU work, followed by "executing".
// SYCL has no device clock sync for user events, so no start timestamp is synthesized.
TEST(sycl_event, user_event_profiling_reports_duration_and_executing) {
    auto ctx = create_sycl_test_context();
    auto user_ev = ctx.sycl_test_stream->create_user_event(true);

    const auto profiling_info = user_ev->get_profiling_info();

    ASSERT_EQ(profiling_info.size(), 2);
    EXPECT_EQ(profiling_info[0].stage, instrumentation::profiling_stage::duration);
    EXPECT_EQ(profiling_info[1].stage, instrumentation::profiling_stage::executing);
    EXPECT_EQ(profiling_info[0].value->value(), profiling_info[1].value->value());
    EXPECT_FALSE(profiling_info[0].is_valid_start);
    EXPECT_FALSE(profiling_info[1].is_valid_start);
}

// Nothing is measured until set(), so an incomplete user event reports no intervals, and that empty
// result must not be cached: once set, the intervals cover the time from creation until set().
TEST(sycl_event, user_event_profiling_is_captured_once_set) {
    auto ctx = create_sycl_test_context();
    auto user_ev = ctx.sycl_test_stream->create_user_event(false);

    ASSERT_TRUE(user_ev->get_profiling_info().empty());

    const auto delay = std::chrono::milliseconds(1);
    std::this_thread::sleep_for(delay);
    user_ev->set();

    const auto profiling_info = user_ev->get_profiling_info();

    ASSERT_EQ(profiling_info.size(), 2);
    EXPECT_EQ(profiling_info[0].stage, instrumentation::profiling_stage::duration);
    EXPECT_EQ(profiling_info[1].stage, instrumentation::profiling_stage::executing);
    EXPECT_GE(profiling_info[0].value->value(), delay);
    EXPECT_EQ(profiling_info[0].value->value(), profiling_info[1].value->value());
}

/*
USER EVENTS AS DEPENDENCIES OF ENQUEUED COMMANDS:
*/

namespace {

// Picks the config which makes stream::get_expected_sync_method() return the requested method:
// profiling -> events, out_of_order queue -> barriers, in_order queue -> none.
std::shared_ptr<cldnn::stream> create_stream_with_sync_method(cldnn::engine& engine, SyncMethods sync_method) {
    auto config = ::tests::get_test_default_config(engine);
    config.set_property(ov::enable_profiling(sync_method == SyncMethods::events));
    config.set_property(ov::intel_gpu::queue_type(sync_method == SyncMethods::barriers ? QueueTypes::out_of_order
                                                                                       : QueueTypes::in_order));
    auto stream = engine.create_stream(config);
    OPENVINO_ASSERT(stream->get_sync_method() == sync_method, "[GPU] Unexpected sync method of the test stream");
    return stream;
}

kernel::ptr build_noop_kernel(cldnn::engine& engine) {
    const std::string source = R"__cl(
        __kernel void noop() {}
    )__cl";

    std::vector<kernel::ptr> kernels;
    engine.create_kernel_builder()->build_kernels(source.data(), source.size(), KernelFormat::SOURCE, "", kernels);
    OPENVINO_ASSERT(kernels.size() == 1, "[GPU] Failed to build the noop kernel for tests");
    return kernels[0];
}

std::string sync_method_name(const testing::TestParamInfo<SyncMethods>& info) {
    switch (info.param) {
        case SyncMethods::events:   return "events";
        case SyncMethods::barriers: return "barriers";
        case SyncMethods::none:     return "none";
        default:                    return "unknown";
    }
}

}  // namespace

// A user event which is not set yet can't be ordered against by the device, so it must be rejected
// as a dependency under every sync method. In barrier mode its zero queue stamp makes sync_events()
// skip it and SyncMethods::none ignores deps entirely, so these modes need an explicit check.
class sycl_user_event_deps : public ::testing::TestWithParam<SyncMethods> {};

TEST_P(sycl_user_event_deps, enqueue_marker_rejects_incomplete_user_event) {
    auto ctx = create_sycl_test_context();
    auto stream = create_stream_with_sync_method(*ctx.sycl_test_engine, GetParam());

    auto user_ev = stream->create_user_event(false);
    auto base_ev = stream->enqueue_marker({}, false);

    // The position of the user event in the list must not matter.
    ASSERT_THROW(stream->enqueue_marker({ user_ev }, false), ov::Exception);
    ASSERT_THROW(stream->enqueue_marker({ user_ev, base_ev }, false), ov::Exception);
    ASSERT_THROW(stream->enqueue_marker({ base_ev, user_ev }, false), ov::Exception);

    user_ev->set();
    event::ptr marker_ev;
    ASSERT_NO_THROW(marker_ev = stream->enqueue_marker({ user_ev, base_ev }, false));
    ASSERT_NE(marker_ev, nullptr);
    ASSERT_NO_THROW(marker_ev->wait());
}

TEST_P(sycl_user_event_deps, enqueue_kernel_rejects_incomplete_user_event) {
    auto ctx = create_sycl_test_context();
    auto stream = create_stream_with_sync_method(*ctx.sycl_test_engine, GetParam());
    auto kernel = build_noop_kernel(*ctx.sycl_test_engine);

    kernel_arguments_desc args_desc;
    kernel_arguments_data args;

    auto user_ev = stream->create_user_event(false);
    auto base_ev = stream->enqueue_marker({}, false);

    ASSERT_THROW(stream->enqueue_kernel(*kernel, args_desc, args, { user_ev }, false), ov::Exception);
    ASSERT_THROW(stream->enqueue_kernel(*kernel, args_desc, args, { user_ev, base_ev }, false), ov::Exception);
    ASSERT_THROW(stream->enqueue_kernel(*kernel, args_desc, args, { base_ev, user_ev }, false), ov::Exception);

    // An empty launch falls back to enqueue_marker(), which must reject it as well.
    kernel_arguments_desc empty_args_desc;
    empty_args_desc.workGroups.global = {0, 1, 1};
    ASSERT_THROW(stream->enqueue_kernel(*kernel, empty_args_desc, args, { user_ev }, false), ov::Exception);

    user_ev->set();
    event::ptr kernel_ev;
    ASSERT_NO_THROW(kernel_ev = stream->enqueue_kernel(*kernel, args_desc, args, { user_ev, base_ev }, false));
    ASSERT_NE(kernel_ev, nullptr);
    ASSERT_NO_THROW(kernel_ev->wait());
}

INSTANTIATE_TEST_SUITE_P(sycl_event,
                         sycl_user_event_deps,
                         ::testing::Values(SyncMethods::events, SyncMethods::barriers, SyncMethods::none),
                         sync_method_name);

#endif  // OV_GPU_WITH_SYCL_RT
