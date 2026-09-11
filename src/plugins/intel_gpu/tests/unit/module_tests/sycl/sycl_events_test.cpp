// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#ifdef OV_GPU_WITH_SYCL_RT

#include "sycl_test_context.hpp"

#include "runtime/sycl/sycl_base_event.hpp"
#include "runtime/sycl/sycl_event.hpp"
#include "runtime/sycl/sycl_user_event.hpp"

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

#endif  // OV_GPU_WITH_SYCL_RT
