// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#ifdef OV_GPU_WITH_SYCL_RT

#include "sycl_test_context.hpp"

#include "runtime/sycl/sycl_base_event.hpp"
#include "runtime/sycl/sycl_event.hpp"

using namespace cldnn;
using namespace sycl_tests;


/*
USER EVENTS:
*/

TEST(sycl_event, can_create_user_event_as_complete) {
    auto ctx = create_sycl_test_context();
    auto user_ev = ctx.sycl_test_stream->create_user_event(true);

    ASSERT_NE(std::dynamic_pointer_cast<cldnn::sycl::sycl_base_event>(user_ev), nullptr);
    ASSERT_TRUE(user_ev->is_set());
}

// TODO: Add tests when create_user_event(false) is implemented
// TEST(sycl_event, create_user_event_as_not_complete)
// TEST(sycl_event, can_create_user_event_as_not_complete_and_set)

TEST(sycl_event, can_create_user_event_as_complete_and_wait) {
    auto ctx = create_sycl_test_context();
    auto user_ev = ctx.sycl_test_stream->create_user_event(true);
    user_ev->wait();

    ASSERT_NE(std::dynamic_pointer_cast<cldnn::sycl::sycl_base_event>(user_ev), nullptr);
    ASSERT_TRUE(user_ev->is_set());
}

// TODO: Add test when create_user_event(false) is implemented
// TEST(sycl_event, can_create_user_event_as_not_complete_set_and_wait)

#endif  // OV_GPU_WITH_SYCL_RT
