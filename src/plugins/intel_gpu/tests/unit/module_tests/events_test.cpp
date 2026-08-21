// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "runtime/ocl/ocl_stream.hpp"
#include "runtime/ocl/ocl_user_event.hpp"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(user_event, can_create_as_complete) {
    auto& stream = get_test_stream();
    auto user_ev = stream.create_user_event(true);

    ASSERT_NE(std::dynamic_pointer_cast<ocl::ocl_user_event>(user_ev), nullptr);
}

TEST(user_event, profiling_has_no_device_start_timestamp) {
    auto& stream = get_test_stream();
    auto user_ev = stream.create_user_event(true);
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
