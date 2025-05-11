// Copyright (C) 2018-2025 Intel Corporation
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
