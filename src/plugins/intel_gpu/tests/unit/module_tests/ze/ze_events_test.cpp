// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef OV_GPU_WITH_ZE_RT

#include "ze_test_context.hpp"

#include "runtime/ze/ze_base_event.hpp"
#include "runtime/ze/ze_counter_based_event.hpp"

using namespace cldnn;
using namespace ze_tests;


/*
USER EVENTS:
*/

TEST(ze_event, can_create_user_event_as_complete) {
	auto ctx = create_ze_test_context();
	auto user_ev = ctx.ze_test_stream->create_user_event(true);

	ASSERT_NE(std::dynamic_pointer_cast<ze::ze_base_event>(user_ev), nullptr);
	ASSERT_TRUE(user_ev->is_set());
}

TEST(ze_event, can_create_user_event_as_not_complete) {
	auto ctx = create_ze_test_context();
	auto user_ev = ctx.ze_test_stream->create_user_event(false);

	ASSERT_NE(std::dynamic_pointer_cast<ze::ze_base_event>(user_ev), nullptr);
	ASSERT_FALSE(user_ev->is_set());
}

TEST(ze_event, can_create_user_event_as_not_complete_and_set) {
	auto ctx = create_ze_test_context();
	auto user_ev = ctx.ze_test_stream->create_user_event(false);

	
	if (std::dynamic_pointer_cast<ze::ze_counter_based_event>(user_ev) != nullptr) {
		GTEST_SKIP() << "Counter based events are always created as complete";
	}

	user_ev->set();
	ASSERT_NE(std::dynamic_pointer_cast<ze::ze_base_event>(user_ev), nullptr);
	ASSERT_TRUE(user_ev->is_set());
}

TEST(ze_event, can_create_user_event_as_complete_and_wait) {
	auto ctx = create_ze_test_context();
	auto user_ev = ctx.ze_test_stream->create_user_event(true);
	user_ev->wait();

	ASSERT_NE(std::dynamic_pointer_cast<ze::ze_base_event>(user_ev), nullptr);
	ASSERT_TRUE(user_ev->is_set());
}

TEST(ze_event, can_create_user_event_as_not_complete_set_and_wait) {
	auto ctx = create_ze_test_context();
	auto user_ev = ctx.ze_test_stream->create_user_event(false);
	user_ev->set();
	user_ev->wait();

	ASSERT_NE(std::dynamic_pointer_cast<ze::ze_base_event>(user_ev), nullptr);
	ASSERT_TRUE(user_ev->is_set());
}

/*
COUNTER BASED EVENTS:
*/

TEST(ze_event, can_create_counter_based_event) {
	auto ctx = create_ze_test_context();
	auto base_ev = ctx.ze_test_stream->create_base_event();

	if (std::dynamic_pointer_cast<ze::ze_counter_based_event>(base_ev) == nullptr)
		GTEST_SKIP() << "Counter based events not supported by this stream";

	ASSERT_NE(std::dynamic_pointer_cast<ze::ze_base_event>(base_ev), nullptr);
	ASSERT_TRUE(base_ev->is_set());
}

TEST(ze_event, can_create_counter_based_event_and_wait) {
	auto ctx = create_ze_test_context();
	auto base_ev = ctx.ze_test_stream->create_base_event();

	if (std::dynamic_pointer_cast<ze::ze_counter_based_event>(base_ev) == nullptr)
		GTEST_SKIP() << "Counter based events not supported by this stream";

	base_ev->wait();

	ASSERT_NE(std::dynamic_pointer_cast<ze::ze_base_event>(base_ev), nullptr);
	ASSERT_TRUE(base_ev->is_set());
}

#endif  // OV_GPU_WITH_ZE_RT
