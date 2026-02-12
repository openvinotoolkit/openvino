// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <atomic>
#include <iostream>
#include <memory>
#include <sstream>

#include "common_test_utils/file_utils.hpp"
#include "openvino/shutdown.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace {

std::string get_shutdown_test_lib_path() {
    std::string name("shutdown_test_lib");
    return ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                              name + OV_BUILD_POSTFIX);
}

TEST(ShutdownTest, TestLibCallbackRegistered) {
    int counter = 0;
    {
        auto handle = ov::util::load_shared_object(get_shutdown_test_lib_path());
        ASSERT_NE(handle, nullptr);

        using SetCallbackCounterFunc = void (*)(int*);
        auto set_callback_counter =
            reinterpret_cast<SetCallbackCounterFunc>(ov::util::get_symbol(handle, "set_callback_counter"));
        ASSERT_NE(set_callback_counter, nullptr);

        set_callback_counter(&counter);

        ASSERT_EQ(counter, 0);
    }  // Library unloaded here

    EXPECT_EQ(counter, 1);
}

TEST(ShutdownTest, TestMultipleLoadUnload) {
    int counter1 = 0;
    int counter2 = 0;

    {
        auto handle1 = ov::util::load_shared_object(get_shutdown_test_lib_path());
        ASSERT_NE(handle1, nullptr);

        using SetCallbackCounterFunc = void (*)(int*);
        auto set_callback_counter1 =
            reinterpret_cast<SetCallbackCounterFunc>(ov::util::get_symbol(handle1, "set_callback_counter"));
        set_callback_counter1(&counter1);

        ASSERT_EQ(counter1, 0);
    }  // First unload

    EXPECT_EQ(counter1, 1);

    {
        auto handle2 = ov::util::load_shared_object(get_shutdown_test_lib_path());
        ASSERT_NE(handle2, nullptr);

        using SetCallbackCounterFunc = void (*)(int*);
        auto set_callback_counter2 =
            reinterpret_cast<SetCallbackCounterFunc>(ov::util::get_symbol(handle2, "set_callback_counter"));
        set_callback_counter2(&counter2);

        ASSERT_EQ(counter2, 0);
    }  // Second unload

    EXPECT_EQ(counter2, 1);
}

}  // namespace
