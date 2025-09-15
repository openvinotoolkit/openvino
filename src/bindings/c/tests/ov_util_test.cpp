// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstring>

#include "openvino/c/openvino.h"
#include "openvino/util/log.hpp"

namespace {
void test_log_msg(const char* msg) {
    ov::util::LogHelper{ov::util::LOG_TYPE::_LOG_TYPE_INFO, "1664804", 101}.stream() << msg;
}

std::string got_message;
void callback_func(const char* msg) {
    got_message = {msg};
}
bool compare_msg(const char* msg) {
    const auto msg_size = std::strlen(msg);
    if (got_message.size() < msg_size) {
        return false;
    }
    return std::strncmp(msg, got_message.data() + got_message.size() - msg_size, msg_size) == 0;
}
}  // namespace

TEST(ov_util, set_log_callback) {
    got_message.clear();
    test_log_msg("default");
    EXPECT_TRUE(got_message.empty());
    ov_util_set_log_callback(&callback_func);
    const char test_msg[] = "callback";
    test_log_msg(test_msg);
    EXPECT_TRUE(compare_msg(test_msg));
}

TEST(ov_util, reset_log_callback) {
    got_message.clear();
    ov_util_set_log_callback(&callback_func);
    ov_util_reset_log_callback();
    test_log_msg("default");
    EXPECT_TRUE(got_message.empty());
}

TEST(ov_util, no_log_callback) {
    got_message.clear();
    ov_util_set_log_callback(NULL);
    EXPECT_NO_THROW(test_log_msg("default"));
    EXPECT_TRUE(got_message.empty());
}
