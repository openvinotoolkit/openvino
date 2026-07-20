// Copyright (C) 2018-2026 Intel Corporation
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
    got_message = msg;
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

TEST(ov_util, ov_get_error_info_returns_static_pointer) {
    const char* p1 = ov_get_error_info(GENERAL_ERROR);
    const char* p2 = ov_get_error_info(GENERAL_ERROR);
    EXPECT_EQ(p1, p2) << "ov_get_error_info must return a pointer to static storage, not a newly allocated string";

    // Every known status code must return a non-null, non-empty string.
    const auto all_codes = {OK,
                            GENERAL_ERROR,
                            NOT_IMPLEMENTED,
                            NETWORK_NOT_LOADED,
                            PARAMETER_MISMATCH,
                            NOT_FOUND,
                            OUT_OF_BOUNDS,
                            UNEXPECTED,
                            REQUEST_BUSY,
                            RESULT_NOT_READY,
                            NOT_ALLOCATED,
                            INFER_NOT_STARTED,
                            NETWORK_NOT_READ,
                            INFER_CANCELLED,
                            INVALID_C_PARAM,
                            UNKNOWN_C_ERROR,
                            NOT_IMPLEMENT_C_METHOD,
                            UNKNOW_EXCEPTION};
    for (auto code : all_codes) {
        const char* msg = ov_get_error_info(code);
        EXPECT_NE(nullptr, msg) << "ov_get_error_info returned NULL for status " << code;
        EXPECT_GT(std::strlen(msg), 0u) << "ov_get_error_info returned empty string for status " << code;
        // Intentionally NOT calling ov_free(msg): the pointer is borrowed static storage.
    }
}

TEST(ov_util, ov_get_error_info_out_of_range) {
    const char* msg = ov_get_error_info(static_cast<ov_status_e>(-9999));
    EXPECT_NE(nullptr, msg);
    EXPECT_GT(std::strlen(msg), 0u);
}
