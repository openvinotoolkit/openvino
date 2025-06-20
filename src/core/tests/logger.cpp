// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <iostream>
#include <regex>
#include <sstream>

#include "openvino/core/log_util.hpp"
#include "openvino/util/log.hpp"

namespace ov::util::test {

using LogEntries = std::tuple<const char*, int, const char*>;

class TestLogHelper : public testing::TestWithParam<LogEntries> {
    std::ostream* const actual_out_stream = &std::cout;
    std::streambuf* const actual_out_buf = actual_out_stream->rdbuf();

    // LogEntries
    const char* m_log_path;
    int m_log_line;
    const char* m_log_message;

protected:
    void SetUp() override {
        reset_log_handler();
        actual_out_stream->flush();
        actual_out_stream->rdbuf(m_mock_out_stream.rdbuf());

        std::tie(m_log_path, m_log_line, m_log_message) = GetParam();

        m_log_handler = [message = &m_callback_message](const std::string& msg) {
            *message = msg;
        };
    }

    void TearDown() override {
        actual_out_stream->rdbuf(actual_out_buf);
        reset_log_handler();
    }

    auto log_test_params() {
        LogHelper{LOG_TYPE::_LOG_TYPE_INFO, m_log_path, m_log_line, get_log_handler()}.stream() << m_log_message;
    }

    auto get_log_regex() const {
        std::stringstream log_regex;
        log_regex << m_log_path << ".*" << m_log_line << ".*" << m_log_message;
        return std::regex{log_regex.str()};
    }

    auto are_params_logged_to(const std::string& buf) {
        return std::regex_search(buf, get_log_regex());
    }

    std::stringstream m_mock_out_stream;

    std::string m_callback_message;
    log_handler_t m_log_handler;
};

TEST_P(TestLogHelper, std_cout) {
    log_test_params();
    EXPECT_TRUE(are_params_logged_to(m_mock_out_stream.str()));
}

TEST_P(TestLogHelper, callback) {
    set_log_handler(&m_log_handler);
    log_test_params();
    EXPECT_TRUE(m_mock_out_stream.str().empty());
    EXPECT_TRUE(are_params_logged_to(m_callback_message));
}

TEST_P(TestLogHelper, reset) {
    set_log_handler(&m_log_handler);
    reset_log_handler();
    log_test_params();
    EXPECT_TRUE(are_params_logged_to(m_mock_out_stream.str()));
    EXPECT_TRUE(m_callback_message.empty());
}

TEST_P(TestLogHelper, no_log) {
    set_log_handler(nullptr);
    log_test_params();
    EXPECT_TRUE(m_mock_out_stream.str().empty());
    EXPECT_TRUE(m_callback_message.empty());
}

INSTANTIATE_TEST_SUITE_P(Logging,
                         TestLogHelper,
                         ::testing::ValuesIn(std::vector<LogEntries>{{"the_path", 42, "tEst-mEssagE"},
                                                                     {"in the middle", 0.f, "the nowhere"}}));
}  // namespace ov::util::test
