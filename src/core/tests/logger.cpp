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

using LogEntries = std::tuple<LOG_TYPE, const char*, int, const char*>;

class TestLogHelper : public testing::TestWithParam<LogEntries> {
    std::ostream* const actual_out_stream = &std::cout;
    std::streambuf* const actual_out_buf = actual_out_stream->rdbuf();

    // LogEntries
    LOG_TYPE m_log_type;
    const char* m_log_path;
    int m_log_line;
    const char* m_log_message;

protected:
    void SetUp() override {
        LogDispatch::reset_callback();
        actual_out_stream->flush();
        actual_out_stream->rdbuf(m_mock_out_stream.rdbuf());

        std::tie(m_log_type, m_log_path, m_log_line, m_log_message) = GetParam();
    }

    void TearDown() override {
        actual_out_stream->rdbuf(actual_out_buf);
        LogDispatch::reset_callback();
    }

    auto log_test_params() {
        LogHelper{m_log_type, m_log_path, m_log_line, LogDispatch::get_callback()}.stream() << m_log_message;
    }

    auto get_log_regex() const {
        static auto log_prefix_pattern =
            std::map<LOG_TYPE, const std::string>{{LOG_TYPE::_LOG_TYPE_ERROR, R"(\[ERR\])"},
                                                  {LOG_TYPE::_LOG_TYPE_WARNING, R"(\[WARN\])"},
                                                  {LOG_TYPE::_LOG_TYPE_INFO, R"(\[INFO\])"},
                                                  {LOG_TYPE::_LOG_TYPE_DEBUG, R"(\[DEBUG\])"}};
        std::stringstream log_regex;
        if (LOG_TYPE::_LOG_TYPE_DEBUG_EMPTY != m_log_type) {
            log_regex << log_prefix_pattern[m_log_type] << ".*" << m_log_path << ".*" << m_log_line << ".*";
        }
        log_regex << m_log_message;
        return std::regex{log_regex.str()};
    }

    auto are_params_logged_to(const std::string& buf) {
        return std::regex_search(buf, get_log_regex());
    }

    std::stringstream m_mock_out_stream;

    std::string m_callback_message;
    LogDispatch::Callback m_log_callback{[this](std::string_view msg) {
        m_callback_message = msg;
    }};
};

TEST_P(TestLogHelper, std_cout) {
    log_test_params();
    EXPECT_TRUE(are_params_logged_to(m_mock_out_stream.str()));
}

TEST_P(TestLogHelper, callback) {
    LogDispatch::set_callback(m_log_callback);
    log_test_params();
    EXPECT_TRUE(m_mock_out_stream.str().empty());
    EXPECT_TRUE(are_params_logged_to(m_callback_message));
}

TEST_P(TestLogHelper, reset) {
    LogDispatch::set_callback(m_log_callback);
    LogDispatch::reset_callback();
    log_test_params();
    EXPECT_TRUE(are_params_logged_to(m_mock_out_stream.str()));
    EXPECT_TRUE(m_callback_message.empty());
}

TEST_P(TestLogHelper, no_log) {
    LogDispatch::set_callback(nullptr);
    log_test_params();
    EXPECT_TRUE(m_mock_out_stream.str().empty());
    EXPECT_TRUE(m_callback_message.empty());
}

INSTANTIATE_TEST_SUITE_P(Logging,
                         TestLogHelper,
                         ::testing::ValuesIn(std::vector<LogEntries>{
                             {LOG_TYPE::_LOG_TYPE_ERROR, "uno", 42, "tre"},
                             {LOG_TYPE::_LOG_TYPE_WARNING, "due", 101, "null"},
                             {LOG_TYPE::_LOG_TYPE_INFO, "to long", 3141592, "to read"},
                             {LOG_TYPE::_LOG_TYPE_DEBUG, "in the middle", 0xF, "the nowhere"},
                             {LOG_TYPE::_LOG_TYPE_DEBUG_EMPTY, "a:\\mio.c++", -101, "loading..."},
                         }));
}  // namespace ov::util::test
