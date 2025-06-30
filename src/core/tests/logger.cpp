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

/* Capturing std::cout streambuffer doesn't work in CI job on Windows. It needs to be checked whether such test of
 * logging to std::cout is doable other way. Though it's disabled.
 */
#define ENABLE_LOGGING_TO_STD_COUT_TESTS 0

using LogEntries = std::tuple<LOG_TYPE, const char*, int, const char*>;

class TestLogHelper : public testing::TestWithParam<LogEntries> {
#if ENABLE_LOGGING_TO_STD_COUT_TESTS
    std::ostream* const actual_out_stream = &std::cout;
    std::streambuf* const actual_out_buf = actual_out_stream->rdbuf();
#endif

    // LogEntries
    LOG_TYPE m_log_type;
    const char* m_log_path;
    int m_log_line;
    const char* m_log_message;

protected:
    void SetUp() override {
        LogDispatch::reset_callback();
#if ENABLE_LOGGING_TO_STD_COUT_TESTS
        actual_out_stream->flush();
        actual_out_stream->rdbuf(m_mock_out_stream.rdbuf());
#endif

        std::tie(m_log_type, m_log_path, m_log_line, m_log_message) = GetParam();
    }

    void TearDown() override {
#if ENABLE_LOGGING_TO_STD_COUT_TESTS
        actual_out_stream->rdbuf(actual_out_buf);
#endif
        LogDispatch::reset_callback();
    }

    auto log_test_params() {
        LogHelper{m_log_type, m_log_path, m_log_line, LogDispatch::get_callback()}.stream() << m_log_message;
    }

    auto get_log_regex() const {
        static auto log_prefix_pattern =
            std::map<LOG_TYPE, const std::string>{{LOG_TYPE::_LOG_TYPE_ERROR, R"(\[ERROR\])"},
                                                  {LOG_TYPE::_LOG_TYPE_WARNING, R"(\[WARNING\])"},
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

#if ENABLE_LOGGING_TO_STD_COUT_TESTS
TEST_P(TestLogHelper, std_cout) {
    log_test_params();
    EXPECT_TRUE(are_params_logged_to(m_mock_out_stream.str()))
        << "Mock cout got: '" << m_mock_out_stream.str() << "'\n";
}
#endif

TEST_P(TestLogHelper, callback) {
    LogDispatch::set_callback(&m_log_callback);
    log_test_params();
    EXPECT_TRUE(m_mock_out_stream.str().empty()) << "Expected no cout. Got: '" << m_mock_out_stream.str() << "'\n";
    EXPECT_TRUE(are_params_logged_to(m_callback_message)) << "Callback got: '" << m_callback_message << "'\n";
}

TEST_P(TestLogHelper, toggle) {
    LogDispatch::set_callback(&m_log_callback);
    log_test_params();
    EXPECT_TRUE(are_params_logged_to(m_callback_message)) << "1st callback got: '" << m_callback_message << "'\n";
    m_callback_message.clear();
    std::string aux_callback_msg;
    LogDispatch::Callback aux_callback = [&aux_callback_msg](std::string_view msg) {
        aux_callback_msg = msg;
    };
    LogDispatch::set_callback(&aux_callback);
    log_test_params();
    EXPECT_TRUE(are_params_logged_to(aux_callback_msg)) << "2st callback got: '" << aux_callback_msg << "'\n";
    EXPECT_TRUE(m_callback_message.empty()) << "Expected no 1st callback. Got: '" << m_callback_message << "'\n";
}

#if ENABLE_LOGGING_TO_STD_COUT_TESTS
TEST_P(TestLogHelper, reset) {
    LogDispatch::set_callback(&m_log_callback);
    LogDispatch::reset_callback();
    log_test_params();
    EXPECT_TRUE(are_params_logged_to(m_mock_out_stream.str()))
        << "Mock cout got: '" << m_mock_out_stream.str() << "'\n";
    EXPECT_TRUE(m_callback_message.empty()) << "Expected no callback. Got: '" << m_callback_message << "'\n";
}
#endif

TEST_P(TestLogHelper, no_log) {
    LogDispatch::set_callback(&m_log_callback);
    LogDispatch::set_callback(nullptr);
    log_test_params();
    EXPECT_TRUE(m_callback_message.empty()) << "Expected no callback. Got: '" << m_callback_message << "'\n";

    LogDispatch::set_callback(&m_log_callback);
    auto empty_callback = LogDispatch::Callback{};
    LogDispatch::set_callback(&empty_callback);
    log_test_params();
    EXPECT_TRUE(m_callback_message.empty()) << "Expected no callback. Got: '" << m_callback_message << "'\n";
}

INSTANTIATE_TEST_SUITE_P(Logging,
                         TestLogHelper,
                         ::testing::ValuesIn(std::vector<LogEntries>{
                             {LOG_TYPE::_LOG_TYPE_ERROR, "path_1", 1, "text 1"},
                             {LOG_TYPE::_LOG_TYPE_WARNING, "path_2", 2, "text 2"},
                             {LOG_TYPE::_LOG_TYPE_INFO, "path_3", 3, "text 3"},
                             {LOG_TYPE::_LOG_TYPE_DEBUG, "path_4", 4, "text 4"},
                             {LOG_TYPE::_LOG_TYPE_DEBUG_EMPTY, "path_5", 5, "text 5"},
                         }));
}  // namespace ov::util::test
