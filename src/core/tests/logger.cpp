// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <iostream>
#include <regex>
#include <sstream>

#include "openvino/core/log.hpp"
#include "openvino/core/log_util.hpp"
#include "openvino/util/log.hpp"

namespace ov::tests {

using namespace ov::util;

// Capturing std::cout streambuffer doesn't work in a CI job on Windows, so disabled. Tested locally on Ubuntu.
static constexpr bool enable_logging_to_std_cout_test = false;

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
        reset_log_callback();
        if (enable_logging_to_std_cout_test) {
            actual_out_stream->flush();
            actual_out_stream->rdbuf(m_mock_out_stream.rdbuf());
        }

        std::tie(m_log_type, m_log_path, m_log_line, m_log_message) = GetParam();
    }

    void TearDown() override {
        if (enable_logging_to_std_cout_test) {
            actual_out_stream->rdbuf(actual_out_buf);
        }
        reset_log_callback();
    }

    auto log_test_params() {
        LogHelper{m_log_type, m_log_path, m_log_line}.stream() << m_log_message;
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
    const LogCallback m_log_callback{[this](std::string_view msg) {
        m_callback_message = msg;
    }};
};

TEST_P(TestLogHelper, set_callback) {
    if (enable_logging_to_std_cout_test) {
        log_test_params();
        EXPECT_TRUE(are_params_logged_to(m_mock_out_stream.str()))
            << "Mock cout got: '" << m_mock_out_stream.str() << "'\n";
        m_mock_out_stream.str("");
        m_mock_out_stream.clear();
    }
    set_log_callback(m_log_callback);
    log_test_params();
    if (enable_logging_to_std_cout_test) {
        EXPECT_TRUE(m_mock_out_stream.str().empty()) << "Expected no cout. Got: '" << m_mock_out_stream.str() << "'\n";
    }
    EXPECT_TRUE(are_params_logged_to(m_callback_message)) << "Callback got: '" << m_callback_message << "'\n";
}

TEST_P(TestLogHelper, toggle_callbacks) {
    set_log_callback(m_log_callback);
    log_test_params();
    EXPECT_TRUE(are_params_logged_to(m_callback_message)) << "1st callback got: '" << m_callback_message << "'\n";
    m_callback_message.clear();
    std::string aux_callback_msg;
    const LogCallback aux_callback = [&aux_callback_msg](std::string_view msg) {
        aux_callback_msg = msg;
    };
    set_log_callback(aux_callback);
    log_test_params();
    EXPECT_TRUE(are_params_logged_to(aux_callback_msg)) << "2st callback got: '" << aux_callback_msg << "'\n";
    EXPECT_TRUE(m_callback_message.empty()) << "Expected no 1st callback. Got: '" << m_callback_message << "'\n";
}

TEST_P(TestLogHelper, reset) {
    set_log_callback(m_log_callback);
    reset_log_callback();
    log_test_params();

    if (enable_logging_to_std_cout_test) {
        EXPECT_TRUE(are_params_logged_to(m_mock_out_stream.str()))
            << "Mock cout got: '" << m_mock_out_stream.str() << "'\n";
    }
    EXPECT_TRUE(m_callback_message.empty()) << "Expected no callback. Got: '" << m_callback_message << "'\n";
}

TEST_P(TestLogHelper, no_log) {
    set_log_callback(m_log_callback);
    const auto empty_callback = LogCallback{};
    set_log_callback(empty_callback);
    ASSERT_NO_THROW(log_test_params());
    EXPECT_TRUE(m_callback_message.empty()) << "Expected no callback. Got: '" << m_callback_message << "'\n";
}

INSTANTIATE_TEST_SUITE_P(Log_callback,
                         TestLogHelper,
                         ::testing::Values(LogEntries{LOG_TYPE::_LOG_TYPE_ERROR, "path_1", 1, "text 1"},
                                           LogEntries{LOG_TYPE::_LOG_TYPE_WARNING, "path_2", 2, "text 2"},
                                           LogEntries{LOG_TYPE::_LOG_TYPE_INFO, "path_3", 3, "text 3"},
                                           LogEntries{LOG_TYPE::_LOG_TYPE_DEBUG, "path_4", 4, "text 4"},
                                           LogEntries{LOG_TYPE::_LOG_TYPE_DEBUG_EMPTY, "path_5", 5, "text 5"}));
}  // namespace ov::tests
